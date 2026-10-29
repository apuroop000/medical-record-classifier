"""
Supervisory dashboard for the unattended RPA robot.

Reads data/rpa.db (written by scripts/robot_worker.py). The only thing it ever writes
is a *review decision* (and files dropped into the dropzone); the bot does all the filing.

Run from the project root (same folder the worker runs in):
    streamlit run hospital_app/app.py
"""
import io
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(os.environ.get("RPA_BASE_DIR", "data"))
DB_PATH = BASE_DIR / "rpa.db"
DROPZONE_DIR = BASE_DIR / "incoming_dropzone"
PROCESSING_DIR = BASE_DIR / "processing"

OFFLINE_AFTER_SECONDS = 60
REFRESH_SECONDS = 5
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Full-width widgets: new Streamlit uses width="stretch", older versions use use_container_width
try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except ValueError:
    _ver = (0, 0)
FULL = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

# Keep in sync with DEPARTMENT_MAP in robot_worker.py (move both to config.yaml later)
DOC_TYPES = ["X-Ray Report", "ECG Report", "CBC Report", "Prescription",
             "Discharge Summary", "Other / General"]

st.set_page_config(page_title="Medical Record RPA Control Room", page_icon="🏥", layout="wide")
st.markdown("""
    <h1 style='color:#1f4e78;margin-bottom:0;'>🏥 Medical Record RPA Control Room</h1>
    <p style='font-size:1.05rem;color:#555;margin-top:0;'>Unattended bot monitoring, human review queue and patient folders</p>
    <hr style='margin-top:0;'>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def query(sql: str, params=()) -> pd.DataFrame:
    """Read-only query. Returns an empty frame if the DB or table does not exist yet."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with closing(sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10)) as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def clean(v, default=""):
    return default if v is None or (isinstance(v, float) and pd.isna(v)) else v


def count_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".part"))


def ago(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def queue_review(audit_id: int, patient_name: str, document_type: str) -> bool:
    """Record a reviewer's decision. The bot picks it up on its next cycle."""
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as conn, conn:
        already = conn.execute(
            "SELECT 1 FROM review_actions WHERE audit_id = ? AND handled_at IS NULL",
            (audit_id,)).fetchone()
        if already:
            return False
        conn.execute(
            "INSERT INTO review_actions (audit_id, patient_name, document_type, requested_at) "
            "VALUES (?, ?, ?, ?)",
            (audit_id, patient_name, document_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return True


def preview_file(path_str, key: str):
    path = Path(path_str) if path_str else None
    if path is None or not path.exists():
        st.warning("File not found on disk. Was it moved, or is the dashboard running from a different folder than the bot?")
        return
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        st.image(str(path), caption=path.name, **FULL)
    elif suffix == ".txt":
        st.text(path.read_text(encoding="utf-8", errors="ignore")[:3000])
    with open(path, "rb") as f:
        st.download_button("Download file", f.read(), file_name=path.name, key=f"dl_{key}")


def live(fn):
    """Auto-refresh a section every few seconds (needs Streamlit >= 1.37; otherwise static)."""
    if hasattr(st, "fragment"):
        return st.fragment(run_every=REFRESH_SECONDS)(fn)
    return fn


# --------------------------------------------------------------------------
# SIDEBAR: manual intake (drops files in the dropzone; the bot does the rest)
# --------------------------------------------------------------------------
st.session_state.setdefault("uploader_key", 0)
with st.sidebar:
    st.header("Manual intake")
    st.caption("Same as a scanner or lab server dropping a file. The bot classifies and files it.")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    uploads = st.file_uploader("Add documents", type=["pdf", "png", "jpg", "jpeg", "txt"],
                               accept_multiple_files=True,
                               key=f"up_{st.session_state.uploader_key}")
    if uploads and st.button("Send to dropzone", type="primary"):
        DROPZONE_DIR.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            name = Path(up.name).name
            dest = DROPZONE_DIR / name
            n = 1
            while dest.exists():
                dest = DROPZONE_DIR / f"{Path(name).stem}_{n}{Path(name).suffix}"
                n += 1
            tmp = dest.with_name(dest.name + ".part")  # bot ignores .part files
            tmp.write_bytes(up.getvalue())
            os.replace(tmp, dest)
        st.session_state.flash = f"{len(uploads)} file(s) dropped. The bot will pick them up shortly."
        st.session_state.uploader_key += 1
        st.rerun()

tab_overview, tab_review, tab_log, tab_folders = st.tabs(
    ["Overview", "Needs Review", "Audit Log", "Patient Folders"])


# --------------------------------------------------------------------------
# TAB 1: OVERVIEW (bot status + KPIs)
# --------------------------------------------------------------------------
@live
def render_overview():
    hb = query("SELECT * FROM heartbeat WHERE id = 1")
    if hb.empty:
        st.warning("No heartbeat yet. Start the bot with `python scripts/robot_worker.py`.")
    else:
        row = hb.iloc[0]
        age = (datetime.now() - datetime.strptime(row["last_seen"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        if row["state"] == "stopped":
            st.error(f"Bot: STOPPED (last seen {ago(age)})")
        elif age > OFFLINE_AFTER_SECONDS:
            st.error(f"Bot: OFFLINE (last seen {ago(age)}). Check the service and worker.log.")
        elif row["state"] == "processing":
            st.success(f"Bot: ONLINE, processing `{clean(row['current_file'])}`")
        else:
            st.success(f"Bot: ONLINE, idle (heartbeat {ago(age)})")

    df = query("SELECT status, department, duration_sec, timestamp FROM audit_log")
    counts = df["status"].value_counts() if not df.empty else pd.Series(dtype=int)
    filed, review, failed = (int(counts.get(s, 0)) for s in ("Filed", "Needs Review", "Failed"))
    finished = filed + review + failed
    today = datetime.now().strftime("%Y-%m-%d")
    failed_today = int(((df["status"] == "Failed") & df["timestamp"].str.startswith(today)).sum()) if not df.empty else 0
    avg_time = df.loc[df["status"] == "Filed", "duration_sec"].mean() if not df.empty else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filed", filed)
    c2.metric("Awaiting review", review)
    c3.metric("Failed", failed, delta=f"{failed_today} today" if failed_today else None,
              delta_color="inverse")
    c4.metric("Duplicates", int(counts.get("Duplicate", 0)))

    c5, c6, c7 = st.columns(3)
    c5.metric("Success rate", f"{filed / finished:.0%}" if finished else "n/a")
    c6.metric("Avg processing time", f"{avg_time:.1f}s" if pd.notna(avg_time) else "n/a")
    c7.metric("Queue depth", count_files(DROPZONE_DIR) + count_files(PROCESSING_DIR),
              help="Files waiting in the dropzone or currently being processed")

    if filed:
        st.markdown("#### Filed documents by department")
        st.bar_chart(df.loc[df["status"] == "Filed", "department"].value_counts())


with tab_overview:
    render_overview()


# --------------------------------------------------------------------------
# TAB 2: NEEDS REVIEW (human-in-the-loop)
# --------------------------------------------------------------------------
with tab_review:
    pending = query("SELECT * FROM audit_log WHERE status = 'Needs Review' ORDER BY id")
    queued = query("SELECT audit_id FROM review_actions WHERE handled_at IS NULL")
    queued_ids = set(queued["audit_id"]) if not queued.empty else set()

    if pending.empty:
        st.success("Nothing is waiting for review.")
    else:
        st.write(f"**{len(pending)}** document(s) need a human decision.")
        labels = {int(r["id"]): f"#{int(r['id'])}  {r['original_filename']}  ({clean(r['notes'], 'no reason recorded')})"
                  for _, r in pending.iterrows()}
        sel = st.selectbox("Document", list(labels), format_func=labels.get)
        row = pending[pending["id"] == sel].iloc[0]

        left, right = st.columns(2)
        with left:
            st.markdown("#### Preview")
            preview_file(clean(row["storage_path"]), f"review_{sel}")
        with right:
            st.markdown("#### Decision")
            st.caption(f"Why it is here: {clean(row['notes'], 'n/a')}")
            guess = clean(row["document_type"])
            if sel in queued_ids:
                st.info("Decision already submitted. The bot will file it within a few seconds.")
            else:
                with st.form(f"review_form_{sel}"):
                    name = st.text_input("Patient name", value=clean(row["patient_name"]))
                    doc_type = st.selectbox("Document type", DOC_TYPES,
                                            index=DOC_TYPES.index(guess) if guess in DOC_TYPES else 0)
                    submitted = st.form_submit_button("Approve & file", type="primary")
                if submitted:
                    if not name.strip():
                        st.error("Patient name is required.")
                    elif queue_review(int(sel), name.strip(), doc_type):
                        st.success("Submitted. The bot will file it on its next cycle.")
                    else:
                        st.info("A decision for this document is already queued.")

    history = query("SELECT audit_id, patient_name, document_type, requested_at, handled_at, result "
                    "FROM review_actions ORDER BY id DESC LIMIT 10")
    if not history.empty:
        st.markdown("#### Recent review decisions")
        st.dataframe(history, **FULL)


# --------------------------------------------------------------------------
# TAB 3: AUDIT LOG
# --------------------------------------------------------------------------
with tab_log:
    log_df = query("SELECT id, timestamp, original_filename, patient_name, patient_id, document_type, "
                   "department, confidence, method, status, notes, attempts, duration_sec, storage_path "
                   "FROM audit_log ORDER BY id DESC")
    if log_df.empty:
        st.info("No transactions yet.")
    else:
        f1, f2, f3 = st.columns(3)
        statuses = f1.multiselect("Status", sorted(log_df["status"].unique()),
                                  default=sorted(log_df["status"].unique()))
        depts = f2.multiselect("Department", sorted(log_df["department"].dropna().unique()))
        text = f3.text_input("Search patient or filename")

        view = log_df[log_df["status"].isin(statuses)]
        if depts:
            view = view[view["department"].isin(depts)]
        if text.strip():
            t = text.strip().lower()
            view = view[view["patient_name"].fillna("").str.lower().str.contains(t, regex=False)
                        | view["original_filename"].str.lower().str.contains(t, regex=False)]

        st.dataframe(view, **FULL)

        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name="audit_log.csv", mime="text/csv")
        buf = io.BytesIO()
        view.to_excel(buf, index=False, sheet_name="Audit Log")
        d2.download_button("Download Excel", buf.getvalue(), file_name="audit_log.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# --------------------------------------------------------------------------
# TAB 4: PATIENT FOLDERS
# --------------------------------------------------------------------------
with tab_folders:
    filed_df = query("SELECT * FROM audit_log WHERE status = 'Filed' ORDER BY department, id")
    if filed_df.empty:
        st.info("No filed documents yet.")
    else:
        patients = sorted(filed_df["patient_name"].dropna().unique())
        who = st.selectbox("Patient", patients)
        mine = filed_df[filed_df["patient_name"] == who]
        st.success(f"{who}: {len(mine)} document(s) in {mine['department'].nunique()} department(s)")
        for dept, group in mine.groupby("department"):
            with st.expander(f"{dept} ({len(group)})", expanded=True):
                for _, r in group.iterrows():
                    with st.expander(f"{r['original_filename']}  |  {r['document_type']}  |  confidence {r['confidence']}"):
                        st.caption(f"Filed {r['timestamp']}  |  Patient ID: {clean(r['patient_id'], 'N/A')}"
                                   f"  |  {clean(r['notes'])}")
                        preview_file(clean(r["storage_path"]), f"folder_{int(r['id'])}")