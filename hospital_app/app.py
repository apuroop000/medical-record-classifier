import streamlit as st
import pandas as pd
import os
from pathlib import Path

st.set_page_config(
    page_title="Intelligent Medical Record Router — Pipeline Monitor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1e293b; border-radius: 8px; color: #cbd5e1; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
    .status-badge { padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85em; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = "data"
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DROPZONE_DIR = os.path.join(BASE_DIR, "incoming_dropzone")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
AUDIT_LOG_PATH = "audit_log.xlsx"

st.markdown("# 🏥 Intelligent Medical Record Router")
st.markdown("##### Autonomous Pipeline Monitor — files are processed by `robot_worker.py` running in the background. This dashboard is a *read-only* window into that pipeline.")

# --------------------------------------------------------------------------
# LIVE STATUS STRIP — proves the robot is doing the work, not a person
# --------------------------------------------------------------------------

def count_files_in(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for _, _, files in os.walk(folder) for f in files if not f.startswith("."))


col1, col2, col3, col4 = st.columns(4)
col1.metric("📥 Waiting in Dropzone", count_files_in(DROPZONE_DIR))
col2.metric("📁 Filed in Patient Storage", count_files_in(STORAGE_DIR))
col3.metric("🗄️ Archived Originals", count_files_in(ARCHIVE_DIR))

log_exists = os.path.exists(AUDIT_LOG_PATH)
if log_exists:
    last_modified = pd.Timestamp(os.path.getmtime(AUDIT_LOG_PATH), unit="s")
    col4.metric("🕒 Last Robot Activity", last_modified.strftime("%H:%M:%S"))
else:
    col4.metric("🕒 Last Robot Activity", "No log yet")

st.caption(
    "These numbers update from disk and the audit log only — nothing here depends on this "
    "dashboard being open. Close it, drop a file in `data/incoming_dropzone/`, and reopen "
    "later: the counts and log below will already reflect what the robot did while you were away."
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📋 Audit Log (Robot Activity)",
    "📁 Patient Folder Browser",
    "🧪 Manual Test Upload (optional)"
])

# --------------------------------------------------------------------------
# TAB 1 — AUDIT LOG, straight from audit_log.xlsx
# --------------------------------------------------------------------------

with tab1:
    st.subheader("Robot Worker Activity Log")

    if not log_exists:
        st.info("ℹ️ No audit_log.xlsx yet. Start `robot_worker.py` and drop a file into the dropzone to generate the first entry.")
    else:
        df = pd.read_excel(AUDIT_LOG_PATH, sheet_name="Audit Log")
        if df.empty:
            st.info("ℹ️ Audit log exists but has no entries yet.")
        else:
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                status_filter = st.selectbox("Filter by Status", ["All"] + sorted(df["Status"].dropna().unique().tolist()))
            with colf2:
                dept_filter = st.selectbox("Filter by Department", ["All"] + sorted(df["Department"].dropna().unique().tolist()))
            with colf3:
                patient_filter = st.text_input("Search Patient Name")

            filtered = df.copy()
            if status_filter != "All":
                filtered = filtered[filtered["Status"] == status_filter]
            if dept_filter != "All":
                filtered = filtered[filtered["Department"] == dept_filter]
            if patient_filter:
                filtered = filtered[filtered["Patient Name"].str.contains(patient_filter, case=False, na=False)]

            needs_review = filtered[filtered["Status"] == "Needs Manual Review"]
            if not needs_review.empty:
                st.warning(f"⚠️ {len(needs_review)} document(s) need manual review — low-confidence classification.")

            st.dataframe(
                filtered.sort_values("Timestamp", ascending=False),
                use_container_width=True,
                height=420
            )
            st.caption(f"Showing {len(filtered)} of {len(df)} total processed documents.")

# --------------------------------------------------------------------------
# TAB 2 — BROWSE THE ACTUAL FOLDER STRUCTURE THE ROBOT BUILT
# --------------------------------------------------------------------------

with tab2:
    st.subheader("📁 Patient Folder Structure (as filed by the robot)")

    if not os.path.isdir(STORAGE_DIR):
        st.info("ℹ️ No patient storage folder yet — it's created automatically on the first file the robot files.")
    else:
        patients = sorted([p for p in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, p))])
        if not patients:
            st.info("ℹ️ No patients filed yet.")
        else:
            selected_patient = st.selectbox("Select Patient", patients)
            patient_path = os.path.join(STORAGE_DIR, selected_patient)

            departments = sorted([d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))])
            st.markdown(f"### `{selected_patient}` — {len(departments)} department folder(s)")

            for dept in departments:
                dept_path = os.path.join(patient_path, dept)
                files = sorted(os.listdir(dept_path))
                with st.expander(f"📂 {dept} ({len(files)} file(s))"):
                    for fname in files:
                        fpath = os.path.join(dept_path, fname)
                        colx, coly = st.columns([4, 1])
                        colx.write(f"📄 {fname}")
                        try:
                            with open(fpath, "rb") as f:
                                coly.download_button(
                                    "Download", data=f.read(), file_name=fname,
                                    key=f"dl_{selected_patient}_{dept}_{fname}"
                                )
                        except Exception:
                            coly.caption("unreadable")

# --------------------------------------------------------------------------
# TAB 3 — OPTIONAL MANUAL UPLOAD, clearly marked as a side-door for testing
# --------------------------------------------------------------------------

with tab3:
    st.subheader("🧪 Manual Test Upload")
    st.caption(
        "This is **not** part of the production pipeline. It exists only so you can drop a "
        "single test file without touching the filesystem. The correct way to feed the system "
        "is placing files into `data/incoming_dropzone/` — the same thing this button does under "
        "the hood — which is what a scanner or an email-attachment saver would do automatically."
    )

    uploaded_file = st.file_uploader("Drop a test file into the dropzone", type=["pdf", "docx", "png", "jpg", "jpeg"])
    if uploaded_file and st.button("Place in Dropzone"):
        os.makedirs(DROPZONE_DIR, exist_ok=True)
        dest = os.path.join(DROPZONE_DIR, uploaded_file.name)
        with open(dest, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"✅ Placed `{uploaded_file.name}` into the dropzone. The robot worker will pick it up on its next poll (within ~5 seconds) — no further action needed here.")
