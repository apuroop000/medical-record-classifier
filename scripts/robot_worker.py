"""
Unattended RPA robot worker for hospital medical records.

Pipeline per file:
    dropzone -> processing (claim) -> hash/dedupe -> extract -> classify
             -> Filed (storage + archive) | Needs Review | Failed | Duplicate

Run:
    python scripts/robot_worker.py            # run forever
    python scripts/robot_worker.py --once     # process what is waiting, then exit
    python scripts/robot_worker.py --export   # rewrite audit_log.xlsx from the DB, then exit
"""
import argparse
import hashlib
import io
import logging
import os
import re
import shutil
import signal
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG  (move these into config.yaml later)
# --------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("RPA_BASE_DIR", "data"))
DROPZONE_DIR = BASE_DIR / "incoming_dropzone"
PROCESSING_DIR = BASE_DIR / "processing"
STORAGE_DIR = BASE_DIR / "storage"
ARCHIVE_DIR = BASE_DIR / "archive"
DUPLICATES_DIR = BASE_DIR / "duplicates"
NEEDS_REVIEW_DIR = BASE_DIR / "needs_review"
FAILED_DIR = BASE_DIR / "failed"
LOG_DIR = BASE_DIR / "logs"

DB_PATH = BASE_DIR / "rpa.db"
EXCEL_EXPORT_PATH = Path(os.environ.get("RPA_EXCEL_PATH", "audit_log.xlsx"))
EXCEL_BRIDGE = True  # keep audit_log.xlsx updated for the old dashboard; set False once app.py reads SQLite

POLL_INTERVAL_SECONDS = 5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3
REVIEW_CONFIDENCE_THRESHOLD = 0.60
EMPTY_FILE_GRACE_SECONDS = 30
OCR_MIN_WIDTH = 1800  # small screenshots are upscaled to this width before OCR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
IGNORED_SUFFIXES = (".part", ".tmp", ".crdownload")

DEPARTMENT_MAP = {
    "Radiology Report": "Radiology",
    "ECG Report": "Cardiology",
    "Lab Report": "Laboratory",
    "Prescription": "Pharmacy",
    "Discharge Summary": "Medical Records",
    "Pathology Report": "Pathology",
    "Neurology Report": "Neurology",
    "Pulmonology Report": "Pulmonology",
    "Ophthalmology Report": "Ophthalmology",
    "Operative Note": "Surgery",
    "Medical Certificate": "Medical Records",
    "Billing Document": "Billing",
    # names used by earlier versions, kept so old audit rows still map to a department
    "X-Ray Report": "Radiology",
    "CBC Report": "Laboratory",
}

for _d in (DROPZONE_DIR, PROCESSING_DIR, STORAGE_DIR, ARCHIVE_DIR,
           DUPLICATES_DIR, NEEDS_REVIEW_DIR, FAILED_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# LOGGING  (file + console, no emojis so it survives Task Scheduler / services)
# --------------------------------------------------------------------------
log = logging.getLogger("robot")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
_fh = RotatingFileHandler(LOG_DIR / "worker.log", maxBytes=2_000_000,
                          backupCount=5, encoding="utf-8")
_fh.setFormatter(_fmt)
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_ch)

stop_event = threading.Event()


def _request_stop(signum, frame):
    log.info("Stop requested (signal %s). Finishing current file, then exiting.", signum)
    stop_event.set()


# --------------------------------------------------------------------------
# EXCEPTIONS
# --------------------------------------------------------------------------
class BusinessException(Exception):
    """The document itself is the problem. Never retry; route to needs_review/."""


# --------------------------------------------------------------------------
# DATABASE (SQLite is the source of truth; Excel is just an export)
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_hash         TEXT,
    patient_name      TEXT,
    patient_id        TEXT,
    document_type     TEXT,
    department        TEXT,
    confidence        REAL,
    method            TEXT,
    name_source       TEXT,
    storage_path      TEXT,
    status            TEXT NOT NULL,
    notes             TEXT,
    attempts          INTEGER DEFAULT 0,
    duration_sec      REAL
);
CREATE INDEX IF NOT EXISTS idx_audit_hash ON audit_log(file_hash);
CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status);

CREATE TABLE IF NOT EXISTS heartbeat (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    last_seen    TEXT NOT NULL,
    state        TEXT,
    current_file TEXT,
    pid          INTEGER
);

CREATE TABLE IF NOT EXISTS review_actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id      INTEGER NOT NULL,
    patient_name  TEXT NOT NULL,
    document_type TEXT NOT NULL,
    requested_at  TEXT NOT NULL,
    handled_at    TEXT,
    result        TEXT
);
"""

UPDATABLE = {"file_hash", "patient_name", "patient_id", "document_type", "department",
             "confidence", "method", "name_source", "storage_path", "status",
             "notes", "attempts", "duration_sec"}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # dashboard can read while the bot writes
    return conn


def init_db():
    with closing(connect()) as conn, conn:
        conn.executescript(SCHEMA)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def db_insert(filename: str, file_hash: str) -> int:
    with closing(connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO audit_log (timestamp, original_filename, file_hash, status) "
            "VALUES (?, ?, ?, 'Processing')", (now(), filename, file_hash))
        return cur.lastrowid


def db_update(row_id: int, **fields):
    bad = set(fields) - UPDATABLE
    if bad:
        raise ValueError(f"Unknown audit columns: {bad}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with closing(connect()) as conn, conn:
        conn.execute(f"UPDATE audit_log SET {sets} WHERE id = ?", (*fields.values(), row_id))


def is_duplicate(file_hash: str, row_id: int) -> bool:
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT 1 FROM audit_log WHERE file_hash = ? AND id <> ? "
            "AND status IN ('Filed', 'Needs Review') LIMIT 1", (file_hash, row_id)).fetchone()
    return row is not None


def heartbeat(state: str, current_file: str = ""):
    try:
        with closing(connect()) as conn, conn:
            conn.execute(
                "INSERT INTO heartbeat (id, last_seen, state, current_file, pid) "
                "VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen, "
                "state=excluded.state, current_file=excluded.current_file, pid=excluded.pid",
                (now(), state, current_file, os.getpid()))
    except sqlite3.Error as e:
        log.warning("Heartbeat write failed: %s", e)


# --------------------------------------------------------------------------
# EXCEL EXPORT (bridge so the current dashboard keeps working)
# --------------------------------------------------------------------------
AUDIT_COLUMNS = [
    "Timestamp", "Original Filename", "Patient Name", "Patient ID Guess",
    "Document Type", "Department", "Confidence", "Classification Method",
    "Patient Name Source", "Storage Path", "Status", "Notes",
]
LEGACY_STATUS = {"Filed": "Completed & Routed", "Needs Review": "Needs Manual Review"}


def export_excel() -> bool:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    with closing(connect()) as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()

    wb = Workbook()
    ws = wb.active
    ws.title = "Audit Log"
    ws.append(AUDIT_COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in rows:
        ws.append([
            r["timestamp"], r["original_filename"], r["patient_name"], r["patient_id"],
            r["document_type"], r["department"], r["confidence"], r["method"],
            r["name_source"], r["storage_path"],
            LEGACY_STATUS.get(r["status"], r["status"]), r["notes"],
        ])
    tmp = EXCEL_EXPORT_PATH.with_suffix(".tmp.xlsx")
    try:
        wb.save(tmp)
        os.replace(tmp, EXCEL_EXPORT_PATH)  # fails if someone has the file open
        return True
    except PermissionError:
        log.warning("%s is open elsewhere; Excel export skipped (DB is unaffected).",
                    EXCEL_EXPORT_PATH)
        return False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------
# FILE HELPERS
# --------------------------------------------------------------------------
def unique_path(folder: Path, name: str) -> Path:
    p = folder / name
    n = 1
    stem, suffix = Path(name).stem, Path(name).suffix
    while p.exists():
        p = folder / f"{stem}_{n}{suffix}"
        n += 1
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_stable(path: Path, wait: float = 1.0) -> bool:
    """True when the file has stopped growing (the scanner/email is done writing it)."""
    try:
        st1 = path.stat()
        time.sleep(wait)
        st2 = path.stat()
    except FileNotFoundError:
        return False
    if st1.st_size != st2.st_size:
        return False
    if st2.st_size == 0:  # genuinely empty files get picked up after a grace period
        return (time.time() - st2.st_mtime) > EMPTY_FILE_GRACE_SECONDS
    return True


def claim(path: Path):
    """Move the file out of the dropzone so nothing else touches it. None if it is locked."""
    dest = unique_path(PROCESSING_DIR, path.name)
    try:
        shutil.move(str(path), str(dest))
    except (PermissionError, FileNotFoundError, OSError) as e:
        log.info("Could not claim %s yet (%s); will retry next cycle.", path.name, e)
        return None
    return dest


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _.\-]", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")[:60]
    return cleaned or "Unidentified Patient"


def move_to(folder: Path, src: Path, original_name: str) -> Path:
    dest = unique_path(folder, original_name)
    shutil.move(str(src), str(dest))
    return dest


# --------------------------------------------------------------------------
# EXTRACTION
# --------------------------------------------------------------------------
def ocr_image(img) -> str:
    """OCR an image twice (page-layout mode + single-block mode) and join the results.

    Report headers are usually two-column key/value tables. Layout mode sometimes reads
    them column by column ("Patient / Date of birth / ...") while block mode keeps each
    row together ("Patient  Ava Okafor"). Having both in the text lets the name and
    department rules work on whichever reading came out cleaner.
    """
    import pytesseract
    from PIL import Image, ImageOps

    img = ImageOps.grayscale(img)
    w, h = img.size
    if w < OCR_MIN_WIDTH:  # low-resolution screenshots OCR badly unless enlarged
        scale = OCR_MIN_WIDTH / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    texts = []
    for psm in (3, 6):
        try:
            texts.append(pytesseract.image_to_string(img, config=f"--psm {psm}"))
        except pytesseract.TesseractError as e:
            log.warning("Tesseract psm %s failed: %s", psm, e)
    return "\n".join(texts)


def ocr_pdf(doc) -> str:
    from PIL import Image
    parts = []
    for page in doc:
        pix = page.get_pixmap(dpi=250)
        parts.append(ocr_image(Image.open(io.BytesIO(pix.tobytes("png")))))
    return "\n".join(parts)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        import fitz  # PyMuPDF
        with fitz.open(str(path)) as doc:
            text = "".join(page.get_text() for page in doc)
            if len(text.strip()) < 40:  # scanned PDF (or only a stray text layer): OCR it
                text = text + "\n" + ocr_pdf(doc)
        return text
    if ext in IMAGE_EXTS:
        from PIL import Image
        with Image.open(path) as img:
            return ocr_image(img)
    raise BusinessException(f"Unsupported file type '{ext}'")


def normalise_text(text: str) -> str:
    """Remove OCR/table noise that confuses the patterns (pipes, bullets, odd spaces)."""
    text = text.replace("\u00a0", " ").replace("\t", "  ")
    text = re.sub(r"[|¦•·►▪■□]", " ", text)
    text = re.sub(r"[ ]{2,}", "  ", text)
    return text


# --------------------------------------------------------------------------
# CLASSIFICATION (weighted keyword scoring -> real confidence)
# --------------------------------------------------------------------------
# Each pattern adds its weight once if it appears anywhere in the text or file name.
BODY_PARTS = (r"\b(?:chest|elbow|knee|shoulder|spine|pelvis|skull|wrist|ankle|hip|abdomen|"
              r"femur|humerus|tibia|fibula|clavicle|foot|hand|neck|sinus|liver|kidney|"
              r"gall\s*bladder|pancreas|spleen|prostate|uterus|thyroid)\b")
DOC_RULES = {
    "Radiology Report": [
        (r"\bx[\s\-]*ray\b", 3), (r"radiograph", 3),
        (r"ultrasound|ultrasonograph|sonograph|sonolog|\busg\b|mammogra|computed tomography|"
         r"magnetic resonance|\bct[\s\-]*scan\b|\bmri\b|doppler|\bdexa\b", 3),
        (r"\bradiolog", 3),
        (r"echotexture|echogenic|hypoechoic|hyperechoic|anechoic", 2),
        (r"\b(?:pa|ap|lateral|oblique) view\b|\b(?:ap|pa)\s*/\s*lateral\b", 2),
        (r"\bimpression\b", 1),
        (r"fracture|osteolytic|sclerotic|periosteal|soft tissue|opacity|consolidation|effusion|"
         r"calculus|hydronephrosis|abnormality", 1),
        (BODY_PARTS, 1),
    ],
    "ECG Report": [
        (r"\be[ck]g\b", 3), (r"electrocardiogram", 3), (r"\bqrs\b", 2),
        (r"sinus rhythm|pr interval|qt interval", 2),
        (r"echocardiogra|\b2d echo\b|\btmt\b|treadmill|holter", 2), (r"heart rate", 1),
    ],
    "Lab Report": [
        (r"\bcbc\b", 3), (r"complete blood count", 3),
        (r"lipid profile|liver function|kidney function|renal function|thyroid profile|"
         r"blood sugar|glucose|hba1c|hemoglobin a1c|urinalysis|urine routine|serum", 3),
        (r"h(?:a)?emoglobin", 2), (r"\b(?:wbc|rbc|platelet)|white blood cell|red blood cell", 2),
        (r"cholesterol|triglycerid|creatinine|bilirubin|\bsgpt\b|\bsgot\b|\btsh\b|\burea\b|"
         r"ferritin|hematocrit|haematocrit|\bmcv\b|\bmch\b|sodium|potassium", 2),
        (r"reference (?:interval|range)|biological reference|test parameter|observed value", 2),
        (r"specimen|sample collected|collected on|accession", 1),
        (r"pathology|pathologist|blood test|haematology|hematology|biochemistry", 1),
        (r"\b(?:mg|g)\s*/\s*dl\b|/cu\.?\s?mm|million\s*/|mmol\s*/\s*l|x10", 1),
    ],
    "Prescription": [
        (r"\bprescription\b", 3), (r"\brx\b|℞", 2), (r"\b(?:tablet|capsule|syrup)s?\b", 1),
        (r"\b\d+\s?mg\b(?!\s*/)", 1), (r"\b(?:dosage|sig)\b", 1),
        (r"\b(?:bd|tds|qid|sos)\b|\b\d\s*-\s*\d\s*-\s*\d\b", 2),
    ],
    "Discharge Summary": [
        (r"discharge summary", 4), (r"\bdischarge\b", 2),
        (r"date of (?:discharge|admission)", 2), (r"\badmission\b", 1),
        (r"hospital course|condition (?:at|on) discharge|discharge (?:advice|medication|instructions)", 2),
    ],
    "Pathology Report": [
        (r"histopatholog|biopsy|cytolog|\bfnac\b|pap smear", 4),
        (r"gross description|microscopic (?:examination|description)|tumou?r|carcinoma|malignan", 2),
    ],
    "Neurology Report": [
        (r"\beeg\b|electroencephalo", 3), (r"\bemg\b|\bncv\b|nerve conduction", 3),
        (r"neurolog", 2), (r"seizure|epileps", 1),
    ],
    "Pulmonology Report": [
        (r"spirometry|\bpft\b|pulmonary function", 3), (r"\bfev1?\b|\bfvc\b", 3),
        (r"peak flow|pulmonolog|bronchodilator", 2),
    ],
    "Ophthalmology Report": [
        (r"visual acuity|intraocular|fundus|ophthalm", 3),
        (r"cornea|retina|refraction|spherical|cylindrical|\bipd\b", 2),
    ],
    "Operative Note": [
        (r"operative (?:note|report)|operation notes?|procedure performed", 4),
        (r"surgeon|anaesthe|anesthe|post[\s\-]?operative|pre[\s\-]?operative|incision", 2),
    ],
    "Medical Certificate": [
        (r"medical certificate|fitness certificate|certificate of fitness|sick leave|"
         r"to whom it may concern|certif(?:y|ied) that", 3),
    ],
    "Billing Document": [
        (r"\binvoice\b|\bbill (?:no|number)\b|\breceipt\b|\bgst(?:in)?\b", 3),
        (r"amount (?:paid|due)|total amount|net payable|payment mode|grand total", 2),
    ],
}

# A short line that STARTS with one of these is a document title ("RADIOLOGY", "Haematology Report").
# Titles are the strongest hint there is, so they add a bonus. Letterhead lines such as
# "CarePoint Pathology" do not start with these words, so they do not trigger it.
HEADING_RULES = {
    "Radiology Report": r"(?:radiology|x[\s\-]*ray|ultrasound|ultrasonograph|sonograph|usg|ct[\s\-]*scan|mri|mammogra|doppler)\b",
    "Lab Report": r"(?:haematology|hematology|biochemistry|microbiology|serology|clinical pathology|"
                  r"complete blood count|lipid profile|liver function|kidney function|thyroid profile|urinalysis|cbc)\b",
    "ECG Report": r"(?:e[ck]g|electrocardiogram|echocardiogra|2d echo|tmt|treadmill|cardiology)\b",
    "Prescription": r"(?:rx|℞|prescription)\b",
    "Discharge Summary": r"discharge (?:summary|card)\b",
    "Pathology Report": r"(?:histopathology|cytology|biopsy)\b",
    "Neurology Report": r"(?:eeg|emg|nerve conduction|neurology)\b",
    "Pulmonology Report": r"(?:pulmonary function|spirometry|pulmonology)\b",
    "Ophthalmology Report": r"(?:ophthalmology|eye examination)\b",
    "Operative Note": r"operat(?:ive|ion) (?:note|report)s?\b",
    "Medical Certificate": r"(?:medical|fitness) certificate\b",
    "Billing Document": r"(?:invoice|receipt|bill of supply|tax invoice)\b",
}

# Last resort when no keyword rule fires: read the department straight off the page
# ("Department of Nephrology", "Dept: Orthopaedics").
DEPT_HEADER_RE = re.compile(
    r"(?:department\s+of|dept\.?\s*of|department\s*[:\-]|dept\.?\s*[:\-])\s*([A-Za-z][A-Za-z &]{2,40})",
    re.IGNORECASE)
DEPT_STOP = {"report", "date", "name", "id", "ward", "doctor", "dr", "phone", "patient", "no", "number",
             "of", "and", "the"}


def _department_from_text(text: str):
    for m in DEPT_HEADER_RE.finditer(text):
        words = []
        for w in m.group(1).split():
            if w.lower() in DEPT_STOP and words:
                break
            if w.lower() in DEPT_STOP:
                continue
            words.append(w)
            if len(words) == 3:
                break
        dept = " ".join(words).title().strip(" &")
        if len(dept) >= 4:
            return dept
    return None


def classify_document(filename: str, doc_text: str):
    """Returns (document_type, confidence, method, department)."""
    text = doc_text.lower()
    haystack = f"{text}\n{Path(filename).stem.replace('_', ' ').replace('-', ' ')}".lower()
    scores = {t: sum(w for pattern, w in rules if re.search(pattern, haystack))
              for t, rules in DOC_RULES.items()}

    lines = [ln.strip(" \t:-*#_.") for ln in text.splitlines()]
    short_lines = [ln for ln in lines if 0 < len(ln) <= 80]
    for doc_type, pattern in HEADING_RULES.items():
        if any(re.match(pattern, ln) for ln in short_lines):
            scores[doc_type] += 4

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best_type, best), (_, second) = ranked[0], ranked[1]
    if best == 0:
        dept = _department_from_text(doc_text)
        if dept:
            return f"{dept} Report", 0.62, "department_header", dept
        return "Unclassified", 0.0, "fallback", "General"
    confidence = min(0.97, 0.40 + 0.12 * best)
    if second >= 0.7 * best:  # two types almost tied -> ambiguous
        confidence -= 0.20
    return (best_type, round(max(confidence, 0.0), 2), "keyword_score",
            DEPARTMENT_MAP.get(best_type, "General"))


# --------------------------------------------------------------------------
# PATIENT NAME & ID
# --------------------------------------------------------------------------
# Strategy, most reliable first:
#   1. a labelled value   (Patient Name: Mr. John Doe / "Patient   Ava Okafor" / "Name : Mr. X")
#   2. position           (the name line just above the Age / Sex lines of the header block)
#   3. an honorific       (Mrs. Priya Nair, anywhere in the text)
#   4. the file name, only if it looks like a real name
# Anything else -> "Unidentified Patient" (name_source "none"), which goes to human review.
TITLE_RE = re.compile(r"^(?:(?:mr|mrs|ms|miss|mx|master|baby|smt|shri|sri|shrimati|kum|km)\b\.?\s*)+", re.IGNORECASE)

# (label regex, colon required)
NAME_LABELS = [
    (r"patient['’]?s?\s*name", False),
    (r"name\s+of\s+(?:the\s+)?patient", False),
    (r"\bpt\.?\s*name", False),
    (r"client\s*name", False),
    (r"\bpatient", True),
    (r"\bname", True),
]
# Bare labels with NO colon ("Patient   Ava Okafor"). Only accepted at the start of a line
# and only when the value is Capitalised or UPPERCASE, so ordinary sentences are not matched.
BARE_LABELS = [r"patient", r"pt\.?", r"client", r"name"]
CAP_VALUE = r"([A-Z][A-Za-z.'’]*(?:[ \t]+[A-Z][A-Za-z.'’]*){0,4})"

# A label is ignored when one of these appears just before it (e.g. "Doctor Name:", "Father's Name:").
NOT_THE_PATIENT = ("doctor", "dr.", "consultant", "referr", "physician", "hospital", "clinic", "centre",
                   "center", "pathologist", "radiologist", "technolog", "sample", "specimen", "father",
                   "mother", "husband", "wife", "guardian", "relative", "nominee", "emergency", "lab ",
                   "company", "insurance", "ordered")
# Text that ends a name when OCR joins two table columns on one line ("... Doe Report Date: ...").
CUT_RE = re.compile(
    r"\b(?:age|sex|gender|dob|date|birth|id|uhid|mrn|pid|ref|referred|report|reported|registered|"
    r"registration|collection|collected|received|ordered|accession|final|status|lab|sample|"
    r"phone|mobile|contact|address|ward|bed|room|visit|bill|invoice|apt|appointment|"
    r"doctor|dr|consultant|years?|yrs?|clinic|patient|barcode|qr)\b", re.IGNORECASE)
NAME_STOPWORDS = {
    "report", "reports", "hospital", "clinic", "centre", "center", "diagnostics", "diagnostic",
    "laboratory", "laboratories", "imaging", "pathology", "radiology", "patient", "name", "test", "tests",
    "parameter", "value", "unit", "page", "date", "time", "address", "phone", "email", "road", "street",
    "sector", "boulevard", "complex", "not", "na", "n/a", "unknown", "nil", "none", "x", "ray", "xray",
    "ecg", "cbc", "scan", "view", "impression", "findings", "summary", "discharge", "prescription",
    "signature", "doctor", "dr", "thanks", "reference", "generated", "verified", "referred", "registered",
    "reported", "lipid", "profile", "blood", "count", "complete", "male", "female", "sex", "age", "years",
    "gender", "mr", "mrs", "ms", "miss", "master", "informational", "purposes", "medical", "health",
    "care", "department", "lab", "ct", "mri", "usg", "elbow", "joint", "chest",
    "information", "details", "detail", "info", "history", "data", "type", "record", "records",
    "copy", "portal", "synthetic", "fictional", "final", "ordered", "accession", "result", "results",
    "ultrasound", "abdomen", "whole", "scanning", "pathologist", "technician", "emergency", "hours",
    "sunrise", "apartments", "chennai", "india", "comment", "note", "self", "id", "no", "number",
}
NAME_LINE_RE = re.compile(r"^[A-Z][A-Za-z'’]+(?:\s+(?:[A-Z]\.?|[A-Z][A-Za-z'’]+)){1,3}$")
HONORIFIC_RE = re.compile(
    r"(?<![A-Za-z.])(?:Mr|MR|Mrs|MRS|Ms|MS|Miss|MISS|Master|MASTER|Smt|SMT|Shri|SHRI|Baby|BABY)\.?[ \t]+"
    r"([A-Z][A-Za-z'’]+(?:[ \t]+[A-Z][A-Za-z'’.]*){0,3})")

ID_PATTERNS = [
    r"(?:patient\s*(?:id|no\.?|number)|\bp\.?\s?id\b|\buhid\b|\bmrn\b|\bmr\s*no\.?|"
    r"\breg(?:istration)?\.?\s*(?:no\.?|number|id)|\b(?:ip|op)\s*(?:no\.?|number|id)|\bid\s*no\.?)"
    r"\s*[:#\-]?\s*((?=[A-Za-z0-9\-/]*\d)[A-Za-z0-9][A-Za-z0-9\-/]{1,19})",
]
FILENAME_NOISE_WORDS = {
    "blood", "test", "report", "xray", "x-ray", "ecg", "ekg", "prescription", "scan", "scanned", "pdf",
    "jpg", "jpeg", "png", "cbc", "discharge", "summary", "image", "img", "screenshot", "document", "doc",
    "file", "copy", "final", "new", "lab", "mri", "ct", "usg", "photo", "page", "sample",
    "notes", "note", "untitled", "download", "whatsapp", "pic", "picture", "at", "am", "pm",
    "scanner", "capture", "snip", "report", "record", "medical", "patient",
}


def _clean_name(raw: str):
    """Turn a captured string into a patient name, or None if it is not plausibly one."""
    s = CUT_RE.split(raw)[0]
    s = re.sub(r"[^A-Za-z.'’ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .:-'’")
    s = TITLE_RE.sub("", s).strip(" .")
    tokens = s.split()
    if not 1 <= len(tokens) <= 5:
        return None
    if len(s.replace(" ", "").replace(".", "")) < 3:
        return None
    for t in tokens:
        if t.lower().strip(".") in NAME_STOPWORDS:
            return None
    return " ".join(tokens).title()


def _compile_label_patterns():
    pats = []
    for label, colon_required in NAME_LABELS:
        sep = r"[ \t]*[:\-–][ \t]*" if colon_required else r"[ \t]*[:\-–]?[ \t]*"
        pats.append((re.compile(label + sep + r"(?:\r?\n[ \t]*)?([A-Za-z][A-Za-z.'’ ]{1,60})",
                                re.IGNORECASE), True))
    for label in BARE_LABELS:
        pats.append((re.compile(r"^[ \t]*(?i:" + label + r")\b[ \t]*[:\-–]?[ \t]*(?:\r?\n[ \t]*)?" + CAP_VALUE,
                                re.MULTILINE), False))
    return pats


LABEL_PATTERNS = _compile_label_patterns()


def _name_from_label(text: str):
    for pattern, check_before in LABEL_PATTERNS:
        for m in pattern.finditer(text):
            if check_before:  # same line only
                before = text[max(0, m.start() - 20):m.start()].rsplit("\n", 1)[-1].lower()
                if any(w in before for w in NOT_THE_PATIENT):
                    continue
            name = _clean_name(m.group(m.lastindex))
            if name:
                return name
    return None


STACKED_LABEL_RE = re.compile(
    r"^(?:patient(?:['’]?s)?(?:\s*name)?|name|date of birth|dob|age|sex|gender|age\s*/\s*gender|mrn|uhid|"
    r"(?:patient\s*)?id|address|phone|mobile|ordered by|referred by|clinic|collected|reported|"
    r"report(?:ed)? date|collection date|registration date|received date|report id|report status)\s*:?$",
    re.IGNORECASE)


def _name_from_stacked(text: str):
    """OCR sometimes reads a header table column by column: a block of labels
    (Patient / Date of birth / MRN ...) followed by a block of values in the same order."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if not re.match(r"^(?:patient(?:['’]?s)?(?:\s*name)?|name)\s*:?$", line, re.IGNORECASE):
            continue
        start = i
        while start > 0 and STACKED_LABEL_RE.match(lines[start - 1]):
            start -= 1
        end = i
        while end + 1 < len(lines) and STACKED_LABEL_RE.match(lines[end + 1]):
            end += 1
        idx = i - start
        if end == i:  # nothing stacked below the label: the plain label rules already covered it
            continue
        values = lines[end + 1:end + 1 + (end - start + 1)]
        if idx < len(values):
            name = _clean_name(values[idx])
            if name:
                return name
    return None


def _name_from_position(text: str):
    """Unlabelled header block: the name sits on the line just above 'Age' / 'Sex'."""
    lines = [ln.strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        m = re.search(r"\bage\b\s*[:/\-]", line, re.IGNORECASE)
        if not m:
            continue
        window = lines[i:i + 3]
        if not any(re.search(r"\b(?:sex|gender)\b", w, re.IGNORECASE) for w in window) and \
                not re.search(r"\b(?:sex|gender)\b", line, re.IGNORECASE):
            continue
        candidates = [line[:m.start()].strip()]  # same line, before 'Age'
        k = i - 1
        while k >= 0 and len(candidates) < 3:  # at most the two non-blank lines above
            if lines[k]:
                # the first two non-blank lines of a document are normally the letterhead
                if sum(1 for x in lines[:k] if x) >= 2:
                    candidates.append(lines[k])
            k -= 1
        for cand in candidates:
            if cand and NAME_LINE_RE.match(cand):
                name = _clean_name(cand)
                if name:
                    return name
    return None


def _name_from_honorific(text: str):
    for m in HONORIFIC_RE.finditer(text):
        name = _clean_name(m.group(1))
        if name:
            return name
    return None


def _name_from_filename(filename: str):
    """Only accept a file name that looks like a person's name (letters only, up to 4 words)."""
    tokens = [t for t in re.split(r"[_\-\s.]+", Path(filename).stem) if t]
    tokens = [t for t in tokens if t.lower() not in FILENAME_NOISE_WORDS and not t.isdigit()]
    if 1 <= len(tokens) <= 4 and all(t.isalpha() and len(t) >= 2 for t in tokens):
        return " ".join(tokens).title()
    return None


def extract_patient_name_and_id(doc_text: str, filename: str):
    name = _name_from_label(doc_text)
    source = "ocr_regex"
    if not name:
        name, source = _name_from_stacked(doc_text), "ocr_stacked"
    if not name:
        name, source = _name_from_position(doc_text), "ocr_position"
    if not name:
        name, source = _name_from_honorific(doc_text), "ocr_honorific"
    if not name:
        name, source = _name_from_filename(filename), "filename_guess"
    if not name:
        name, source = "Unidentified Patient", "none"

    patient_id = "N/A"
    for pattern in ID_PATTERNS:
        m = re.search(pattern, doc_text, re.IGNORECASE)
        if m:
            patient_id = m.group(1).upper()
            break
    return name, patient_id, source


# --------------------------------------------------------------------------
# ANALYSE = the retryable part (extract + classify + parse)
# --------------------------------------------------------------------------
def analyse(path: Path, original_name: str) -> dict:
    text = normalise_text(extract_text(path))
    if not text.strip():
        raise BusinessException("No readable text found (blank or unreadable scan)")
    patient_name, patient_id, name_source = extract_patient_name_and_id(text, original_name)
    doc_type, confidence, method, department = classify_document(original_name, text)
    return {
        "patient_name": patient_name, "patient_id": patient_id, "name_source": name_source,
        "document_type": doc_type, "confidence": confidence, "method": method,
        "department": department,
    }


def run_with_retries(fn):
    """Retry system errors only. BusinessException propagates immediately."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(), attempt
        except BusinessException:
            raise
        except Exception as e:
            log.warning("System error on attempt %d/%d: %s", attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                e.attempts = attempt  # type: ignore[attr-defined]
                raise
            if stop_event.wait(RETRY_DELAY_SECONDS):
                raise


# --------------------------------------------------------------------------
# PER-FILE WORKFLOW
# --------------------------------------------------------------------------
def process_file(src: Path) -> bool:
    """Returns True if the audit log changed."""
    if not is_stable(src):
        log.info("%s is still being written; skipping this cycle.", src.name)
        return False
    original_name = src.name
    claimed = claim(src)
    if claimed is None:
        return False

    started = time.time()
    log.info("Claimed: %s", original_name)
    file_hash = sha256_file(claimed)
    row_id = db_insert(original_name, file_hash)
    attempts = 0
    try:
        if is_duplicate(file_hash, row_id):
            dest = move_to(DUPLICATES_DIR, claimed, original_name)
            db_update(row_id, status="Duplicate", storage_path=str(dest),
                      notes="Identical content was already processed")
            log.info("Duplicate: %s", original_name)
            return True

        result, attempts = run_with_retries(lambda: analyse(claimed, original_name))

        reasons = []
        if result["confidence"] < REVIEW_CONFIDENCE_THRESHOLD:
            reasons.append(f"Low confidence ({result['confidence']})")
        if result["name_source"] == "none":
            reasons.append("Patient name could not be identified")

        if reasons:
            dest = move_to(NEEDS_REVIEW_DIR, claimed, original_name)
            db_update(row_id, status="Needs Review", storage_path=str(dest),
                      notes="; ".join(reasons), attempts=attempts, **result)
            log.info("Needs review: %s (%s)", original_name, "; ".join(reasons))
            return True

        dest_folder = STORAGE_DIR / safe_name(result["patient_name"]) / result["department"]
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest = unique_path(dest_folder, original_name)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copy2(claimed, tmp)
        os.replace(tmp, dest)  # file appears in storage only when fully copied
        shutil.move(str(claimed), str(unique_path(ARCHIVE_DIR, original_name)))

        note = "Patient name guessed from filename" if result["name_source"] == "filename_guess" else ""
        db_update(row_id, status="Filed", storage_path=str(dest), notes=note,
                  attempts=attempts, **result)
        log.info("Filed: %s -> %s / %s", original_name, result["patient_name"], result["department"])
        return True

    except BusinessException as e:
        dest = move_to(NEEDS_REVIEW_DIR, claimed, original_name)
        db_update(row_id, status="Needs Review", storage_path=str(dest), notes=str(e))
        log.info("Needs review: %s (%s)", original_name, e)
        return True

    except Exception as e:
        log.exception("Failed: %s", original_name)
        try:
            dest = move_to(FAILED_DIR, claimed, original_name)
            db_update(row_id, status="Failed", storage_path=str(dest),
                      attempts=getattr(e, "attempts", attempts), notes=f"{type(e).__name__}: {e}"[:500])
        except Exception:
            log.exception("Could not even move %s to failed/", original_name)
        return True

    finally:
        try:
            db_update(row_id, duration_sec=round(time.time() - started, 2))
        except Exception:
            pass


# --------------------------------------------------------------------------
# HUMAN REVIEW (the dashboard only queues a decision; the bot does the filing)
# --------------------------------------------------------------------------
def apply_review_actions() -> bool:
    """File documents a reviewer approved or reclassified in the dashboard."""
    with closing(connect()) as conn:
        pending = conn.execute(
            "SELECT * FROM review_actions WHERE handled_at IS NULL ORDER BY id").fetchall()
        rows = {a["id"]: conn.execute("SELECT * FROM audit_log WHERE id = ?",
                                      (a["audit_id"],)).fetchone() for a in pending}

    for a in pending:
        row = rows[a["id"]]
        result = "Filed"
        try:
            src = Path(row["storage_path"]) if row and row["storage_path"] else None
            if row is None or row["status"] != "Needs Review" or src is None or not src.exists():
                result = "Skipped: file is no longer in the review queue"
            else:
                doc_type = a["document_type"]
                department = DEPARTMENT_MAP.get(doc_type, "General")
                patient = safe_name(a["patient_name"])
                dest_folder = STORAGE_DIR / patient / department
                dest_folder.mkdir(parents=True, exist_ok=True)
                dest = unique_path(dest_folder, row["original_filename"])
                tmp = dest.with_name(dest.name + ".part")
                shutil.copy2(src, tmp)
                os.replace(tmp, dest)
                shutil.move(str(src), str(unique_path(ARCHIVE_DIR, row["original_filename"])))
                db_update(row["id"], status="Filed", patient_name=patient,
                          document_type=doc_type, department=department,
                          method="manual_review", confidence=1.0,
                          storage_path=str(dest), notes="Approved by human reviewer")
                log.info("Review approved: %s -> %s / %s",
                         row["original_filename"], patient, department)
        except Exception as e:
            log.exception("Review action %s failed", a["id"])
            result = f"Error: {e}"[:200]
        with closing(connect()) as conn, conn:
            conn.execute("UPDATE review_actions SET handled_at = ?, result = ? WHERE id = ?",
                         (now(), result, a["id"]))
    return bool(pending)


# --------------------------------------------------------------------------
# STARTUP RECOVERY + MAIN LOOP
# --------------------------------------------------------------------------
def recover_interrupted():
    """Files left in processing/ mean the bot died mid-file. Put them back."""
    leftovers = [p for p in PROCESSING_DIR.iterdir() if p.is_file()]
    for p in leftovers:
        dest = unique_path(DROPZONE_DIR, p.name)
        shutil.move(str(p), str(dest))
        log.warning("Recovered interrupted file back to dropzone: %s", p.name)
    if leftovers:
        with closing(connect()) as conn, conn:
            conn.execute("UPDATE audit_log SET status='Interrupted', "
                         "notes='Bot stopped mid-file; file re-queued' WHERE status='Processing'")


def list_pending():
    return sorted(
        p for p in DROPZONE_DIR.iterdir()
        if p.is_file() and not p.name.startswith((".", "~$"))
        and not p.name.lower().endswith(IGNORED_SUFFIXES)
    )


def run_cycle() -> bool:
    changed = False
    try:
        changed |= apply_review_actions()
    except Exception:
        log.exception("Review actions failed (loop continues)")
    for path in list_pending():
        if stop_event.is_set():
            break
        heartbeat("processing", path.name)
        try:
            changed |= process_file(path)
        except Exception:
            log.exception("Unexpected error on %s (loop continues)", path.name)
    return changed


def main_loop():
    log.info("Robot worker started. Watching %s (poll %ss).", DROPZONE_DIR.resolve(), POLL_INTERVAL_SECONDS)
    dirty = True  # export once at startup
    while not stop_event.is_set():
        heartbeat("idle")
        try:
            dirty |= run_cycle()
            if dirty and EXCEL_BRIDGE:
                dirty = not export_excel()
        except Exception:
            log.exception("Loop error")
        stop_event.wait(POLL_INTERVAL_SECONDS)
    heartbeat("stopped")
    log.info("Robot worker stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unattended medical-record RPA worker")
    parser.add_argument("--once", action="store_true", help="process waiting files, then exit")
    parser.add_argument("--export", action="store_true", help="rewrite the Excel export from the DB and exit")
    args = parser.parse_args()

    init_db()
    if args.export:
        export_excel()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    recover_interrupted()
    if args.once:
        run_cycle()
        heartbeat("idle")
        if EXCEL_BRIDGE:
            export_excel()
    else:
        main_loop()