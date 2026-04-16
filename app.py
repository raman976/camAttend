import cv2
import streamlit as st
import numpy as np
import hashlib
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime

from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.matcher import FaceMatcher
from core.langgraph_agent import LangGraphAttendanceAgent
from database.supabase_db import SupabaseDB

load_dotenv()

st.set_page_config(
    page_title="CamAttend – Smart Attendance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp { background: #0a0a0a; }

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1280px;
}

/* ── Typography ── */
h1 { color: #f5f5f7; font-size: 2rem !important; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.25rem !important; }
h2 { color: #f5f5f7; font-size: 1.5rem !important; font-weight: 600; letter-spacing: -0.01em; }
h3 { color: #f5f5f7; font-size: 1.125rem !important; font-weight: 600; }
p  { color: #86868b; font-size: 0.9375rem; line-height: 1.6; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #222222;
}
[data-testid="stSidebar"] * { color: #f5f5f7 !important; }

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: #1c1c1e;
    color: #f5f5f7;
    border: 1px solid #3a3a3c;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.9375rem;
}
.stTextInput > div > div > input:focus {
    border-color: #0071e3;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15);
    outline: none;
}
.stTextInput > label { color: #aeaeb2 !important; font-size: 0.875rem; font-weight: 500; }

/* ── Buttons ── */
.stButton > button {
    background: #1c1c1e;
    color: #f5f5f7;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 0.9375rem;
    font-weight: 500;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: #2c2c2e;
    border-color: #636366;
}
.stButton > button[kind="primary"] {
    background: #0071e3;
    border-color: #0071e3;
    color: #fff;
}
.stButton > button[kind="primary"]:hover { background: #0077ed; border-color: #0077ed; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1c1c1e;
    border: 1.5px dashed #3a3a3c;
    border-radius: 12px;
    padding: 1.5rem;
}
[data-testid="stFileUploader"]:hover { border-color: #0071e3; }
[data-testid="stFileUploader"] label { color: #86868b !important; font-size: 0.875rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #3a3a3c;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #86868b;
    border-radius: 0;
    padding: 10px 18px;
    font-size: 0.9375rem;
    border-bottom: 2px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: #f5f5f7; }
.stTabs [aria-selected="true"] {
    color: #f5f5f7 !important;
    background: transparent;
    border-bottom: 2px solid #0071e3;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #f5f5f7; font-size: 2rem !important; font-weight: 600; letter-spacing: -0.02em; }
[data-testid="stMetricLabel"] { color: #86868b; font-size: 0.8125rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Sliders & radio ── */
.stSlider > div > div > div > div { background: #0071e3; }
.stSlider label { color: #aeaeb2 !important; font-size: 0.875rem; }
.stRadio > div > label > div[data-testid="stMarkdownContainer"] > p { color: #f5f5f7 !important; font-size: 0.9375rem; }

/* ── Status / alert ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Divider ── */
hr { border-color: #2c2c2e; margin: 1.5rem 0; opacity: 1; }

/* ── Cards ── */
.cam-card {
    background: #1c1c1e;
    border: 1px solid #2c2c2e;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.cam-card-accent {
    background: rgba(0, 113, 227, 0.07);
    border: 1px solid rgba(0, 113, 227, 0.25);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}

/* ── Decision badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-present  { background: rgba(52, 199, 89,  0.15); color: #34c759; border: 1px solid rgba(52,199,89,0.3); }
.badge-late     { background: rgba(255,159, 10,  0.15); color: #ff9f0a; border: 1px solid rgba(255,159,10,0.3); }
.badge-absent   { background: rgba(255, 59, 48,  0.15); color: #ff3b30; border: 1px solid rgba(255,59,48,0.3); }
.badge-flagged  { background: rgba(175,82,222,  0.15); color: #af52de; border: 1px solid rgba(175,82,222,0.3); }

/* ── Agent trace ── */
.trace-step {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.8125rem;
    color: #86868b;
    padding: 4px 0;
    border-left: 2px solid #3a3a3c;
    padding-left: 10px;
    margin: 3px 0;
}
.trace-step:last-child { color: #f5f5f7; border-left-color: #0071e3; }

/* ── Progress ── */
.stProgress > div > div > div > div { background: #0071e3; }

/* ── Checkbox ── */
.stCheckbox label { color: #aeaeb2 !important; font-size: 0.875rem; }

/* ── Caption ── */
.stCaption { color: #636366 !important; font-size: 0.8125rem; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1c1c1e;
    border: 1px solid #2c2c2e;
    border-radius: 10px;
}
[data-testid="stExpander"] summary { color: #aeaeb2 !important; font-size: 0.875rem; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Image ── */
[data-testid="stImage"] { border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "db": None,
    "admin_user": None,
    "organization": None,
    "detector": None,
    "embedder": None,
    "matcher": None,
    "agent": None,
    "student_lookup": {},
    "agent_results": [],
    "agent_session_id": None,
    "review_queue": [],
    "current_lecture": None,
    "last_lecture_summary": None,
    "result_image": None,
    "current_page": "Dashboard",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.db is None:
    st.session_state.db = SupabaseDB()


# ── Helper: render decision badge ─────────────────────────────────────────────
def _badge(decision: str) -> str:
    cls = {
        "PRESENT": "badge-present",
        "LATE": "badge-late",
        "ABSENT": "badge-absent",
        "FLAGGED": "badge-flagged",
    }.get(decision.upper(), "badge-flagged")
    return f'<span class="badge {cls}">{decision}</span>'


def _action_icon(action: str) -> str:
    return {
        "MARK_PRESENT": "✅",
        "SOFT_FLAG": "🟡",
        "ESCALATE_TO_INSTRUCTOR": "🔶",
        "RETAKE_PHOTO": "🔴",
        "SUGGEST_ENROLL_NEW_STUDENT": "🟣",
    }.get(action, "⚪")


# ── Login page ────────────────────────────────────────────────────────────────
def login_page():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center'>"
            "<div style='font-size:2.5rem'>🎯</div>"
            "<h1 style='margin:0.5rem 0 0.25rem'>CamAttend</h1>"
            "<p style='margin-bottom:2rem'>AI-Powered Attendance for Colleges</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        tab_login, tab_reg = st.tabs(["Sign In", "Register"])

        with tab_login:
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            email = st.text_input("Email", key="li_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="li_pass", placeholder="Password")
            org_code = st.text_input("Organization Code", key="li_org", placeholder="org-code")
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("Sign In", type="primary", use_container_width=True, key="li_btn"):
                if not (email and password and org_code):
                    st.error("Please fill all fields.")
                else:
                    with st.spinner("Verifying…"):
                        org = st.session_state.db.get_organization_by_code(org_code)
                        if not org:
                            st.error("Organization not found.")
                        else:
                            user = st.session_state.db.verify_user_password(email, password)
                            if not user:
                                st.error("Invalid email or password.")
                            elif user["organization_id"] != org["id"]:
                                st.error("You don't belong to this organization.")
                            else:
                                st.session_state.admin_user = user
                                st.session_state.organization = org
                                st.rerun()

        with tab_reg:
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            name = st.text_input("Full Name", key="r_name", placeholder="Jane Doe")
            email = st.text_input("Email", key="r_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="r_pass", placeholder="Min 6 characters")
            org_code = st.text_input("Organization Code", key="r_org", placeholder="Create or join")

            org_name = None
            if org_code:
                existing = st.session_state.db.get_organization_by_code(org_code)
                if not existing:
                    st.info("New organization code — enter a name to create it.")
                    org_name = st.text_input("Organization Name", key="r_org_name", placeholder="e.g., My College")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True, key="r_btn"):
                if not (name and email and password and org_code):
                    st.error("Please fill all fields.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account…"):
                        org = st.session_state.db.get_organization_by_code(org_code)
                        if not org:
                            if not org_name:
                                st.error("Enter an organization name.")
                            else:
                                org = st.session_state.db.create_organization(
                                    name=org_name, code=org_code, contact_email=email
                                )
                                if not org:
                                    st.error("Failed to create organization.")
                                    st.stop()
                        if org:
                            user = st.session_state.db.create_user(
                                organization_id=org["id"],
                                email=email,
                                name=name,
                                password=password,
                                role="admin",
                            )
                            if user:
                                st.success("Account created! You can now sign in.")
                            else:
                                st.error("Registration failed — email may already exist.")


# ── Initialize AI models ───────────────────────────────────────────────────────
def initialize_models():
    if st.session_state.detector is None:
        with st.spinner("Loading face detection models…"):
            st.session_state.detector = FaceDetector()
            st.session_state.embedder = FaceEmbedder()

    if st.session_state.agent is None:
        st.session_state.agent = LangGraphAttendanceAgent()

    if st.session_state.matcher is None:
        st.session_state.matcher = FaceMatcher()

    if not st.session_state.student_lookup:
        embeddings = st.session_state.db.get_student_embeddings(
            st.session_state.organization["id"]
        )
        for student_id, name, emb in embeddings:
            st.session_state.student_lookup[student_id] = name
            if student_id not in st.session_state.matcher.student_ids:
                st.session_state.matcher.add_embedding(emb, student_id)


# ── Utility functions ──────────────────────────────────────────────────────────
def compute_face_quality(image_bgr, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.3
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.3
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = np.clip(np.mean(gray) / 255.0, 0.0, 1.0)
    sharpness = np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 180.0, 0.0, 1.0)
    size_ratio = ((x2 - x1) * (y2 - y1)) / float(h * w)
    size_score = np.clip(size_ratio / 0.07, 0.0, 1.0)
    return float(np.clip(0.45 * sharpness + 0.35 * brightness + 0.20 * size_score, 0.0, 1.0))


def make_face_signature(embedding):
    if embedding is None:
        return None
    sig = np.round(embedding[:16], 2).astype(np.float32).tobytes()
    return hashlib.md5(sig).hexdigest()


def build_review_queue(agent_results, session_id):
    queue = []
    flagged_actions = {
        "SOFT_FLAG",
        "ESCALATE_TO_INSTRUCTOR",
        "RETAKE_PHOTO",
        "SUGGEST_ENROLL_NEW_STUDENT",
    }
    for idx, item in enumerate(agent_results):
        if item.get("requires_review") or item.get("action") in flagged_actions:
            queue.append(
                {
                    "queue_id": f"{session_id}_{idx}",
                    "student_name": item.get("student_name", "Unknown"),
                    "student_id": item.get("student_id"),
                    "decision": item.get("agent_decision"),
                    "action": item.get("action"),
                    "uncertainty": item.get("uncertainty", 0.0),
                    "reasoning": item.get("reasoning", ""),
                    "status": "pending",
                    "final_override": None,
                }
            )
    return queue


def apply_review_action(queue_item, action, override_value=None):
    queue_item["status"] = "resolved"
    queue_item["final_override"] = {"action": action, "value": override_value}
    for result in st.session_state.agent_results:
        if (
            result.get("student_name") == queue_item.get("student_name")
            and result.get("action") == queue_item.get("action")
        ):
            result["review_status"] = action
            result["override_value"] = override_value


def start_lecture_session(title, subject=None, location=None):
    now = datetime.now()
    lecture = st.session_state.db.create_lecture(
        organization_id=st.session_state.organization["id"],
        created_by=st.session_state.admin_user["id"],
        title=title,
        subject=subject,
        lecture_date=now.date(),
        start_time=now.time(),
        location=location,
    )
    if not lecture:
        return None
    try:
        st.session_state.db.update_lecture_status(lecture["id"], "ongoing")
    except Exception:
        pass
    st.session_state.current_lecture = {
        "id": lecture["id"],
        "title": lecture.get("title", title),
        "subject": lecture.get("subject", subject),
        "location": lecture.get("location", location),
        "started_at": now,
        "processed_images": 0,
        "faces_processed": 0,
        "recognized_faces": 0,
        "flagged_faces": 0,
        "unknown_faces": 0,
    }
    st.session_state.agent_session_id = lecture["id"]
    st.session_state.review_queue = []
    st.session_state.agent_results = []
    return st.session_state.current_lecture


def end_lecture_session():
    lecture = st.session_state.current_lecture
    if not lecture:
        return None
    try:
        st.session_state.db.update_lecture_status(lecture["id"], "completed")
    except Exception:
        pass
    db_stats = {}
    try:
        db_stats = st.session_state.db.get_attendance_stats(lecture["id"]) or {}
    except Exception:
        pass
    summary = {
        "lecture_id": lecture["id"],
        "title": lecture.get("title"),
        "started_at": lecture.get("started_at"),
        "ended_at": datetime.now(),
        "processed_images": lecture.get("processed_images", 0),
        "faces_processed": lecture.get("faces_processed", 0),
        "recognized_faces": lecture.get("recognized_faces", 0),
        "flagged_faces": lecture.get("flagged_faces", 0),
        "unknown_faces": lecture.get("unknown_faces", 0),
        "db_attendance_stats": db_stats,
        "resolved_reviews": len(
            [x for x in st.session_state.review_queue if x.get("status") == "resolved"]
        ),
    }
    st.session_state.last_lecture_summary = summary
    st.session_state.current_lecture = None
    st.session_state.agent_session_id = None
    return summary


# ── Dashboard page ─────────────────────────────────────────────────────────────
def dashboard_page():
    st.markdown("## Dashboard")
    st.markdown(f"Welcome back, **{st.session_state.admin_user['name']}** · {st.session_state.organization['name']}")
    st.divider()

    embeddings = st.session_state.db.get_student_embeddings(
        st.session_state.organization["id"]
    )
    total_students = len(embeddings)

    # ── Top metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students Enrolled", total_students)
    m2.metric("Organization", st.session_state.organization["code"])
    m3.metric("Admin", st.session_state.admin_user["name"].split()[0])
    m4.metric("Agent", "LangGraph ✓")

    st.divider()

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("### Active Lecture")
        if st.session_state.current_lecture:
            cur = st.session_state.current_lecture
            elapsed = int(
                (datetime.now() - cur["started_at"]).total_seconds() / 60
            )
            st.markdown(
                f"""<div class="cam-card-accent">
                    <div style="color:#f5f5f7;font-weight:600;font-size:1.05rem">{cur['title']}</div>
                    <div style="color:#86868b;margin-top:0.5rem;font-size:0.875rem">
                        ⏱ {elapsed} min elapsed &nbsp;·&nbsp;
                        🖼 {cur['processed_images']} images &nbsp;·&nbsp;
                        👤 {cur['faces_processed']} faces processed
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No active lecture. Start one from the **Recognize** page.")

        if st.session_state.last_lecture_summary:
            s = st.session_state.last_lecture_summary
            st.markdown("### Last Lecture")
            st.markdown(
                f"""<div class="cam-card">
                    <div style="color:#f5f5f7;font-weight:600">{s['title']}</div>
                    <div style="color:#86868b;margin-top:0.5rem;font-size:0.875rem">
                        Images: {s['processed_images']} &nbsp;·&nbsp;
                        Faces: {s['faces_processed']} &nbsp;·&nbsp;
                        Recognized: {s['recognized_faces']} &nbsp;·&nbsp;
                        Flagged: {s['flagged_faces']}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("### Enrolled Students")
        if total_students == 0:
            st.info("No students enrolled yet. Use the **Enroll** page to add students.")
        else:
            for i, (sid, sname, _) in enumerate(embeddings[:10]):
                st.markdown(
                    f"""<div class="cam-card" style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="color:#f5f5f7;font-weight:500">{sname}</span>
                            <span style="color:#636366;font-size:0.8125rem;margin-left:0.75rem">{sid}</span>
                        </div>
                        <span style="color:#636366;font-size:0.8125rem">#{i+1}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if total_students > 10:
                st.caption(f"Showing 10 of {total_students} students")

    with right:
        st.markdown("### Quick Actions")
        if st.button("➕  Enroll Student", use_container_width=True):
            st.session_state.current_page = "Enroll"
            st.rerun()
        if st.button("🔍  Recognize Faces", use_container_width=True):
            st.session_state.current_page = "Recognize"
            st.rerun()

        st.markdown("### Attendance Stats")
        db_stats = {}
        try:
            if st.session_state.current_lecture:
                db_stats = (
                    st.session_state.db.get_attendance_stats(
                        st.session_state.current_lecture["id"]
                    )
                    or {}
                )
        except Exception:
            pass

        if db_stats:
            st.metric("Present", db_stats.get("present", 0))
            st.metric("Late", db_stats.get("late", 0))
            st.metric("Absent", db_stats.get("absent", 0))
        else:
            st.info("Stats appear during an active lecture.")

        st.markdown("### System")
        st.markdown(
            f"""<div class="cam-card" style="font-size:0.875rem;line-height:2">
                <div style="color:#86868b">Organization</div>
                <div style="color:#f5f5f7">{st.session_state.organization['name']}</div>
                <div style="color:#86868b;margin-top:0.5rem">Date</div>
                <div style="color:#f5f5f7">{datetime.now().strftime('%B %d, %Y')}</div>
                <div style="color:#86868b;margin-top:0.5rem">Agent</div>
                <div style="color:#34c759">Active</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Enroll page ────────────────────────────────────────────────────────────────
def enroll_page():
    st.markdown("## Enroll Student")
    st.markdown("Add a new student and register their face with the AI system.")
    st.divider()

    col_form, col_preview = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### Student Information")
        name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
        student_id_input = st.text_input(
            "Student ID (Optional)", placeholder="Auto-generated if blank"
        )

        st.markdown("### Upload Photo")
        uploaded_file = st.file_uploader(
            "Clear photo — one face only",
            type=["jpg", "jpeg", "png"],
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("Enroll Student", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload a photo.")
            elif not name:
                st.error("Please enter the student's name.")
            else:
                image = Image.open(uploaded_file)
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                bar = st.progress(0, "Detecting face…")
                boxes, probs, landmarks = st.session_state.detector.detect_faces(image_np)

                if len(boxes) == 0:
                    st.error("No face detected. Use a clear, well-lit photo.")
                    bar.empty()
                elif len(boxes) > 1:
                    st.error(f"{len(boxes)} faces detected — upload a solo photo.")
                    bar.empty()
                else:
                    bar.progress(50, "Extracting face embedding…")
                    emb = st.session_state.embedder.get_embedding(
                        image_np, bbox=boxes[0], landmark=landmarks[0]
                    )
                    if emb is None:
                        st.error("Could not process face. Try another photo.")
                        bar.empty()
                    else:
                        bar.progress(80, "Saving to database…")
                        student_id = (
                            student_id_input.upper()
                            if student_id_input
                            else name.replace(" ", "_").upper()
                        )
                        student = st.session_state.db.enroll_student(
                            organization_id=st.session_state.organization["id"],
                            student_id=student_id,
                            name=name,
                            embedding=emb,
                        )
                        bar.progress(100, "Done!")
                        bar.empty()

                        if student:
                            st.session_state.matcher.add_embedding(emb, student_id)
                            st.session_state.student_lookup[student["student_id"]] = student["name"]
                            st.success(f"✅ {name} enrolled (ID: {student_id})")
                            st.balloons()
                        else:
                            st.error("Enrollment failed — ID may already exist.")

    with col_preview:
        if uploaded_file:
            st.markdown("### Preview")
            st.image(uploaded_file, use_container_width=True)
            st.markdown(
                """<div class="cam-card">
                    <div style="color:#f5f5f7;font-weight:600;margin-bottom:0.5rem">Tips for best results</div>
                    <div style="color:#86868b;font-size:0.875rem;line-height:1.8">
                        ✅ Clear, well-lit photo<br>
                        ✅ Face clearly visible<br>
                        ✅ Front-facing<br>
                        ❌ No sunglasses or masks<br>
                        ❌ No blurry images
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            embeddings = st.session_state.db.get_student_embeddings(
                st.session_state.organization["id"]
            )
            st.markdown("### Enrollment Stats")
            c1, c2 = st.columns(2)
            c1.metric("Total Students", len(embeddings))
            c2.metric("Org Code", st.session_state.organization["code"])
            st.markdown(
                """<div class="cam-card" style="margin-top:1rem;text-align:center;padding:2rem">
                    <div style="font-size:2rem">📷</div>
                    <div style="color:#86868b;margin-top:0.75rem;font-size:0.875rem">
                        Upload a photo to preview and enroll
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Recognize page ─────────────────────────────────────────────────────────────
def recognize_page():
    st.markdown("## Face Recognition")
    st.markdown("Upload a class photo — the agent will identify students and decide attendance.")
    st.divider()

    # ── Lecture session bar (full width) ──
    lec_col, status_col = st.columns([2, 1], gap="large")

    with lec_col:
        if st.session_state.current_lecture is None:
            st.markdown("### Start a Lecture")
            t1, t2, t3 = st.columns([2, 1, 1])
            with t1:
                lecture_title = st.text_input(
                    "Title", value="Attendance Session", key="lec_title"
                )
            with t2:
                lecture_subject = st.text_input("Subject", key="lec_subj")
            with t3:
                lecture_location = st.text_input("Location", key="lec_loc")
            if st.button("▶ Start Lecture", type="primary", use_container_width=True):
                started = start_lecture_session(
                    title=lecture_title.strip() or "Attendance Session",
                    subject=lecture_subject.strip() or None,
                    location=lecture_location.strip() or None,
                )
                if started:
                    st.success(f"Lecture started: **{started['title']}**")
                    st.rerun()
                else:
                    st.error("Could not start lecture.")
        else:
            cur = st.session_state.current_lecture
            elapsed = int((datetime.now() - cur["started_at"]).total_seconds() / 60)
            st.markdown(
                f"""<div class="cam-card-accent">
                    <div style="color:#f5f5f7;font-weight:600">{cur['title']}</div>
                    <div style="color:#86868b;font-size:0.875rem;margin-top:0.25rem">
                        ⏱ {elapsed} min &nbsp;·&nbsp;
                        🖼 {cur['processed_images']} images &nbsp;·&nbsp;
                        👤 {cur['faces_processed']} faces
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("⏹ End Lecture", use_container_width=True):
                summary = end_lecture_session()
                if summary:
                    st.success("Lecture ended.")
                    st.rerun()

    with status_col:
        if st.session_state.last_lecture_summary:
            s = st.session_state.last_lecture_summary
            st.caption("Last lecture")
            st.markdown(
                f"""<div class="cam-card" style="font-size:0.8125rem;line-height:1.8;color:#86868b">
                    <b style="color:#f5f5f7">{s['title']}</b><br>
                    Faces: {s['faces_processed']} &nbsp; Recognized: {s['recognized_faces']}<br>
                    Flagged: {s['flagged_faces']} &nbsp; Unknown: {s['unknown_faces']}
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Two-column main area ──
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("### Settings")
        threshold = st.slider("Recognition Threshold", 0.3, 0.9, 0.5, 0.05)
        class_start_offset = st.slider(
            "Minutes since class start",
            0, 45, 5, 1,
            help="Agent uses this to classify on-time / late / absent",
        )
        enable_agent = st.checkbox("Enable agent reasoning", value=True)

        st.markdown("### Upload Photo")
        uploaded_file = st.file_uploader(
            "Group class photo", type=["jpg", "jpeg", "png"], key="rec_upload"
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        run_btn = st.button(
            "Recognize Faces",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None or st.session_state.current_lecture is None),
        )

        if uploaded_file and st.session_state.current_lecture is None:
            st.warning("Start a lecture first to enable recognition.")

        # ── Recognition logic ──
        if run_btn and uploaded_file and st.session_state.current_lecture:
            image = Image.open(uploaded_file)
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            class_start_time = st.session_state.current_lecture["started_at"]

            if not st.session_state.agent_session_id:
                st.session_state.agent_session_id = (
                    f"{st.session_state.organization['id']}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

            bar = st.progress(0, "Detecting faces…")
            boxes, probs, landmarks = st.session_state.detector.detect_faces(image_np)

            if len(boxes) == 0:
                st.error("No faces detected in the image.")
                bar.empty()
            else:
                bar.progress(35, f"Embedding {len(boxes)} face(s)…")

                candidates = []
                for box, landmark in zip(boxes, landmarks):
                    emb = st.session_state.embedder.get_embedding(
                        image_np, bbox=box, landmark=landmark
                    )
                    if emb is None:
                        continue
                    student_id, score = st.session_state.matcher.match(emb)
                    quality = compute_face_quality(image_np, box)
                    sig = make_face_signature(emb)
                    candidates.append(
                        {
                            "box": box,
                            "student_id": student_id,
                            "score": float(score) if score is not None else None,
                            "quality": quality,
                            "signature": sig,
                        }
                    )

                if not candidates:
                    st.error("Could not extract embeddings from detected faces.")
                    bar.empty()
                else:
                    low_conf_count = sum(
                        1
                        for c in candidates
                        if c["student_id"] is None
                        or c["score"] is None
                        or c["score"] <= threshold
                    )
                    low_conf_ratio = low_conf_count / max(1, len(candidates))

                    img_annotated = image_np.copy()
                    face_decisions = []
                    present_students: set = set()
                    unknown_count = 0
                    flagged_count = 0
                    retake_required = False

                    for i, cand in enumerate(candidates):
                        bar.progress(35 + int((i + 1) / len(candidates) * 55))
                        box = cand["box"]
                        student_id = cand["student_id"]
                        score = cand["score"]
                        quality = cand["quality"]
                        sig = cand["signature"]
                        x1, y1, x2, y2 = map(int, box)
                        now = datetime.now()
                        known = student_id is not None and score is not None and score > threshold

                        if known:
                            student_name = st.session_state.student_lookup.get(student_id, student_id)
                            try:
                                student_history = st.session_state.db.get_student_attendance_stats(
                                    student_id, st.session_state.organization["id"]
                                )
                            except Exception:
                                student_history = None
                            try:
                                attendance_records = st.session_state.db.get_student_attendance_history(student_id)
                            except Exception:
                                attendance_records = []
                            prev_errors = int(student_history.get("previous_recognition_errors", 0)) if student_history else 0

                            if enable_agent and st.session_state.agent:
                                ar = st.session_state.agent.make_decision(
                                    student_name=student_name,
                                    confidence_score=float(score),
                                    current_time=now,
                                    class_start_time=class_start_time,
                                    student_history=student_history,
                                    attendance_records=attendance_records,
                                    lecture_context={
                                        "mode": "image_upload",
                                        "class_start_offset_minutes": class_start_offset,
                                        "threshold": threshold,
                                        "image_quality": quality,
                                        "previous_recognition_errors": prev_errors,
                                        "low_conf_ratio": low_conf_ratio,
                                        "total_faces": len(candidates),
                                        "is_unknown": False,
                                        "face_signature": sig,
                                    },
                                    session_id=st.session_state.agent_session_id,
                                )
                            else:
                                ar = {
                                    "decision": "PRESENT",
                                    "confidence": float(score),
                                    "uncertainty_score": 0.1,
                                    "action": "MARK_PRESENT",
                                    "reasoning": "Agent disabled — using threshold match only.",
                                    "requires_review": False,
                                    "agent_type": "disabled",
                                    "time_offset_minutes": (now - class_start_time).total_seconds() / 60,
                                    "trace": [],
                                }
                        else:
                            student_name = "Unknown"
                            if enable_agent and st.session_state.agent:
                                ar = st.session_state.agent.make_decision(
                                    student_name=student_name,
                                    confidence_score=float(score) if score else 0.0,
                                    current_time=now,
                                    class_start_time=class_start_time,
                                    student_history=None,
                                    attendance_records=[],
                                    lecture_context={
                                        "mode": "image_upload",
                                        "class_start_offset_minutes": class_start_offset,
                                        "threshold": threshold,
                                        "image_quality": quality,
                                        "previous_recognition_errors": 0,
                                        "low_conf_ratio": low_conf_ratio,
                                        "total_faces": len(candidates),
                                        "is_unknown": True,
                                        "face_signature": sig,
                                    },
                                    session_id=st.session_state.agent_session_id,
                                )
                            else:
                                ar = {
                                    "decision": "FLAGGED",
                                    "confidence": float(score) if score else 0.0,
                                    "uncertainty_score": 0.8,
                                    "action": "ESCALATE_TO_INSTRUCTOR",
                                    "reasoning": "Below recognition threshold.",
                                    "requires_review": True,
                                    "agent_type": "threshold_filter",
                                    "time_offset_minutes": (now - class_start_time).total_seconds() / 60,
                                    "trace": [],
                                }
                            unknown_count += 1

                        action = ar.get("action", "ESCALATE_TO_INSTRUCTOR")

                        # Save audit trail + attendance
                        try:
                            st.session_state.db.save_agent_decision(
                                lecture_id=st.session_state.current_lecture["id"],
                                student_id=student_id if known else None,
                                student_name=student_name,
                                face_confidence=float(score) if score else 0.0,
                                agent_decision=ar.get("decision", "FLAGGED"),
                                agent_reasoning=ar.get("reasoning", ""),
                                agent_type=ar.get("agent_type", "rule_based"),
                                time_offset_minutes=ar.get("time_offset_minutes"),
                                requires_review=ar.get("requires_review", False),
                            )
                        except Exception:
                            pass

                        if known and ar.get("decision") in {"PRESENT", "LATE", "ABSENT"}:
                            try:
                                st.session_state.db.mark_attendance(
                                    lecture_id=st.session_state.current_lecture["id"],
                                    student_id=student_id,
                                    marked_by=st.session_state.admin_user["id"],
                                    confidence_score=float(score),
                                    status=ar.get("decision", "PRESENT").lower(),
                                    notes=ar.get("reasoning", ""),
                                )
                            except Exception:
                                pass

                        face_decisions.append(
                            {
                                "student_id": student_id,
                                "student_name": student_name,
                                "face_score": round(float(score), 3) if score else None,
                                "image_quality": round(float(quality), 3),
                                "uncertainty": round(float(ar.get("uncertainty_score", 0)), 3),
                                "agent_decision": ar.get("decision"),
                                "action": action,
                                "requires_review": ar.get("requires_review", False),
                                "agent_type": ar.get("agent_type"),
                                "reasoning": ar.get("reasoning"),
                                "trace": ar.get("trace", []),
                            }
                        )

                        # Annotate image
                        color_map = {
                            "MARK_PRESENT": (0, 220, 90),
                            "SOFT_FLAG": (0, 210, 255),
                            "ESCALATE_TO_INSTRUCTOR": (0, 140, 255),
                            "RETAKE_PHOTO": (0, 60, 255),
                            "SUGGEST_ENROLL_NEW_STUDENT": (200, 0, 255),
                        }
                        color = color_map.get(action, (120, 120, 120))
                        decision_label = ar.get("decision", "FLAGGED")
                        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            img_annotated,
                            f"{student_name} | {decision_label}",
                            (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                        if action == "MARK_PRESENT" and ar.get("decision") in {"PRESENT", "LATE"}:
                            present_students.add(student_name)
                        elif action in {
                            "SOFT_FLAG",
                            "RETAKE_PHOTO",
                            "ESCALATE_TO_INSTRUCTOR",
                            "SUGGEST_ENROLL_NEW_STUDENT",
                        }:
                            flagged_count += 1
                            if action == "RETAKE_PHOTO":
                                retake_required = True

                    bar.progress(100, "Done!")
                    bar.empty()

                    # Sort: auto-present first (most confident), then soft-flagged present/late,
                    # then other flagged, then absent — within each tier sort by confidence desc.
                    def _result_sort_key(r):
                        d = r.get("agent_decision", "FLAGGED")
                        conf = r.get("face_score") or 0.0
                        flagged = r.get("requires_review", False)
                        if d in {"PRESENT", "LATE"} and not flagged:
                            return (0, -conf)
                        if d in {"PRESENT", "LATE"} and flagged:
                            return (1, -conf)
                        if d == "FLAGGED":
                            return (2, -conf)
                        return (3, -conf)  # ABSENT

                    face_decisions.sort(key=_result_sort_key)

                    st.session_state.result_image = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
                    st.session_state.agent_results = face_decisions
                    st.session_state.review_queue = build_review_queue(
                        face_decisions, st.session_state.agent_session_id
                    )

                    cur = st.session_state.current_lecture
                    cur["processed_images"] += 1
                    cur["faces_processed"] += len(candidates)
                    cur["recognized_faces"] += len(present_students)
                    cur["flagged_faces"] += flagged_count
                    cur["unknown_faces"] += unknown_count

    with right_col:
        if st.session_state.result_image is not None:
            st.markdown("### Annotated Result")
            st.image(st.session_state.result_image, use_container_width=True)
            st.markdown(
                """<div class="cam-card" style="font-size:0.8125rem;line-height:2">
                    <b style="color:#f5f5f7">Legend</b><br>
                    <span style="color:#00dc5a">■</span> Auto-marked present/late &nbsp;
                    <span style="color:#00d2ff">■</span> Soft flag<br>
                    <span style="color:#008cff">■</span> Escalate to instructor &nbsp;
                    <span style="color:#003cff">■</span> Retake photo<br>
                    <span style="color:#c800ff">■</span> Suggest enrollment
                </div>""",
                unsafe_allow_html=True,
            )
        elif uploaded_file:
            st.markdown("### Preview")
            st.image(uploaded_file, use_container_width=True)
            st.info("Click **Recognize Faces** to process this image.")
        else:
            st.markdown(
                """<div class="cam-card" style="text-align:center;padding:3rem 1rem">
                    <div style="font-size:2.5rem">🎯</div>
                    <div style="color:#f5f5f7;font-weight:600;margin:0.75rem 0 0.25rem">Upload a class photo</div>
                    <div style="color:#86868b;font-size:0.875rem">
                        Start a lecture, upload a photo, and the agent<br>
                        will identify each face and mark attendance.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Results section (full width, below columns) ───────────────────────────
    if st.session_state.agent_results:
        st.divider()
        st.markdown("### Recognition Results")

        results = st.session_state.agent_results
        present_n = sum(1 for r in results if r.get("agent_decision") in {"PRESENT", "LATE"})
        flagged_n = sum(1 for r in results if r.get("requires_review"))
        unknown_n = sum(1 for r in results if r.get("student_name") == "Unknown")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Faces", len(results))
        m2.metric("Marked Attendance", present_n)
        m3.metric("Needs Review", flagged_n)
        m4.metric("Unknown", unknown_n)

        if retake_required if "retake_required" in dir() else any(  # noqa: F821
            r.get("action") == "RETAKE_PHOTO" for r in results
        ):
            st.error("Agent recommends a retake — photo quality is too low for reliable identification.")

        st.markdown("#### Per-Student Agent Decisions")
        for idx, r in enumerate(results):
            decision = r.get("agent_decision", "FLAGGED")
            action = r.get("action", "ESCALATE_TO_INSTRUCTOR")
            name = r.get("student_name", "Unknown")
            icon = _action_icon(action)
            badge_html = _badge(decision)

            with st.expander(
                f"{icon}  {name} — {decision}  (uncertainty: {r.get('uncertainty', 0):.2f})",
                expanded=(idx == 0),
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Face Score", f"{r.get('face_score') or 0:.3f}")
                c2.metric("Image Quality", f"{r.get('image_quality', 0):.2f}")
                c3.metric("Uncertainty", f"{r.get('uncertainty', 0):.3f}")

                st.markdown(
                    f"**Decision:** {badge_html} &nbsp; **Action:** `{action}` &nbsp; "
                    f"**Agent:** `{r.get('agent_type', 'unknown')}`",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Reasoning:** {r.get('reasoning', '—')}")

                trace = r.get("trace", [])
                if trace:
                    st.markdown("**Agent Graph Trace:**")
                    trace_html = "".join(
                        f'<div class="trace-step">{step}</div>' for step in trace
                    )
                    st.markdown(
                        f'<div style="margin-top:0.25rem">{trace_html}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Review Queue ──────────────────────────────────────────────────────
        pending_reviews = [
            q for q in st.session_state.review_queue if q.get("status") == "pending"
        ]
        if pending_reviews:
            st.divider()
            st.markdown(f"### Instructor Review Queue ({len(pending_reviews)} pending)")
            st.caption("Approve the agent's decision or override it with the correct attendance status.")

            for item in pending_reviews:
                with st.container(border=True):
                    r1, r2 = st.columns([3, 1.4])
                    with r1:
                        st.markdown(
                            f"**{item.get('student_name')}** &nbsp; "
                            f"`{item.get('action')}` &nbsp; "
                            f"uncertainty: `{float(item.get('uncertainty', 0)):.3f}`"
                        )
                        st.caption(item.get("reasoning", ""))
                    with r2:
                        ov = st.selectbox(
                            "Set attendance",
                            ["PRESENT", "LATE", "ABSENT"],
                            key=f"ov_sel_{item['queue_id']}",
                        )
                        if st.button(
                            f"Mark as {ov}",
                            key=f"ov_btn_{item['queue_id']}",
                            use_container_width=True,
                            type="primary",
                        ):
                            apply_review_action(item, "override", ov)
                            # Write the final decision to the database
                            sid = item.get("student_id")
                            if sid and st.session_state.current_lecture:
                                try:
                                    st.session_state.db.mark_attendance(
                                        lecture_id=st.session_state.current_lecture["id"],
                                        student_id=sid,
                                        marked_by=st.session_state.admin_user["id"],
                                        confidence_score=None,
                                        status=ov.lower(),
                                        notes=f"Manual override by instructor → {ov}",
                                    )
                                except Exception:
                                    pass
                            st.rerun()

            resolved = len(
                [q for q in st.session_state.review_queue if q.get("status") == "resolved"]
            )
            if resolved:
                st.success(f"{resolved} review(s) resolved this session.")


# ── Main app shell ─────────────────────────────────────────────────────────────
def main_app():
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0 0.5rem'>"
            "<div style='font-size:2rem'>🎯</div>"
            "<div style='font-weight:600;font-size:1.1rem;color:#f5f5f7'>CamAttend</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            f"<div style='font-size:0.875rem;color:#aeaeb2;line-height:1.8'>"
            f"<b style='color:#f5f5f7'>{st.session_state.organization['name']}</b><br>"
            f"<span style='color:#636366'>{st.session_state.admin_user['name']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Dashboard", "Enroll", "Recognize"],
            index=["Dashboard", "Enroll", "Recognize"].index(
                st.session_state.current_page
            ),
            label_visibility="collapsed",
        )
        st.session_state.current_page = page

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Agent status indicator
        agent_ok = st.session_state.agent is not None
        groq_ok = (
            agent_ok
            and getattr(st.session_state.agent, "groq_client", None) is not None
        )
        st.markdown(
            f"""<div class="cam-card" style="font-size:0.8rem;line-height:1.9">
                <div style="color:#aeaeb2;font-weight:600;margin-bottom:0.25rem">Agent Status</div>
                <div>{"🟢" if agent_ok else "🔴"} AI Agent</div>
                <div>{"🟢" if groq_ok else "⚪"} Groq LLM (reasoning)</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.db = None
            st.rerun()

        st.markdown(
            "<div style='position:fixed;bottom:1.5rem;font-size:0.75rem;color:#3a3a3c'>© 2026 CamAttend v2.0</div>",
            unsafe_allow_html=True,
        )

    initialize_models()

    # Persistent top navigation for easy page switching from anywhere.
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 2])
    with nav_col1:
        if st.button("Dashboard", use_container_width=True, type="primary" if st.session_state.current_page == "Dashboard" else "secondary", key="top_nav_dashboard"):
            st.session_state.current_page = "Dashboard"
            st.rerun()
    with nav_col2:
        if st.button("Enroll", use_container_width=True, type="primary" if st.session_state.current_page == "Enroll" else "secondary", key="top_nav_enroll"):
            st.session_state.current_page = "Enroll"
            st.rerun()
    with nav_col3:
        if st.button("Recognize", use_container_width=True, type="primary" if st.session_state.current_page == "Recognize" else "secondary", key="top_nav_recognize"):
            st.session_state.current_page = "Recognize"
            st.rerun()
    with nav_col4:
        st.caption("")

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    page = st.session_state.current_page

    if page == "Dashboard":
        dashboard_page()
    elif page == "Enroll":
        enroll_page()
    else:
        recognize_page()


# ── Entry point ────────────────────────────────────────────────────────────────
if st.session_state.admin_user is None:
    login_page()
else:
    main_app()
