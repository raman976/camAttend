import cv2
import streamlit as st
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime

from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.matcher import FaceMatcher
from database.supabase_db import SupabaseDB

load_dotenv()

st.set_page_config(
    page_title="CamAttend - Smart Attendance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    .stApp {
        background: #000000;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    h1 {
        color: #ffffff;
        font-weight: 600;
        font-size: 3.5rem !important;
        margin-bottom: 1rem !important;
        line-height: 1.1 !important;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #f5f5f7;
        font-weight: 600;
        font-size: 2.5rem !important;
        line-height: 1.2 !important;
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: #f5f5f7;
        font-weight: 600;
        font-size: 1.75rem !important;
        line-height: 1.3 !important;
    }
    
    p {
        color: #a1a1a6;
        font-size: 1.125rem;
        line-height: 1.6;
    }
    
    [data-testid="stSidebar"] {
        background: #000000;
        border-right: 1px solid #1d1d1f;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #f5f5f7 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #1d1d1f;
        color: #ffffff;
        border: 1px solid #424245;
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 17px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0071e3;
        outline: none;
        box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1);
    }
    
    .stTextInput > label {
        color: #f5f5f7 !important;
        font-weight: 500;
        font-size: 17px;
        margin-bottom: 8px;
    }
    
    .stButton > button {
        background: #0071e3;
        color: #ffffff;
        border: none;
        border-radius: 980px;
        padding: 12px 24px;
        font-weight: 400;
        font-size: 17px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #0077ed;
        transform: scale(1.02);
    }
    
    .stButton > button:active {
        background: #006edb;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #1d1d1f;
        border: 2px dashed #424245;
        border-radius: 18px;
        padding: 3rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0071e3;
        background-color: rgba(0, 113, 227, 0.05);
    }
    
    [data-testid="stFileUploader"] label {
        color: #a1a1a6 !important;
        font-size: 17px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 1px solid #424245;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #a1a1a6;
        border-radius: 0;
        padding: 12px 20px;
        font-weight: 400;
        font-size: 17px;
        border: none;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f5f5f7;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #ffffff !important;
        border-bottom: 2px solid #0071e3;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a1a1a6;
        font-weight: 400;
        font-size: 17px;
    }
    
    .stRadio > div {
        background-color: transparent;
    }
    
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] > p {
        color: #f5f5f7 !important;
        font-weight: 400;
        font-size: 17px;
    }
    
    .stSlider > div > div > div > div {
        background-color: #0071e3;
    }
    
    .stSlider > div > div > div {
        color: #f5f5f7;
    }
    
    .stSlider label {
        color: #f5f5f7 !important;
    }
    
    .stSuccess {
        background-color: rgba(0, 113, 227, 0.1);
        color: #0071e3;
        border: 1px solid #0071e3;
        border-radius: 12px;
        padding: 16px;
    }
    
    .stError {
        background-color: rgba(255, 59, 48, 0.1);
        color: #ff3b30;
        border: 1px solid #ff3b30;
        border-radius: 12px;
        padding: 16px;
    }
    
    .stInfo {
        background-color: rgba(0, 113, 227, 0.1);
        color: #0071e3;
        border: 1px solid #0071e3;
        border-radius: 12px;
        padding: 16px;
    }
    
    .stWarning {
        background-color: rgba(255, 159, 10, 0.1);
        color: #ff9f0a;
        border: 1px solid #ff9f0a;
        border-radius: 12px;
        padding: 16px;
    }
    
    .custom-card {
        background: transparent;
        border: none;
        border-radius: 18px;
        padding: 24px;
        margin: 2rem 0;
    }
    
    .custom-card:hover {
        background: rgba(255, 255, 255, 0.02);
    }
    
    .metric-card {
        background: transparent;
        border: none;
        border-radius: 18px;
        padding: 32px 24px;
        text-align: center;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.02);
    }
    
    .metric-value {
        font-size: 3.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 17px;
        color: #a1a1a6;
        font-weight: 400;
        text-transform: none;
        letter-spacing: 0;
    }
    
    .stProgress > div > div > div > div {
        background-color: #0071e3;
    }
    
    [data-testid="stImage"] {
        border-radius: 18px;
        overflow: hidden;
        border: none;
    }
    
    hr {
        border-color: #424245;
        margin: 3rem 0;
        opacity: 0.3;
    }
    
    .stCaption {
        color: #a1a1a6 !important;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

if 'db' not in st.session_state:
    st.session_state.db = SupabaseDB()

if 'admin_user' not in st.session_state:
    st.session_state.admin_user = None

if 'organization' not in st.session_state:
    st.session_state.organization = None

if 'detector' not in st.session_state:
    st.session_state.detector = None

if 'embedder' not in st.session_state:
    st.session_state.embedder = None

if 'matcher' not in st.session_state:
    st.session_state.matcher = None


def login_page():
    col1, col_center, col2 = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown("<div style='padding: 3rem 0 2rem 0; text-align: center;'></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📷 CamAttend</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.125rem; color: #a1a1a6; margin-bottom: 2.5rem;'>AI-Powered Attendance Management</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.markdown("<div style='padding: 1.5rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)
            
            email = st.text_input("Email", key="login_email", placeholder="your.email@example.com", label_visibility="visible")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password", label_visibility="visible")
            org_code = st.text_input("Organization Code", key="login_org", placeholder="Your organization code", label_visibility="visible")
            
            st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
            
            if st.button("Login", type="primary", use_container_width=True):
                if not email or not password or not org_code:
                    st.error("Please fill all fields")
                else:
                    with st.spinner("Verifying credentials..."):
                        org = st.session_state.db.get_organization_by_code(org_code)
                        if not org:
                            st.error("Organization not found")
                        else:
                            user = st.session_state.db.verify_user_password(email, password)
                            if not user:
                                st.error("Invalid email or password")
                            elif user['organization_id'] != org['id']:
                                st.error("You don't belong to this organization")
                            else:
                                st.session_state.admin_user = user
                                st.session_state.organization = org
                                st.success(f"Welcome back, {user['name']}!")
                                st.balloons()
                                st.rerun()
            
            st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #a1a1a6;'>Don't have an account? <span style='color: #0071e3; cursor: pointer;'>Register</span></p>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div style='padding: 1.5rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)
            
            name = st.text_input("Full Name", key="reg_name", placeholder="John Doe")
            email = st.text_input("Email", key="reg_email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", key="reg_password", placeholder="Create a password (min 6 characters)")
            org_code = st.text_input("Organization Code", key="reg_org", placeholder="Enter or create organization code")
            
            org_name = None
            if org_code:
                org = st.session_state.db.get_organization_by_code(org_code)
                if not org:
                    st.info("🎉 This organization code is available! Enter a name to create it.")
                    org_name = st.text_input("Organization Name", key="new_org_name", placeholder="e.g., My School, Tech Corp")
            
            st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
            
            if st.button("Create Account", type="primary", use_container_width=True):
                if not name or not email or not password or not org_code:
                    st.error("Please fill all fields")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    with st.spinner("Creating your account..."):
                        org = st.session_state.db.get_organization_by_code(org_code)
                        if not org:
                            if not org_name:
                                st.error("Please enter an organization name")
                            else:
                                org = st.session_state.db.create_organization(
                                    name=org_name,
                                    code=org_code,
                                    contact_email=email
                                )
                                if org:
                                    st.success(f"Organization '{org_name}' created!")
                                else:
                                    st.error("Failed to create organization")
                                    return
                        
                        if org:
                            user = st.session_state.db.create_user(
                                organization_id=org['id'],
                                email=email,
                                name=name,
                                password=password,
                                role='admin'
                            )
                            
                            if user:
                                st.success("Account created successfully! You can now login.")
                                st.snow()
                            else:
                                st.error("Registration failed. Email might already exist.")
            
            st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #a1a1a6;'>Already have an account? <span style='color: #0071e3; cursor: pointer;'>Login</span></p>", unsafe_allow_html=True)



def initialize_models():
    if st.session_state.detector is None:
        with st.spinner("Loading AI models..."):
            st.session_state.detector = FaceDetector()
            st.session_state.embedder = FaceEmbedder()
    
    if st.session_state.matcher is None:
        st.session_state.matcher = FaceMatcher()
        embeddings = st.session_state.db.get_student_embeddings(st.session_state.organization['id'])
        for student_id, name, emb in embeddings:
            st.session_state.matcher.add_embedding(emb, name)


def enroll_page():
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 2])
    with nav_col1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
    with nav_col2:
        st.button("📝 Enroll", use_container_width=True, disabled=True)
    with nav_col3:
        if st.button("🔍 Recognize", use_container_width=True):
            st.session_state.current_page = "Recognize Faces"
            st.rerun()
    
    st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>📝 Enroll New Student</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.25rem; color: #a1a1a6; margin-bottom: 3rem;'>Add a new student to your organization's attendance system</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Student Information</h3>", unsafe_allow_html=True)
        
        name = st.text_input("Full Name", placeholder="Enter student name")
        student_id_input = st.text_input("Student ID (Optional)", placeholder="Auto-generated if left blank")
        
        st.markdown("<div style='padding: 2rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Upload Photo</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a clear photo with one visible face", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a high-quality image"
        )
        
        st.markdown("###")
        
        if st.button("Enroll Student", type="primary", use_container_width=True) and uploaded_file and name:
            image = Image.open(uploaded_file)
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            progress = st.progress(0, "Detecting faces...")
            boxes, probs, landmarks = st.session_state.detector.detect_faces(image_np)
            
            if len(boxes) == 0:
                st.error("No face detected. Please upload a clear photo.")
                progress.empty()
                return
            
            if len(boxes) > 1:
                st.error(f"Multiple faces detected ({len(boxes)}). Use one face only.")
                progress.empty()
                return
            
            progress.progress(50, "Extracting features...")
            emb = st.session_state.embedder.get_embedding(image_np, bbox=boxes[0], landmark=landmarks[0])
            
            if emb is None:
                st.error("Failed to process face. Try another photo.")
                progress.empty()
                return
            
            progress.progress(75, "Saving to database...")
            
            student_id = student_id_input.upper() if student_id_input else name.replace(" ", "_").upper()
            
            student = st.session_state.db.enroll_student(
                organization_id=st.session_state.organization['id'],
                student_id=student_id,
                name=name,
                embedding=emb
            )
            
            progress.progress(100, "Complete!")
            
            if student:
                st.session_state.matcher.add_embedding(emb, name)
                st.success(f"✅ {name} enrolled successfully! (ID: {student_id})")
                st.balloons()
            else:
                st.error("Enrollment failed. Student ID might already exist.")
            
            progress.empty()
        
    with col2:
        if uploaded_file:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Preview</h3>", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
            
            st.markdown('<div style="background: #1d1d1f; border-radius: 18px; padding: 2rem; margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-bottom: 1.5rem;'>💡 Tips for Best Results</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style='color: #f5f5f7; font-size: 17px; line-height: 1.8;'>
            ✅ Use clear, well-lit photos<br>
            ✅ Face should be clearly visible<br>
            ✅ Only one person in frame<br>
            ✅ Front-facing position<br>
            ❌ Avoid sunglasses or masks<br>
            ❌ No blurry or dark images
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 4rem 2rem;">', unsafe_allow_html=True)
            st.markdown("# 📷")
            st.markdown("<h2 style='margin-top: 1.5rem; margin-bottom: 1rem;'>Upload Student Photo</h2>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.25rem; color: #a1a1a6;'>Select a clear photo to begin enrollment</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            embeddings = st.session_state.db.get_student_embeddings(st.session_state.organization['id'])
            total_students = len(embeddings)
            
            st.markdown("<div style='padding: 2rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>📊 Enrollment Statistics</h3>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Students", total_students)
            with col_b:
                st.metric("Organization", st.session_state.organization['code'])



def recognize_page():
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 2])
    with nav_col1:
        if st.button("📊 Dashboard", use_container_width=True, key="rec_dash"):
            st.session_state.current_page = "Dashboard"
            st.rerun()
    with nav_col2:
        if st.button("📝 Enroll", use_container_width=True, key="rec_enroll"):
            st.session_state.current_page = "Enroll Student"
            st.rerun()
    with nav_col3:
        st.button("🔍 Recognize", use_container_width=True, disabled=True, key="rec_recognize")
    
    st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🔍 Face Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.25rem; color: #a1a1a6; margin-bottom: 3rem;'>Upload a photo to identify students and mark attendance</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Settings</h3>", unsafe_allow_html=True)
        threshold = st.slider(
            "Recognition Threshold", 
            0.3, 0.9, 0.5, 0.05,
            help="Higher = more strict matching"
        )
        
        st.markdown("<div style='padding: 2rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Upload Photo</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a photo with student faces", 
            type=['jpg', 'jpeg', 'png'], 
            key="recog_upload"
        )
        
        st.markdown("###")
        
        if st.button("Recognize Faces", type="primary", use_container_width=True) and uploaded_file:
            image = Image.open(uploaded_file)
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            progress = st.progress(0, "Detecting faces...")
            boxes, probs, landmarks = st.session_state.detector.detect_faces(image_np)
            
            if len(boxes) == 0:
                st.error("No faces detected in the image")
                progress.empty()
                return
            
            progress.progress(40, f"Recognizing {len(boxes)} face(s)...")
            
            present_students = set()
            unknown_count = 0
            img = image_np.copy()
            
            for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                emb = st.session_state.embedder.get_embedding(image_np, bbox=box, landmark=landmark)
                
                if emb is None:
                    continue
                
                student, score = st.session_state.matcher.match(emb)
                x1, y1, x2, y2 = map(int, box)
                
                if score > threshold:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, student, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    present_students.add(student)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    unknown_count += 1
                
                progress.progress(40 + int((i + 1) / len(boxes) * 60))
            
            progress.progress(100, "Complete!")
            st.session_state.result_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            progress.empty()
            
            st.markdown("<div style='padding: 2rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Results</h3>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Faces Detected", len(boxes))
            with col_b:
                st.metric("Recognized", len(present_students))
            with col_c:
                st.metric("Unknown", unknown_count)
            
            if present_students:
                st.success("**Present:** " + ", ".join(sorted(present_students)))
            
            if unknown_count > 0:
                st.warning(f"{unknown_count} unknown face(s) detected. Please enroll them first.")
    
    with col2:
        if 'result_image' in st.session_state and st.session_state.result_image is not None:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Result</h3>", unsafe_allow_html=True)
            st.image(st.session_state.result_image, use_container_width=True)
            
            st.markdown('<div style="background: #1d1d1f; border-radius: 18px; padding: 2rem; margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-bottom: 1rem;'>Legend</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style='color: #f5f5f7; font-size: 17px; line-height: 1.8;'>
            🟢 Green Box = Recognized Student<br>
            🔴 Red Box = Unknown Person
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif uploaded_file:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Preview</h3>", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
            
            st.info("👆 Click 'Recognize Faces' button to process this image")
        else:
            st.markdown('<div style="text-align: center; padding: 4rem 2rem;">', unsafe_allow_html=True)
            st.markdown("# 🎯")
            st.markdown("<h2 style='margin-top: 1.5rem; margin-bottom: 1rem;'>Upload Group Photo</h2>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.25rem; color: #a1a1a6;'>Upload a photo to start recognition</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="background: #1d1d1f; border-radius: 18px; padding: 2rem; margin-top: 3rem;">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-bottom: 1.5rem;'>🎯 How it Works</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style='color: #f5f5f7; font-size: 17px; line-height: 2;'>
            1. Upload a clear group photo<br>
            2. AI detects all faces automatically<br>
            3. Matches with enrolled students<br>
            4. View labeled results instantly<br>
            5. Attendance marked automatically
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def dashboard_page():
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 2])
    with nav_col1:
        st.button("📊 Dashboard", use_container_width=True, disabled=True, key="dash_dash")
    with nav_col2:
        if st.button("📝 Enroll", use_container_width=True, key="dash_enroll"):
            st.session_state.current_page = "Enroll Student"
            st.rerun()
    with nav_col3:
        if st.button("🔍 Recognize", use_container_width=True, key="dash_recognize"):
            st.session_state.current_page = "Recognize Faces"
            st.rerun()
    
    st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>📊 Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1.25rem; color: #a1a1a6; margin-bottom: 3rem;'>Welcome to {st.session_state.organization['name']}</p>", unsafe_allow_html=True)
    
    embeddings = st.session_state.db.get_student_embeddings(st.session_state.organization['id'])
    total_students = len(embeddings)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Students", total_students)
    
    with col2:
        st.metric("🏢 Organization", st.session_state.organization['code'])
    
    with col3:
        st.metric("👤 Admin", st.session_state.admin_user['name'].split()[0])
    
    with col4:
        st.metric("✅ Status", "Active")
    
    st.markdown("###")
    
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        st.markdown("<h3 style='font-size: 1.75rem; margin-bottom: 1.5rem;'>👥 Enrolled Students</h3>", unsafe_allow_html=True)
        
        if total_students > 0:
            for i, (student_id, name, _) in enumerate(embeddings[:12]):
                st.markdown(f"""
                <div style="padding: 1.5rem; margin: 1rem 0; background: rgba(255, 255, 255, 0.03); border-radius: 16px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #ffffff; font-weight: 500; font-size: 1.25rem;">{name}</div>
                        <div style="color: #a1a1a6; font-size: 1rem; margin-top: 0.5rem;">ID: {student_id}</div>
                    </div>
                    <div style="background: #0071e3; color: white; padding: 10px 20px; border-radius: 50px; font-weight: 500; font-size: 1rem;">
                        #{i+1}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if total_students > 12:
                st.info(f"📋 Showing 12 of {total_students} students")
        else:
            st.markdown('<div style="text-align: center; padding: 4rem 2rem;">', unsafe_allow_html=True)
            st.markdown("# 📝")
            st.markdown("<h3 style='margin-top: 1.5rem; color: #f5f5f7;'>No Students Enrolled Yet</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.125rem; color: #a1a1a6; margin-top: 1rem;'>Start by enrolling your first student</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>🎯 Quick Actions</h3>", unsafe_allow_html=True)
        
        if st.button("➕ Enroll Student", use_container_width=True):
            st.session_state.current_page = "Enroll Student"
            st.rerun()
        
        if st.button("🔍 Recognize Faces", use_container_width=True):
            st.session_state.current_page = "Recognize Faces"
            st.rerun()
        
        st.markdown("<div style='padding: 2rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>ℹ️ System Info</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="line-height: 2.2; margin-top: 1.5rem;">
            <div style="margin-bottom: 1.5rem;">
                <div style="color: #a1a1a6; font-size: 15px; margin-bottom: 0.5rem;">Organization</div>
                <div style="color: #ffffff; font-size: 1.25rem; font-weight: 500;">{st.session_state.organization['name']}</div>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <div style="color: #a1a1a6; font-size: 15px; margin-bottom: 0.5rem;">Admin Email</div>
                <div style="color: #ffffff; font-size: 1.125rem;">{st.session_state.admin_user['email']}</div>
            </div>
            <div>
                <div style="color: #a1a1a6; font-size: 15px; margin-bottom: 0.5rem;">Current Date</div>
                <div style="color: #ffffff; font-size: 1.125rem;">{datetime.now().strftime('%B %d, %Y')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)



def main_app():
    with st.sidebar:
        st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: 3.5rem; margin-bottom: 0.5rem;">🎯</div>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; font-size: 1.75rem; font-weight: 600;">CamAttend</h2>', unsafe_allow_html=True)
        
        st.markdown("<div style='padding: 1.5rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div style='font-size: 1.125rem; font-weight: 500; color: #ffffff;'>{st.session_state.organization['name']}</div>", unsafe_allow_html=True)
        st.caption(f"Admin: {st.session_state.admin_user['name']}")
        
        st.markdown("<div style='padding: 2rem 0;'></div>", unsafe_allow_html=True)
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"
        
        default_index = ["Dashboard", "Enroll Student", "Recognize Faces"].index(st.session_state.current_page)
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "Enroll Student", "Recognize Faces"],
            index=default_index
        )
        
        st.session_state.current_page = page
        
        st.markdown("###")
        st.markdown("###")
        
        if st.button("🚪 Logout", use_container_width=True):
            for key in ['admin_user', 'organization', 'detector', 'embedder', 'matcher', 'result_image']:
                if key in st.session_state:
                    st.session_state[key] = None
            st.rerun()
        
        st.markdown("###")
        st.caption("© 2026 CamAttend v1.0")
    
    initialize_models()
    
    if page == "Dashboard":
        dashboard_page()
    elif page == "Enroll Student":
        enroll_page()
    else:
        recognize_page()


if st.session_state.admin_user is None:
    login_page()
else:
    main_app()


