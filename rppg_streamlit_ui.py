"""
Streamlit UI for rPPG Vitals Estimation
========================================

Clean, modular Streamlit interface wrapping refactored core modules.
Includes clear disclaimers, progress tracking, and patient-safe messaging.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import uuid
from datetime import datetime, timedelta
import base64
from io import BytesIO
import shutil
from typing import Optional

# Import translations module
from translations import get_text, get_available_languages, translate_dynamic, LANGUAGES
# Native Cloud-Safe camera recording
from live_camera import live_camera_interface
from streamlit_mic_recorder import speech_to_text
from dotenv import load_dotenv

# Load env vars
load_dotenv(override=True)

from rppg_refactored import (
    estimate_vitals_from_video,
    estimate_vitals_from_frames,
    HAVE_MEDIAPIPE,
    get_bp_category
)
try:
    from health_insights import get_health_insights, GROQ_API_KEY
    HAVE_GEMINI = True
except ImportError as e:
    print(f"Warning: Could not import health_insights: {e}")
    HAVE_GEMINI = False

try:
    from session_storage import (
        SessionData, save_session, load_session, 
        list_sessions, get_session_count
    )
    from pdf_report import generate_health_report
    from trend_analysis import get_trend_analysis, TrendAnalysis
    from chatbot import (
        build_chatbot_context, generate_chatbot_response,
        ChatMessage,
        load_chat_history, save_chat_history, clear_chat_history
    )
    HAVE_HISTORY = True
    HAVE_CHATBOT = True
except ImportError as e:
    print(f"Warning: Could not import history/chatbot modules: {e}")
    HAVE_HISTORY = False
    HAVE_CHATBOT = False

# Authentication
try:
    from auth import (
        create_user, authenticate_user, validate_email,
        validate_password, get_password_strength, User,
        update_user_language, get_google_auth_url, exchange_code_for_session,
        get_supabase_client
    )
    from s3_utils import generate_presigned_url, get_s3_client
    HAVE_AUTH = True
    HAVE_S3 = True
except ImportError:
    HAVE_AUTH = False
    HAVE_S3 = False

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Experimental rPPG Vitals Demo",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS / THEME
# ============================================================================

def inject_custom_css():
    st.markdown("""
        <style>
        /* Import a nice font if desired, e.g. Inter or Lato */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #2C3E30; /* Dark charcoal/green text */
        }

        /* Backgrounds */
        .stApp {
            background-color: #F5F5F0; /* Warm cream */
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #E8ECE6; /* Pale sage */
            border-right: 1px solid #D1D9D1;
        }

        /* Buttons (Primary and Secondary) */
        div.stButton > button {
            border-radius: 12px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px rgba(74, 103, 65, 0.1);
            transition: all 0.2s ease;
        }

        /* Primary Button Style Override */
        /* Streamlit handles primary color via config, but we can refine details */
        button[type="primary"], div.stButton > button[kind="primary"] {
            background-color: #4A6741 !important; /* Forest Green */
            color: white !important;
        }
        button[type="primary"]:hover, div.stButton > button[kind="primary"]:hover {
            background-color: #3A5232 !important;
            box-shadow: 0 6px 12px rgba(74, 103, 65, 0.2);
            transform: translateY(-1px);
        }

        /* Secondary Button Override */
        button[type="secondary"], div.stButton > button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #4A6741 !important;
            border: 1px solid #4A6741 !important;
        }
        button[type="secondary"]:hover, div.stButton > button[kind="secondary"]:hover {
            background-color: #F0F5F0 !important;
        }

        /* Inputs */
        div[data-baseweb="input"] > div {
            background-color: #E8ECE6 !important;
            border-radius: 10px !important;
            border: 1px solid #CFD8CF !important;
            color: #2C3E30 !important;
        }
        /* Focus state for inputs */
        div[data-baseweb="input"] > div:focus-within {
            border-color: #4A6741 !important;
            box-shadow: 0 0 0 2px rgba(74, 103, 65, 0.2) !important;
        }

        /* Cards / Expanders */
        div[data-testid="stExpander"] {
            background-color: #FFFFFF;
            border-radius: 12px;
            border: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            overflow: hidden;
        }
        .streamlit-expanderHeader {
            background-color: #FFFFFF;
            color: #2C3E30;
            font-weight: 600;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            color: #4A6741; /* Forest green numbers */
        }
        
        /* Titles and Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #2C3E30;
            font-weight: 700;
        }
        
        /* Links */
        a {
            color: #4A6741 !important;
        }

        /* Dividers */
        hr {
            border-color: #D1D9D1;
        }
        
        /* Toast/Alerts */
        div[data-baseweb="toast"] {
            background-color: #4A6741;
            color: white;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# Inject the styles
inject_custom_css()

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

@st.cache_resource
def get_singleton_supabase_client():
    """
    Get a singleton Supabase client instance.
    This persists across reruns and sessions for the same server process.
    Required for PKCE flow to maintain the code verifier.
    """
    return get_supabase_client()

def get_persistent_supabase_client():
    """Wrapper to get the singleton client"""
    return get_singleton_supabase_client()

def handle_oauth_callback():
    """Check for OAuth code in query params and exchange for session"""
    if not HAVE_AUTH:
        return

    # 1. Check if we already have a session (e.g. from cookie)
    # This avoids re-triggering auth flow if we are already logged in
    try:
        client = get_persistent_supabase_client()
        if client:
            session = client.auth.get_session()
            if session:
                # We have a valid session, ensure state is set
                st.session_state["authenticated"] = True
                user = session.user
                st.session_state["user_email"] = user.email
                # Try to get existing user info from DB or metadata
                name = user.user_metadata.get('full_name', user.email.split('@')[0])
                st.session_state["user_name"] = name
                # Stop processing code if we are already logged in
                return
    except Exception as e:
        print(f"Session check error: {e}")

    # 2. Check query params for 'code'
    try:
        query_params = st.query_params
        code = query_params.get("code")
        
        if code:
            st.toast("🔄 Processing Google Sign-In...", icon="🔄")
            
            # Use persistent singleton client to verify code (PKCE)
            supabase = get_persistent_supabase_client()
            
            # Exchange code
            success, user, message = exchange_code_for_session(code, supabase_client=supabase)
            
            if success:
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = user.email
                st.session_state["user_name"] = user.name
                # Set user's language preference
                st.session_state["user_language"] = user.language
                
                # Clear query params to prevent re-processing
                st.query_params.clear()
                
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ Google Sign-In failed: {message}")
    except Exception as e:
        print(f"Error in OAuth callback: {e}")

# Call immediately after page config
if HAVE_AUTH:
    handle_oauth_callback()

def check_authentication() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def get_current_user_email() -> Optional[str]:
    """Get current user email"""
    return st.session_state.get("user_email", None)

def get_current_user_name() -> str:
    """Get current user name"""
    return st.session_state.get("user_name", "User")

def logout():
    """Logout current user"""
    # Sign out from Supabase to invalidate session
    try:
        supabase = get_persistent_supabase_client()
        if supabase:
            supabase.auth.sign_out()
    except Exception as e:
        print(f"Error signing out from Supabase: {e}")

    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.session_state["user_name"] = None
    # Clear other session data
    st.session_state.pop("chat_messages", None)
    st.session_state.pop("viewing_history", None)
    st.session_state.pop("viewing_trends", None)
    
    # Clear profile data to force re-verification on next login
    st.session_state.pop("profile_completed", None)
    for key in list(st.session_state.keys()):
        if key.startswith("profile_"):
            st.session_state.pop(key, None)
            
    st.rerun()

# ============================================================================
# LANGUAGE MANAGEMENT
# ============================================================================

def get_current_language() -> str:
    """Get current user's language preference"""
    return st.session_state.get("user_language", "en")

def set_language(lang: str):
    """Set language and persist to database if user is logged in"""
    st.session_state["user_language"] = lang
    
    # Persist to database if user is authenticated
    if check_authentication() and HAVE_AUTH:
        email = get_current_user_email()
        if email:
            update_user_language(email, lang)

def t(key: str) -> str:
    """Translation helper - get text for current language"""
    return get_text(key, get_current_language())

def show_login_page():
    """Display login page"""
    # Language selector at top
    lang_options = {code: f"{info['flag']} {info['name']}" for code, info in LANGUAGES.items()}
    selected_lang = st.selectbox(
        t("language_label"),
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(get_current_language()),
        key="login_lang_selector"
    )
    if selected_lang != get_current_language():
        set_language(selected_lang)
        st.rerun()
    
    st.title(f"🩺 {t('login_title')}")
    st.caption(t("login_subtitle"))
    
    with st.form("login_form"):
        email = st.text_input(t("email_label"), placeholder=t("email_placeholder"))
        password = st.text_input(t("password_label"), type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button(t("login_button"), type="primary", use_container_width=True)
        with col2:
            signup_btn = st.form_submit_button(t("signup_button"), use_container_width=True)
        
        if signup_btn:
            st.session_state["show_signup"] = True
            st.rerun()

        # Google Sign-In Button
        st.markdown(
            """
            <div style="text-align: center; color: #666; margin: 10px 0; font-size: 0.9em;">
                OR
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Use persistent client to generate URL (stores PKCE verifier)
        google_auth_url = get_google_auth_url(supabase_client=get_persistent_supabase_client())
        
        if google_auth_url:
            st.markdown(
                f'''
                <a href="{google_auth_url}" target="_self" style="text-decoration:none;">
                    <button style="
                        width: 100%;
                        padding: 0.5rem;
                        background-color: white;
                        color: #333;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 10px;
                        cursor: pointer;
                        font-weight: 500;
                        margin-bottom: 10px;
                    ">
                        <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" width="18" height="18">
                        {t('login_button')} with Google
                    </button>
                </a>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.warning("Google Sign-In not configured")
        
        if submit:
            if not email or not password:
                st.error(t("enter_email_password"))
            else:
                with st.spinner(f"{t('loading')}..."):
                    success, user, message = authenticate_user(email, password)
                    
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = user.email
                        st.session_state["user_name"] = user.name
                        # Set user's language preference
                        st.session_state["user_language"] = user.language
                        st.success(f"✅ {t('login_success')}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
    
    st.divider()
    st.info(f"💡 {t('new_to_wellio')}")

def show_signup_page():
    """Display sign-up page"""
    st.title("🩺 Wellio - Sign Up")
    st.caption("Create your account to start monitoring your health")
    
    with st.form("signup_form"):
        name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password", 
                                 help="At least 8 characters, with uppercase, lowercase, and number")
        password_confirm = st.text_input("Confirm Password", type="password")
        
        # Show password strength if password entered
        if password:
            strength = get_password_strength(password)
            if strength == "Weak":
                st.warning(f"Password Strength: {strength}")
            elif strength == "Medium":
                st.info(f"Password Strength: {strength}")
            else:
                st.success(f"Password Strength: {strength}")
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        with col2:
            login_btn = st.form_submit_button("Back to Login", use_container_width=True)
        
        if login_btn:
            st.session_state["show_signup"] = False
            st.rerun()
        
        if submit:
            # Validation
            if not all([name, email, password, password_confirm]):
                st.error("❌ Please fill in all fields")
            elif password != password_confirm:
                st.error("❌ Passwords do not match")
            else:
                # Validate email
                valid_email, email_msg = validate_email(email)
                if not valid_email:
                    st.error(f"❌ {email_msg}")
                else:
                    # Validate password
                    valid_pass, pass_msg = validate_password(password)
                    if not valid_pass:
                        st.error(f"❌ {pass_msg}")
                    else:
                        # Create user
                        with st.spinner("Creating account..."):
                            success, message = create_user(email, password, name)
                            
                            if success:
                                st.success(f"✅ {message} Please login.")
                                st.session_state["show_signup"] = False
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
    
    st.divider()
    st.info("🔒 **Your data is secure**: Passwords are encrypted using industry-standard bcrypt hashing.")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for storage"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

def base64_to_image(img_base64: str):
    """Convert base64 string back to displayable image"""
    img_data = base64.b64decode(img_base64)
    return BytesIO(img_data)

# ============================================================================
# AUTHENTICATION CHECK
# ============================================================================

# Check if authentication is enabled and user is authenticated
if HAVE_AUTH:
    if not check_authentication():
        # Show login or signup page
        if st.session_state.get("show_signup", False):
            show_signup_page()
        else:
            show_login_page()
        st.stop()  # Don't show rest of app

# ============================================================================
# COMPULSORY PROFILE GATE
# ============================================================================
# Check if profile is completed
if not st.session_state.get("profile_completed", False):
    st.title(f"👤 {t('user_profile_title')}")
    st.info(f"ℹ️ {t('profile_required_info')}")
    
    with st.form("main_profile_form"):
        age = st.number_input(t("age_label"), min_value=0, max_value=120, value=30, key="main_profile_age")
        
        # Gender options
        gender_options = ["prefer_not_say", "female", "male", "other"]
        gender_display = {opt: t(opt) for opt in gender_options}
        gender = st.selectbox(
            t("gender_label"), 
            options=gender_options,
            format_func=lambda x: gender_display[x],
            index=0, 
            key="main_profile_gender"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input(t("height_label"), min_value=50, max_value=250, value=170, key="main_profile_height")
        with col2:
            weight = st.number_input(t("weight_label"), min_value=20, max_value=300, value=70, key="main_profile_weight")
        
        # Diet options
        diet_options = ["non_vegetarian", "vegetarian", "vegan", "other"]
        diet_display = {opt: t(opt) for opt in diet_options}
        diet = st.selectbox(
            t("diet_label"),
            options=diet_options,
            format_func=lambda x: diet_display[x],
            index=0,
            key="main_profile_diet"
        )
        
        # Exercise options
        exercise_options = ["never", "exercise_1_2", "exercise_3_4", "daily"]
        exercise_display = {opt: t(opt) for opt in exercise_options}
        exercise = st.selectbox(
            t("exercise_label"),
            options=exercise_options,
            format_func=lambda x: exercise_display[x],
            index=2,
            key="main_profile_exercise"
        )
        
        sleep = st.number_input(t("sleep_label"), min_value=0.0, max_value=24.0, value=7.0, step=0.5, key="main_profile_sleep")
        
        col1, col2 = st.columns(2)
        with col1:
            # Smoking options
            smoking_options = ["never", "occasional", "regular", "former"]
            smoking_display = {opt: t(opt) for opt in smoking_options}
            smoking = st.selectbox(
                t("smoking_label"),
                options=smoking_options,
                format_func=lambda x: smoking_display[x],
                index=0,
                key="main_profile_smoking"
            )
        with col2:
            # Drinking options
            drinking_options = ["never", "occasional", "regular", "former"]
            drinking_display = {opt: t(opt) for opt in drinking_options}
            drinking = st.selectbox(
                t("drinking_label"),
                options=drinking_options,
                format_func=lambda x: drinking_display[x],
                index=0,
                key="main_profile_drinking"
            )
        
        st.markdown("---")
        submitted = st.form_submit_button(t("save_profile_button"), type="primary", use_container_width=True)
        
        if submitted:
            # Sync to sidebar keys to ensure consistency if sidebar is rendered later
            st.session_state["profile_age"] = age
            st.session_state["profile_gender"] = gender
            st.session_state["profile_height"] = height
            st.session_state["profile_weight"] = weight
            st.session_state["profile_diet"] = diet
            st.session_state["profile_exercise"] = exercise
            st.session_state["profile_sleep"] = sleep
            st.session_state["profile_smoking"] = smoking
            st.session_state["profile_drinking"] = drinking
            
            st.session_state["profile_completed"] = True
            st.success(f"✅ {t('profile_saved')}")
            st.rerun()
            
    # Logout option in case they want to switch user
    if st.button(t("logout_button"), type="secondary"):
        logout()
        
    st.stop() # BLOCK REST OF APP UNLESS PROFILE IS SAVED

# ============================================================================
# SIDEBAR CONFIG
# ============================================================================

# Show user info if authenticated
if HAVE_AUTH and check_authentication():
    st.sidebar.title("🩺 Wellio")
    st.sidebar.caption("rPPG Vitals Estimation")
    
    # Language selector at the top
    st.sidebar.divider()
    lang_options = {code: f"{info['flag']} {info['name']}" for code, info in LANGUAGES.items()}
    selected_lang = st.sidebar.selectbox(
        t("language_label"),
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(get_current_language()),
        key="sidebar_lang_selector"
    )
    if selected_lang != get_current_language():
        set_language(selected_lang)
        st.rerun()
    
    # User info
    st.sidebar.divider()
    user_name = get_current_user_name()
    user_email = get_current_user_email()
    st.sidebar.write(f"👤 **{user_name}**")
    st.sidebar.caption(user_email)
    
    # Logout button
    if st.sidebar.button(t("logout_button"), type="secondary", use_container_width=True):
        logout()
    
    st.sidebar.divider()

st.sidebar.title(f"⚙️ {t('settings_title')}")
use_mediapipe = False  # Using Haar Cascade for better compatibility

show_advanced = st.sidebar.checkbox(t("show_advanced_plots"), value=True)

# USER PROFILE (SIDEBAR)
# ============================================================================
with st.sidebar.expander(t("user_profile_title"), expanded=True):
    with st.form("user_profile", clear_on_submit=False):
        # Initialize defaults if not present
        if "profile_age" not in st.session_state: st.session_state["profile_age"] = 30
        age = st.number_input(t("age_label"), min_value=0, max_value=120, key="profile_age")
        
        # Gender options
        gender_options = ["prefer_not_say", "female", "male", "other"]
        gender_display = {opt: t(opt) for opt in gender_options}
        if "profile_gender" not in st.session_state: st.session_state["profile_gender"] = gender_options[0]
        gender = st.selectbox(
            t("gender_label"), 
            options=gender_options,
            format_func=lambda x: gender_display[x],
            key="profile_gender"
        )
        
        if "profile_height" not in st.session_state: st.session_state["profile_height"] = 170
        height = st.number_input(t("height_label"), min_value=50, max_value=250, key="profile_height")
        
        if "profile_weight" not in st.session_state: st.session_state["profile_weight"] = 70
        weight = st.number_input(t("weight_label"), min_value=20, max_value=300, key="profile_weight")
        
        # Diet options
        diet_options = ["non_vegetarian", "vegetarian", "vegan", "other"]
        diet_display = {opt: t(opt) for opt in diet_options}
        if "profile_diet" not in st.session_state: st.session_state["profile_diet"] = diet_options[0]
        diet = st.selectbox(
            t("diet_label"),
            options=diet_options,
            format_func=lambda x: diet_display[x],
            key="profile_diet"
        )
        
        # Exercise options
        exercise_options = ["never", "exercise_1_2", "exercise_3_4", "daily"]
        exercise_display = {opt: t(opt) for opt in exercise_options}
        if "profile_exercise" not in st.session_state: st.session_state["profile_exercise"] = exercise_options[2]
        exercise = st.selectbox(
            t("exercise_label"),
            options=exercise_options,
            format_func=lambda x: exercise_display[x],
            key="profile_exercise"
        )
        
        if "profile_sleep" not in st.session_state: st.session_state["profile_sleep"] = 7.0
        sleep = st.number_input(t("sleep_label"), min_value=0.0, max_value=24.0, step=0.5, key="profile_sleep")
        
        # Smoking options
        smoking_options = ["never", "occasional", "regular", "former"]
        smoking_display = {opt: t(opt) for opt in smoking_options}
        if "profile_smoking" not in st.session_state: st.session_state["profile_smoking"] = smoking_options[0]
        smoking = st.selectbox(
            t("smoking_label"),
            options=smoking_options,
            format_func=lambda x: smoking_display[x],
            key="profile_smoking"
        )
        
        # Drinking options
        drinking_options = ["never", "occasional", "regular", "former"]
        drinking_display = {opt: t(opt) for opt in drinking_options}
        if "profile_drinking" not in st.session_state: st.session_state["profile_drinking"] = drinking_options[0]
        drinking = st.selectbox(
            t("drinking_label"),
            options=drinking_options,
            format_func=lambda x: drinking_display[x],
            key="profile_drinking"
        )
        
        submitted = st.form_submit_button(t("save_profile_button"))
        if submitted:
            st.session_state["profile_completed"] = True
            st.sidebar.success(f"✅ {t('profile_saved')}")

# ============================================================================
# USAGE HISTORY (SIDEBAR)
# ============================================================================

if HAVE_HISTORY:
    st.sidebar.divider()
    st.sidebar.subheader(f"📜 {t('history_sidebar_title')}")
    
    # Use authenticated user's email as username
    username = get_current_user_email() or "default_user"
    
    # Get session count
    session_count = get_session_count(username)
    st.sidebar.caption(f"{t('total_sessions')}: {session_count}")
    
    # List recent sessions
    if session_count > 0:
        # Get all sessions but only show top 3 in sidebar
        sessions = list_sessions(username, limit=10) # Get enough to show we have more
        
        st.sidebar.caption(f"{t('recent_analyses')}")
        
        # Show only first 3
        for session in sessions[:3]:
            try:
                timestamp_dt = datetime.fromisoformat(session.timestamp)
                # Convert to IST (UTC+5:30) for display
                timestamp_dt = timestamp_dt + timedelta(hours=5, minutes=30)
                timestamp_str = timestamp_dt.strftime("%d %b %Y · %I:%M %p")
            except:
                timestamp_str = "Unknown date"
            
            # Create button for each session
            button_label = f"🩺 {timestamp_str}"
            if st.sidebar.button(button_label, key=f"hist_{session.session_id}"):
                st.session_state["selected_session_id"] = session.session_id
                st.session_state["viewing_history"] = True
                st.session_state["viewing_all_history"] = False # Exit view all mode
                st.rerun()
        
        # View All Button if we have more than 3
        if len(sessions) > 3:
            if st.sidebar.button(f"📂 {t('view_all_history') if 'view_all_history' in locals() else 'View All Past Reports'}", type="secondary", use_container_width=True):
                st.session_state["viewing_all_history"] = True
                st.session_state["viewing_history"] = False
                st.session_state["viewing_trends"] = False
                st.rerun()

    else:
        st.sidebar.info(t("no_history"))
    
    # Trends button (only show if user has at least 2 sessions)
    if session_count >= 2:
        st.sidebar.divider()
        if st.sidebar.button(f"📊 {t('view_trends_button')}", use_container_width=True, type="secondary"):
            st.session_state["viewing_trends"] = True
            st.session_state["viewing_history"] = False
            st.session_state["viewing_all_history"] = False
            st.rerun()

# ============================================================================

# ============================================================================
# MAIN APP
# ============================================================================

st.title(f"🩺 {t('app_title')}", help=t('mandatory_disclaimer'))

# ============================================================================
# HISTORICAL SESSION REPLAY
# ============================================================================

if HAVE_HISTORY and st.session_state.get("viewing_history", False):
    session_id = st.session_state.get("selected_session_id")
    username = get_current_user_email() or "default_user"
    
    if session_id:
        session = load_session(username, session_id)
        
        if session:
            # Header for historical view
            st.info(f"📜 **{t('viewing_history')}** | {t('back_to_new_analysis_instruction')}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                try:
                    timestamp_dt = datetime.fromisoformat(session.timestamp)
                    st.caption(f"{t('analysis_date')}: {timestamp_dt.strftime('%d %B %Y at %I:%M %p')}")
                except:
                    st.caption(f"{t('analysis_date')}: {t('unknown')}")
            
            with col2:
                if st.button(f"← {t('back_to_new_analysis_button')}"):
                    st.session_state["viewing_history"] = False
                    st.session_state.pop("selected_session_id", None)
                    st.rerun()
            
            st.divider()
            
            # Display vitals
            st.subheader(t("vital_signs"))
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(t("estimated_pulse"), f"{session.heart_rate:.1f} BPM",
                         help=f"{t('confidence')}: {session.heart_rate_confidence}")
                
                # Add label below pulse based on BPM ranges (User Request)
                try:
                    bpm_val = float(session.heart_rate)
                except Exception:
                    bpm_val = None

                if bpm_val is None:
                    pulse_label = t("na")
                    color = "#6b7280"
                else:
                    if bpm_val < 50:
                        pulse_label = "⬇️ Low"
                        color = "#3b82f6" # Blue
                    elif bpm_val <= 60:
                        pulse_label = "⬇️ Low-Normal"
                        color = "#60a5fa" # Light Blue
                    elif bpm_val <= 80:
                        pulse_label = "✅ Normal"
                        color = "#16a34a" # Green
                    elif bpm_val <= 100:
                        pulse_label = "⬆️ High-Normal"
                        color = "#f59e0b" # Orange
                    else:
                        pulse_label = "⬆️ High"
                        color = "#dc2626" # Red

                st.markdown(
                    f"<div style='display:inline-block;padding:6px 12px;border-radius:8px;background:{color};color:white;font-weight:600;font-size:14px;margin-top:6px'>{pulse_label}</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                # Use stored backend stress data if available
                stress_lbl = getattr(session, "stress_label", "Unavailable")
                stress_scr = getattr(session, "stress_score", None)
                
                # Fallback for old sessions
                if stress_lbl == "Unavailable" and hasattr(session, "stress_level"):
                     val = session.stress_level
                     if val < 2: stress_lbl = "Normal"
                     elif val < 5: stress_lbl = "Mild Stress" 
                     else: stress_lbl = "High Stress"
                
                stress_display = f"{stress_lbl}"
                if stress_scr is not None:
                    stress_display += f" ({stress_scr}/10)"
                    
                st.metric(t("stress_index"), stress_display,
                         help=getattr(session, "stress_reason", t("experimental_stress")))
            
            with col3:
                sbp = session.bp_systolic
                dbp = session.bp_diastolic
                
                if sbp and dbp:
                    bp_val = f"{sbp:.0f}/{dbp:.0f} mmHg"
                    # Always recalculate label for consistency
                    bp_lbl = get_bp_category(sbp, dbp)
                    
                    if "High-Normal" in bp_lbl: bp_color = "#f59e0b"
                    elif "High" in bp_lbl: bp_color = "#dc2626"
                    elif "Normal" in bp_lbl: bp_color = "#16a34a"
                    elif "Low" in bp_lbl: bp_color = "#3b82f6"
                    
                    st.metric(t("estimated_bp"), bp_val)
                    st.markdown(
                        f"<div style='display:inline-block;padding:6px 12px;border-radius:8px;background:{bp_color};color:white;font-weight:600;font-size:14px;margin-top:6px'>{bp_lbl}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.metric(t("estimated_bp"), t("na"))
            
            with col4:
                if session.spo2:
                    st.metric(t("estimated_spo2"), f"{session.spo2:.1f}%")
                else:
                    st.metric(t("estimated_spo2"), t("na"))
            
            st.divider()
            
            # Risk Assessment
            st.subheader(f"⚠️ {t('risk_assessment')}")
            
            if session.risk_score <= 3:
                display_label = t("low_risk")
                color = "#16a34a"
            elif session.risk_score <= 6:
                display_label = t("moderate_risk")
                color = "#f59e0b"
            else:
                display_label = t("high_risk")
                color = "#dc2626"
            
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:20px'>"
                f"<div style='font-weight:600;font-size:16px'>Risk Score:</div>"
                f"<div style='font-size:24px;font-weight:700'>{session.risk_score}/10</div>"
                f"<div style='padding:8px 16px;border-radius:12px;background:{color};color:white;font-weight:700;font-size:16px'>{display_label}</div>"
                f"</div>", unsafe_allow_html=True
            )
            
            if session.risk_factors:
                st.write(f"**{t('risk_factors')}:**")
                for factor in session.risk_factors:
                    st.write(f"• {factor}")
            
            if session.protective_factors:
                st.write(f"**{t('protective_factors')}:**")
                for factor in session.protective_factors:
                    st.write(f"• {factor}")
            
            st.divider()
            
            # AI Insights
            if session.detailed_analysis or session.recommendations or session.symptoms_to_watch:
                st.subheader(f"💡 {t('health_insights_title')}")
                
                if session.detailed_analysis:
                    st.write(session.detailed_analysis)
                
                col_1, col_2 = st.columns([1, 1])
                
                with col_1:
                    if session.recommendations:
                        with st.expander(f"💪 {t('recommendations_title')}", expanded=True):
                            for rec in session.recommendations:
                                st.write(f"• {rec}")
                
                with col_2:
                    if session.symptoms_to_watch:
                        with st.expander(f"🚨 {t('symptoms_watch_title')}", expanded=True):
                            for symptom in session.symptoms_to_watch:
                                st.write(f"• {symptom}")
                
                st.divider()
            
            # Visualizations (Removed per user request)
            # if session.signal_plot or session.hrv_plot:
            #     st.subheader(f"📈 {t('signal_processing_title')}")
            #     
            #     if session.signal_plot:
            #         st.image(base64_to_image(session.signal_plot), caption=t("filtered_ppg"))
            #     
            #     if session.hrv_plot:
            #         st.subheader(f"💓 {t('hrv_title')}")
            #         st.image(base64_to_image(session.hrv_plot), caption=t("rr_interval_analysis"))
            #         st.info(f"**{t('hrv_summary')}:** SDNN: {session.hrv_sdnn:.1f} ms | pNN50: {session.hrv_pnn50:.1f}% | {t('beats_detected')}: {session.rr_intervals_count}")
            
            # PDF Download for historical session
            st.divider()
            st.subheader(f"📄 {t('download_report')}")
            
            if st.button(t("generate_pdf"), type="secondary", key="hist_pdf"):
                with st.spinner(f"{t('generating_pdf')}..."):
                    try:
                        pdf_bytes = generate_health_report(session)
                        st.download_button(
                            label=f"💾 {t('download_pdf')}",
                            data=pdf_bytes,
                            file_name=f"Health_Report_{session.session_id[:8]}.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                        st.success(f"✅ {t('pdf_generated')}")
                    except Exception as e:
                        st.error(f"{t('pdf_error')}: {str(e)}")
            
            st.stop()  # Don't show the upload section when viewing history
        else:
            st.error(t("session_not_found"))
            if st.button(f"← {t('back_to_new_analysis_button')}"):
                st.session_state["viewing_history"] = False
                st.session_state.pop("selected_session_id", None)
                st.rerun()
            st.stop()


# ============================================================================
# ALL HISTORY BOARD
# ============================================================================

if HAVE_HISTORY and st.session_state.get("viewing_all_history", False):
    username = get_current_user_email() or "default_user"
    
    # Header
    st.title(f"📂 {t('history_sidebar_title')}")
    st.caption("Browse all your past health analysis reports.")
    
    # Navigation
    if st.button(f"🏠 {t('back_to_home')}", type="secondary"):
        st.session_state["viewing_all_history"] = False
        st.rerun()
        
    st.divider()
    
    # Fetch all sessions
    all_sessions = list_sessions(username, limit=100) # Reasonable limit
    
    if not all_sessions:
        st.info(t("no_history"))
    else:
        # Display as a grid of cards
        for session in all_sessions:
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                
                try:
                    timestamp_dt = datetime.fromisoformat(session.timestamp) + timedelta(hours=5, minutes=30)
                    date_str = timestamp_dt.strftime("%d %B %Y")
                    time_str = timestamp_dt.strftime("%I:%M %p")
                except:
                    date_str = "Unknown"
                    time_str = ""
                
                with c1:
                    st.markdown(f"**🗓️ {date_str}**")
                    st.caption(f"🕒 {time_str}")
                
                with c2:
                    st.markdown(f"**❤️ {session.heart_rate:.0f} BPM**")
                    
                    # HR Label
                    try:
                        bpm = float(session.heart_rate)
                        if bpm < 50: lbl="⬇️ Low"
                        elif bpm <= 60: lbl="⬇️ Low-Normal"
                        elif bpm <= 80: lbl="✅ Normal"
                        elif bpm <= 100: lbl="⬆️ High-Normal"
                        else: lbl="⬆️ High"
                    except:
                        lbl = "N/A"
                        
                    st.caption(f"{lbl}")
                    
                    # Stress Label from backend
                    s_lbl = getattr(session, "stress_label", "N/A")
                    if s_lbl == "Unavailable": s_lbl = "N/A"
                    st.caption(f"Stress: {s_lbl}")
                    st.caption(f"Risk Level: {session.risk_level}")
                    
                with c3:
                    if st.button("Open", key=f"open_all_{session.session_id}", use_container_width=True, type="primary"):
                        st.session_state["selected_session_id"] = session.session_id
                        st.session_state["viewing_history"] = True
                        st.session_state["viewing_all_history"] = False
                        st.rerun()
    
    st.stop()


# ============================================================================
# TREND ANALYSIS VIEW
# ============================================================================

if HAVE_HISTORY and st.session_state.get("viewing_trends", False):
    username = get_current_user_email() or "default_user"
    
    # Header
    st.title(f"📊 {t('trends_title')}")
    st.caption(t("trends_subtitle"))
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(f"🏠 {t('back_to_home')}", type="secondary"):
            st.session_state["viewing_trends"] = False
            st.rerun()
    
    st.divider()
    
    # Time period selector
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📅 7 Days", use_container_width=True, 
                    type="primary" if st.session_state.get("trend_period", 30) == 7 else "secondary"):
            st.session_state["trend_period"] = 7
            st.rerun()
    with col2:
        if st.button("📅 14 Days", use_container_width=True,
                    type="primary" if st.session_state.get("trend_period", 30) == 14 else "secondary"):
            st.session_state["trend_period"] = 14
            st.rerun()
    with col3:
        if st.button("📅 30 Days", use_container_width=True,
                    type="primary" if st.session_state.get("trend_period", 30) == 30 else "secondary"):
            st.session_state["trend_period"] = 30
            st.rerun()
    with col4:
        if st.button("📅 90 Days", use_container_width=True,
                    type="primary" if st.session_state.get("trend_period", 30) == 90 else "secondary"):
            st.session_state["trend_period"] = 90
            st.rerun()
    
    period = st.session_state.get("trend_period", 30)
    user_age = st.session_state.get("profile_age", 30)
    
    # Get trend analysis
    with st.spinner(f"Analyzing trends over the last {period} days..."):
        trend_analysis = get_trend_analysis(username, days=period, user_age=user_age)
    
    if trend_analysis is None:
        st.warning(f"⚠️ Not enough data for trend analysis in the last {period} days. Complete at least 2 analyses to unlock trends.")
        st.stop()
    
    # Summary
    st.divider()
    st.subheader(f"📋 {t('summary')}")
    st.info(trend_analysis.summary)
    
    # Key findings
    if trend_analysis.key_findings:
        with st.expander(f"🔍 {t('key_findings')}", expanded=True):
            for finding in trend_analysis.key_findings:
                st.write(f"• {finding}")
    
    # Recommendations
    if trend_analysis.recommendations:
        with st.expander(f"💡 {t('recommendations')}", expanded=True):
            for rec in trend_analysis.recommendations:
                st.write(f"• {rec}")
    
    st.divider()
    
    # Trend charts
    st.subheader(f"📈 {t('metric_trends')}")
    
    # Heart Rate
    if trend_analysis.heart_rate:
        hr = trend_analysis.heart_rate
        st.markdown(f"### ❤️ {hr.metric_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("average"), f"{hr.average:.1f} BPM")
        with col2:
            st.metric(t("min"), f"{hr.min_value:.1f} BPM")
        with col3:
            st.metric(t("max"), f"{hr.max_value:.1f} BPM")
        with col4:
            # Trend indicator
            if hr.trend_direction == "up":
                st.metric(t("trend"), t("increasing_arrow"), delta=f"{hr.percent_change:.1f}%", delta_color="inverse")
            elif hr.trend_direction == "down":
                st.metric(t("trend"), t("decreasing_arrow"), delta=f"{hr.percent_change:.1f}%", delta_color="normal")
            else:
                st.metric(t("trend"), t("stable_arrow"), delta="0%")
        
        # Status badge
        status_colors = {
            "Improving": "#16a34a",
            "Stable": "#3b82f6",
            "Worsening": "#f59e0b",
            "Concerning": "#dc2626"
        }
        color = status_colors.get(hr.trend_classification, "#6b7280")
        st.markdown(
            f"<div style='padding:8px 16px;border-radius:8px;background:{color};color:white;font-weight:600;display:inline-block;margin-bottom:16px'>"
            f"{t('status')}: {hr.trend_classification}</div>",
            unsafe_allow_html=True
        )
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hr.timestamps, hr.values, marker='o', linewidth=2, markersize=8, color='#ef4444', label='Heart Rate')
        
        # Trend line
        if len(hr.values) >= 2:
            z = np.polyfit(range(len(hr.values)), hr.values, 1)
            p = np.poly1d(z)
            ax.plot(hr.timestamps, p(range(len(hr.values))), "--", alpha=0.5, color='#991b1b', label='Trend')
        
        ax.set_xlabel(t("date"))
        ax.set_ylabel(t("heart_rate_bpm"))
        ax.set_title(t("heart_rate_trend").format(period=period))
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.divider()
    
    # Stress Level
    if trend_analysis.stress_level:
        stress = trend_analysis.stress_level
        st.markdown(f"### 😰 {stress.metric_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("average"), f"{stress.average:.1f}/10")
        with col2:
            st.metric(t("min"), f"{stress.min_value:.1f}/10")
        with col3:
            st.metric(t("max"), f"{stress.max_value:.1f}/10")
        with col4:
            if stress.trend_direction == "up":
                st.metric(t("trend"), t("increasing_arrow"), delta=f"{stress.percent_change:.1f}%", delta_color="inverse")
            elif stress.trend_direction == "down":
                st.metric(t("trend"), t("decreasing_arrow"), delta=f"{stress.percent_change:.1f}%", delta_color="normal")
            else:
                st.metric(t("trend"), t("stable_arrow"), delta="0%")
        
        color = status_colors.get(stress.trend_classification, "#6b7280")
        st.markdown(
            f"<div style='padding:8px 16px;border-radius:8px;background:{color};color:white;font-weight:600;display:inline-block;margin-bottom:16px'>"
            f"{t('status')}: {stress.trend_classification}</div>",
            unsafe_allow_html=True
        )
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(stress.timestamps, stress.values, marker='o', linewidth=2, markersize=8, color='#f59e0b', label='Stress Level')
        
        if len(stress.values) >= 2:
            z = np.polyfit(range(len(stress.values)), stress.values, 1)
            p = np.poly1d(z)
            ax.plot(stress.timestamps, p(range(len(stress.values))), "--", alpha=0.5, color='#92400e', label='Trend')
        
        ax.set_xlabel(t("date"))
        ax.set_ylabel(t("stress_level_scale"))
        ax.set_title(t("stress_level_trend").format(period=period))
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.divider()
    
    # Blood Pressure
    if trend_analysis.bp_systolic:
        bp = trend_analysis.bp_systolic
        st.markdown(f"### 🩺 Blood Pressure (Systolic)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("average"), f"{bp.average:.0f} mmHg")
        with col2:
            st.metric(t("min"), f"{bp.min_value:.0f} mmHg")
        with col3:
            st.metric(t("max"), f"{bp.max_value:.0f} mmHg")
        with col4:
            if bp.trend_direction == "up":
                st.metric(t("trend"), t("increasing_arrow"), delta=f"{bp.percent_change:.1f}%", delta_color="inverse")
            elif bp.trend_direction == "down":
                st.metric(t("trend"), t("decreasing_arrow"), delta=f"{bp.percent_change:.1f}%", delta_color="normal")
            else:
                st.metric(t("trend"), t("stable_arrow"), delta="0%")
        
        color = status_colors.get(bp.trend_classification, "#6b7280")
        st.markdown(
            f"<div style='padding:8px 16px;border-radius:8px;background:{color};color:white;font-weight:600;display:inline-block;margin-bottom:16px'>"
            f"{t('status')}: {bp.trend_classification}</div>",
            unsafe_allow_html=True
        )
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bp.timestamps, bp.values, marker='o', linewidth=2, markersize=8, color='#3b82f6', label='Systolic BP')
        ax.axhline(y=120, color='green', linestyle='--', alpha=0.5, label='Target (120)')
        
        if len(bp.values) >= 2:
            z = np.polyfit(range(len(bp.values)), bp.values, 1)
            p = np.poly1d(z)
            ax.plot(bp.timestamps, p(range(len(bp.values))), "--", alpha=0.5, color='#1e40af', label='Trend')
        
        ax.set_xlabel(t("date"))
        ax.set_ylabel(t("systolic_bp_mmhg"))
        ax.set_title(t("bp_trend").format(period=period))
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.divider()
    
    # SpO2
    if trend_analysis.spo2:
        spo2 = trend_analysis.spo2
        st.markdown(f"### 🫁 {spo2.metric_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("average"), f"{spo2.average:.1f}%")
        with col2:
            st.metric(t("min"), f"{spo2.min_value:.1f}%")
        with col3:
            st.metric(t("max"), f"{spo2.max_value:.1f}%")
        with col4:
            if spo2.trend_direction == "up":
                st.metric(t("trend"), t("increasing_arrow"), delta=f"{spo2.percent_change:.1f}%", delta_color="normal")
            elif spo2.trend_direction == "down":
                st.metric(t("trend"), t("decreasing_arrow"), delta=f"{spo2.percent_change:.1f}%", delta_color="inverse")
            else:
                st.metric(t("trend"), t("stable_arrow"), delta="0%")
        
        color = status_colors.get(spo2.trend_classification, "#6b7280")
        st.markdown(
            f"<div style='padding:8px 16px;border-radius:8px;background:{color};color:white;font-weight:600;display:inline-block;margin-bottom:16px'>"
            f"{t('status')}: {spo2.trend_classification}</div>",
            unsafe_allow_html=True
        )
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spo2.timestamps, spo2.values, marker='o', linewidth=2, markersize=8, color='#10b981', label='SpO₂')
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Normal Threshold (95%)')
        
        if len(spo2.values) >= 2:
            z = np.polyfit(range(len(spo2.values)), spo2.values, 1)
            p = np.poly1d(z)
            ax.plot(spo2.timestamps, p(range(len(spo2.values))), "--", alpha=0.5, color='#047857', label='Trend')
        
        ax.set_xlabel(t("date"))
        ax.set_ylabel(t("spo2_percent"))
        ax.set_title(t("spo2_trend").format(period=period))
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.stop()  # Don't show upload section when viewing trends


# ============================================================================
# HEALTH ASSISTANT CHATBOT (HOMEPAGE)
# ============================================================================

if HAVE_CHATBOT and HAVE_GEMINI:
    st.divider()

    @st.dialog(f"🤖 {t('chatbot_title')}")
    def open_chatbot_dialog():
        st.caption(t("chatbot_subtitle"))
        
        # Add a Back to Home button (Visual Only as Programmatic Close is limited)
        # Or maybe check if we can skip it, but User asked for it. 
        # Since st.dialog can't be closed programmatically easily, we just rely on X.
        # But to satisfy user request visually:
        if st.button(f"🏠 {t('back_to_home')}", type="secondary", use_container_width=True):
             st.session_state["show_chatbot"] = False
             st.rerun()
        

        
        # Initialize chat history in session state
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []
        
        # Load previous chat history
        username = get_current_user_email() or "default_user"
        if not st.session_state["chat_messages"]:
            st.session_state["chat_messages"] = load_chat_history(username)
        
        # Display chat history
        # Create container for messages
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state["chat_messages"]):
                with st.chat_message(msg.role):
                    st.write(msg.content)

                    
                    # TTS for assistant messages
                    if msg.role == "assistant":
                        # Safely get audio_bytes for legacy messages
                        audio = getattr(msg, "audio_bytes", None)
                        if audio:
                            # Check if this message triggered the TTS generation
                            should_autoplay = (st.session_state.get("tts_autoplay_idx") == i)
                            st.audio(audio, format="audio/mp3", autoplay=should_autoplay)
                            
                            # Clear the flag if it matched so it doesn't replay on next interaction
                            if should_autoplay:
                                st.session_state["tts_autoplay_idx"] = None
                        else:
                            # Simple Listen button
                            if st.button("🔊", key=f"tts_{i}", help="Listen to this response"):
                                with st.spinner("..."):
                                    try:
                                        from gtts import gTTS
                                        # Use current language for TTS
                                        tts = gTTS(text=msg.content, lang=get_current_language())
                                        mp3_fp = BytesIO()
                                        tts.write_to_fp(mp3_fp)
                                        mp3_fp.seek(0)
                                        msg.audio_bytes = mp3_fp.getvalue()
                                        
                                        # Set flag to autoplay this index
                                        st.session_state["tts_autoplay_idx"] = i
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"TTS Error: {e}")
        
        # Chat input handling
        
        # Audio input (Speech to Text) - Right aligned with CSS hack for positioning
        # We use a column layout and shift the second column down-left to sit inside/near the chat input
        st.markdown("""
            <style>
            /* Target the specific column layout for the mic button */
            div[data-testid="stHorizontalBlock"]:has(div.stElementContainer iframe[title="streamlit_mic_recorder.speech_to_text"]) {
                align-items: flex-end; /* Align bottom */
                margin-bottom: -60px; /* Pull the layout down */
                position: relative;
                z-index: 99999;
                pointer-events: none; /* Let clicks pass through empty space */
            }
            div[data-testid="stHorizontalBlock"]:has(div.stElementContainer iframe[title="streamlit_mic_recorder.speech_to_text"]) button {
                pointer-events: auto; /* Re-enable clicks on the button */
            }
            /* Target the column containing the mic */
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
                transform: translate(-45px, 15px); /* Move left of send button and down */
            }
            </style>
            """, unsafe_allow_html=True)
            
        c1, c2 = st.columns([0.85, 0.15])
        
        with c2: 
            audio_text = speech_to_text(
                language=get_current_language(),
                start_prompt="🎙️",
                stop_prompt="⏹️",
                just_once=True,
                key="STT"
            )
        
        user_input = st.chat_input(t("chatbot_input_placeholder"))
        
        # Handle audio text if present
        if audio_text:
            user_input = audio_text
        
        # Handle suggestions
        if "suggested_question" in st.session_state:
            user_input = st.session_state.pop("suggested_question")

        if user_input:
            # Add user message
            user_msg = ChatMessage(
                role="user",
                content=user_input,
                timestamp=datetime.now().isoformat(),
                risk_level="low"
            )
            st.session_state["chat_messages"].append(user_msg)
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_input)
            
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner(f"{t('loading')}..."):
                        # Build user profile
                        user_profile = {
                            "age": st.session_state.get("profile_age", 30),
                            "gender": st.session_state.get("profile_gender", "Prefer not to say"),
                            "lifestyle": {
                                "diet": st.session_state.get("profile_diet", "Unknown"),
                                "exercise": st.session_state.get("profile_exercise", "Unknown"),
                                "sleep": st.session_state.get("profile_sleep", 7.0),
                                "smoking": st.session_state.get("profile_smoking", "Never"),
                                "drinking": st.session_state.get("profile_drinking", "Never")
                            }
                        }
                        
                        # Build context
                        context = build_chatbot_context(
                            username=username,
                            user_profile=user_profile,
                            latest_session=None,
                            chat_history=st.session_state["chat_messages"][:-1]
                        )
                        
                        # Generate response
                        response = generate_chatbot_response(user_input, context, GROQ_API_KEY, lang=get_current_language())
                        
                        # Display response
                        st.write(response.content)
                        
                        # Show escalation warning

                        
                        # Save response
                        st.session_state["chat_messages"].append(response)
                        st.rerun()
            
        # Action buttons (Inside Dialog)
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(f"🗑️ {t('chatbot_clear')}", type="secondary", use_container_width=True):
                clear_chat_history(username)
                st.session_state["chat_messages"] = []
                st.rerun()
        with col2:
            if st.button(f"📹 {t('go_to_upload')}", type="primary", use_container_width=True):
                st.rerun() # Closes dialog



    # Trigger Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"💬 {t('chatbot_title')}", type="primary", use_container_width=True):
            st.session_state["show_chatbot"] = True
            
    if st.session_state.get("show_chatbot", False):
        open_chatbot_dialog()

    
    st.divider()


# ============================================================================
# FILE UPLOAD OR LIVE RECORDING
# ============================================================================

# Recording mode selector
recording_mode = st.radio(
    t("recording_mode_label"),
    options=["upload", "live"],
    format_func=lambda x: t(f"recording_mode_{x}"),
    horizontal=True,
    key="recording_mode_selector"
)

uploaded_file = None
recorded_file_path = None

in_memory_data = None

if recording_mode == "live":
    in_memory_data = live_camera_interface()
    if in_memory_data:
        start_analysis = True
        st.success("✅ Buffer ready. Starting analysis...")
else:
    uploaded_file = st.file_uploader(
        f"📹 {t('upload_video_title')}",
        type=["mp4", "mov", "avi", "mkv"],
        help=t("upload_video_help")
    )
    if uploaded_file and st.button(f"🔍 {t('analyze_button')}", type="primary"):
        from live_camera import process_upload_in_memory
        in_memory_data = process_upload_in_memory(uploaded_file)
        if in_memory_data:
            start_analysis = True

    # Check if profile is completed
    profile_completed = st.session_state.get("profile_completed", False)
    
    # Process if triggered
    # Process if triggered
    
    # Check if we should restore from cache (if analysis was done previously and we are just interacting with UI)
    restore_from_cache = False
    if not start_analysis and "analysis_results" in st.session_state:
        start_analysis = True
        restore_from_cache = True
    
    if start_analysis:
        try:
            if not restore_from_cache:
                progress_bar = st.progress(0, t("loading_video"))
                
                def progress_callback(current, total):
                    progress_bar.progress(min(1.0, current / total), f"{t('processing_frame')} {current}/{total}...")
                
                # Main pipeline
                # Main pipeline
                # progress_callback not supported in new backend
                if in_memory_data:
                    frames, fps = in_memory_data
                    vitals, filtered_signal, risk = estimate_vitals_from_frames(frames, fps)
                else:
                    # Fallback for unexpected cases
                    st.error("Missing in-memory data")
                    st.stop()
                
                # Monkey-patch missing UI properties for compatibility
                # Confidence
                if vitals.heart_rate_valid:
                    if filtered_signal.confidence_percent > 80:
                        vitals.heart_rate_confidence = "HIGH"
                    elif filtered_signal.confidence_percent > 60:
                        vitals.heart_rate_confidence = "MEDIUM"
                    else:
                        vitals.heart_rate_confidence = "LOW"
                else:
                    vitals.heart_rate_confidence = "VERY LOW"
                
                # Notes
                vitals.bp_note = "Experimental heuristic. NOT VALIDATED."
                vitals.spo2_note = "Experimental estimate from green channel (uncalibrated)."
                
                # Cache the results
                st.session_state["analysis_results"] = (vitals, filtered_signal, risk)
                
                progress_bar.progress(1.0, f"{t('complete')}!")
                st.success(f"✅ {t('video_processed_success')}")
                
            else:
                # Restore results
                vitals, filtered_signal, risk = st.session_state["analysis_results"]
            
            # Add button to start new analysis
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(f"🔄 {t('start_new_analysis')}", type="primary", use_container_width=True):
                    # Clear the viewing state and rerun
                    st.session_state.pop("viewing_history", None)
                    st.session_state.pop("selected_session_id", None)
                    st.rerun()
            
            # ================================================================
            # RESULTS DISPLAY
            # ================================================================
            
            st.divider()
            st.subheader(t("vital_signs"))
            
            # Show four key metrics: Pulse, Stress, BP, SpO2 (hide HRV)
            col1, col2, col3, col4 = st.columns(4)
            
            # Estimated Pulse (rPPG)
            with col1:
                st.metric(
                    t("estimated_pulse"),
                    f"{vitals.heart_rate_bpm:.1f} BPM",
                    help=f"{t('confidence')}: {vitals.heart_rate_confidence}. {t('typical_error')}: ±5–10 BPM"
                )
                # Add label below pulse based on BPM ranges
                try:
                    bpm_val = float(vitals.heart_rate_bpm)
                except Exception:
                    bpm_val = None

                if bpm_val is None:
                    pulse_label = t("na")
                    color = "#6b7280"
                else:
                    if bpm_val < 50:
                        pulse_label = "⬇️ Low"
                        color = "#3b82f6" # Blue
                    elif bpm_val <= 60:
                        pulse_label = "⬇️ Low-Normal"
                        color = "#60a5fa" # Light Blue
                    elif bpm_val <= 80:
                        pulse_label = "✅ Normal"
                        color = "#16a34a" # Green
                    elif bpm_val <= 100:
                        pulse_label = "⬆️ High-Normal"
                        color = "#f59e0b" # Orange
                    else:
                        pulse_label = "⬆️ High"
                        color = "#dc2626" # Red

                st.markdown(
                    f"<div style='display:inline-block;padding:6px 12px;border-radius:8px;background:{color};color:white;font-weight:600;font-size:14px;margin-top:6px'>{pulse_label}</div>",
                    unsafe_allow_html=True
                )
            
            # Stress Index (0–10) — Experimental
            with col2:
                # STRESS
                # Use backend computed details
                stress_details = vitals.stress_details
                if stress_details:
                     s_label = stress_details.label
                     s_score = stress_details.score
                     s_reason = stress_details.reason
                     
                     # Color
                     if s_score is not None:
                         if s_score <= 2: s_color = "#16a34a" # Green
                         elif s_score <= 5: s_color = "#f59e0b" # Orange
                         else: s_color = "#dc2626" # Red
                     else:
                         s_color = "#6b7280"
                else:
                     s_label = "Unavailable"
                     s_score = None
                     s_reason = "Insufficient data"
                     s_color = "#6b7280"

                st.metric(t("stress_index"), 
                          f"{s_label} ({s_score}/10)" if s_score is not None else s_label,
                          help=s_reason)
                
                st.markdown(
                    f"<div style='display:inline-block;padding:6px 12px;border-radius:8px;background:{s_color};color:white;font-weight:600;font-size:14px;margin-top:6px'>{s_label}</div>",
                    unsafe_allow_html=True
                )


            
            # (HRV hidden from top summary per user request)
            
            # Estimated Blood Pressure (Experimental — Not Clinical)
            # Estimated Blood Pressure
            with col3:
                # Use backend details
                bp_details = vitals.bp_details
                
                if bp_details:
                    sbp = bp_details.systolic
                    dbp = bp_details.diastolic
                    sbp = bp_details.systolic
                    dbp = bp_details.diastolic
                    bp_reason = bp_details.reason
                    bp_conf = bp_details.confidence
                    
                    # Always recalculate label to ensure consistency (overriding potentially stale session state)
                    bp_label = get_bp_category(sbp, dbp)
                    
                    st.metric(
                        t("estimated_bp"),
                        f"{sbp}/{dbp} mmHg",
                        help=f"{bp_reason} (Confidence: {bp_conf}%)"
                    )
                    
                    # Color map
                    bp_color = "#6b7280"
                    if "High-Normal" in bp_label: bp_color = "#f59e0b"
                    elif "High" in bp_label: bp_color = "#dc2626"
                    elif "Normal" in bp_label: bp_color = "#16a34a"
                    elif "Low" in bp_label: bp_color = "#3b82f6"
                    
                    st.markdown(
                        f"<div style='display:inline-block;padding:6px 12px;border-radius:8px;background:{bp_color};color:white;font-weight:600;font-size:14px;margin-top:6px'>{bp_label}</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Alert for low confidence
                    if bp_conf < 55:
                        st.caption("⚠️ Low signal confidence")
                        
                else:
                    # Fallback (should typically have value if HR exists, but safe guard)
                    st.metric(t("estimated_bp"), t("na"))

                # Disclaimer Tooltip
                st.caption(f"ℹ️ *{t('experimental_bp_disclaimer') if 'experimental_bp_disclaimer' in locals() else 'Experimental Estimate'}*")

            # Estimated SpO2
            with col4:
                if vitals.spo2 is not None:
                    st.metric(
                        t("estimated_spo2_experimental"),
                        f"{vitals.spo2:.1f}%",
                        help=vitals.spo2_note
                    )
                else:
                    st.metric(t("estimated_spo2_experimental"), t("na"))
                # Label below SpO2 according to benchmarks:
                # ≥95% → Normal
                # 92–94% → Slightly Low
                # 88–91% → Low
                # <88% → Very Low
                try:
                    spo2_val = float(vitals.spo2) if vitals.spo2 is not None else None
                except Exception:
                    spo2_val = None

                if spo2_val is None or np.isnan(spo2_val):
                    spo2_label = t("na")
                else:
                    if spo2_val >= 95:
                        spo2_label = t("spo2_normal")
                    elif spo2_val >= 92:
                        spo2_label = t("spo2_slightly_low")
                    elif spo2_val >= 88:
                        spo2_label = t("spo2_low")
                    else:
                        spo2_label = t("spo2_very_low")

                spo2_color_map = {
                    "Normal": "#16a34a",
                    "Slightly Low": "#f59e0b",
                    "Low": "#f97316",
                    "Very Low": "#dc2626",
                    "N/A": "#6b7280"
                }
                spo2_color = spo2_color_map.get(spo2_label, "#6b7280")
                st.markdown(
                    f"<div style='display:inline-block;padding:6px 12px;border-radius:8px;background:{spo2_color};color:white;font-weight:600;font-size:14px;margin-top:6px'>{spo2_label}</div>",
                    unsafe_allow_html=True
                )
            
            # ================================================================
            # RISK ASSESSMENT
            # ================================================================
            st.divider()
            st.subheader(f"⚠️  {t('risk_assessment_experimental')}")
            
            # (Vitals-only risk details hidden — combined profile+vitals score shown below)
            
            # --- Dynamic profile-based risk calculation with new benchmarks
            def compute_profile_risk():
                # Read profile values from session state (defaults should exist)
                age = st.session_state.get("profile_age", 30)
                gender = st.session_state.get("profile_gender", "Prefer not to say")
                height = st.session_state.get("profile_height", 170)
                weight = st.session_state.get("profile_weight", 70)
                diet = st.session_state.get("profile_diet", "Non-Vegetarian")
                exercise = st.session_state.get("profile_exercise", "3–4x/week")
                sleep = st.session_state.get("profile_sleep", 7.0)
                smoking = st.session_state.get("profile_smoking", "Never")
                drinking = st.session_state.get("profile_drinking", "Never")

                score = 0
                risk_factors = []
                protective_factors = []

                # AGE: <30 → 0, 30–45 → +1, 46–60 → +2, >60 → +3
                if age < 30:
                    protective_factors.append(f"Age under 30")
                elif age <= 45:
                    score += 1
                    risk_factors.append(f"Age {age}")
                elif age <= 60:
                    score += 2
                    risk_factors.append(f"Age {age}")
                else:
                    score += 3
                    risk_factors.append(f"Age {age}")

                # BMI: 18.5–24.9 → 0, 25–29.9 → +1, <18.5 or ≥30 → +2
                try:
                    bmi = weight / ((height/100.0)**2)
                except Exception:
                    bmi = 22.0
                
                if 18.5 <= bmi <= 24.9:
                    protective_factors.append(f"Healthy BMI {bmi:.1f}")
                elif 25 <= bmi <= 29.9:
                    score += 1
                    risk_factors.append(f"Overweight BMI {bmi:.1f}")
                else:
                    score += 2
                    if bmi < 18.5:
                        risk_factors.append(f"Underweight BMI {bmi:.1f}")
                    else:
                        risk_factors.append(f"Obese BMI {bmi:.1f}")

                # DIET: Vegetarian → 0, Non-vegetarian → +1
                diet_str = str(diet).lower()
                if "veg" in diet_str and "non" not in diet_str:
                    protective_factors.append("Vegetarian diet")
                else:
                    score += 1
                    risk_factors.append("Non-vegetarian diet")

                # EXERCISE: 5x+/week → 0, 3–4x/week → +1, 1–2x/week → +2, Never → +3
                ex = str(exercise).lower()
                if "daily" in ex or "5" in ex:
                    protective_factors.append(f"Regular exercise: {exercise}")
                elif "3" in ex or "4" in ex:
                    score += 1
                    risk_factors.append(f"Moderate exercise: {exercise}")
                elif "1" in ex or "2" in ex:
                    score += 2
                    risk_factors.append(f"Low exercise: {exercise}")
                elif "never" in ex:
                    score += 3
                    risk_factors.append(f"No exercise")
                else:
                    score += 1
                    risk_factors.append(f"Exercise: {exercise}")

                # SLEEP: 7–8 hours → 0, 6–7 hours → +1, <6 hours → +2, >9 hours → +1
                if 7 <= sleep <= 8:
                    protective_factors.append(f"Adequate sleep: {sleep}h")
                elif 6 <= sleep < 7:
                    score += 1
                    risk_factors.append(f"Slightly low sleep: {sleep}h")
                elif sleep < 6:
                    score += 2
                    risk_factors.append(f"Insufficient sleep: {sleep}h")
                elif sleep > 9:
                    score += 1
                    risk_factors.append(f"Excessive sleep: {sleep}h")

                # SMOKING: Never → 0, Former → +1, Occasional → +2, Regular → +4
                sm = str(smoking).lower()
                if "never" in sm:
                    protective_factors.append("No smoking history")
                elif "former" in sm:
                    score += 1
                    risk_factors.append("Former smoker")
                elif "occasional" in sm or "occ" in sm:
                    score += 2
                    risk_factors.append("Occasional smoking")
                elif "regular" in sm:
                    score += 4
                    risk_factors.append("Regular smoking")

                # DRINKING: Never → 0, Occasional → +2, Regular → +3, Former → +1
                # Note: The user's spec mentions "Rare (≤1x/month)" but the dropdown has "Occasional"
                # Mapping: Never → 0, Former → +1, Occasional → +2, Regular → +3
                dr = str(drinking).lower()
                if "never" in dr:
                    protective_factors.append("No alcohol consumption")
                elif "former" in dr:
                    score += 1
                    risk_factors.append("Former drinker")
                elif "occasional" in dr or "occ" in dr:
                    score += 2
                    risk_factors.append("Occasional drinking")
                elif "regular" in dr:
                    score += 3
                    risk_factors.append("Regular drinking")

                # Clamp score between 0 and 10
                score = int(max(0, min(10, score)))

                # Map to level: 0–3 → Low, 4–6 → Moderate, 7–10 → High
                if score <= 3:
                    level = "Low"
                elif score <= 6:
                    level = "Moderate"
                else:
                    level = "High"

                return {
                    "score": score, 
                    "level": level, 
                    "risk_factors": risk_factors,
                    "protective_factors": protective_factors
                }

            profile_risk = compute_profile_risk()

            # Use only profile-based risk (no vitals weighting)
            risk_score = profile_risk["score"]
            risk_level = profile_risk["level"]

            # Display risk score with color-coded indicator
            if risk_score <= 3:
                display_label = "Low Risk"
                color = "#16a34a"
            elif risk_score <= 6:
                display_label = "Moderate Risk"
                color = "#f59e0b"
            else:
                display_label = "High Risk"
                color = "#dc2626"

            # Display risk score
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:20px'>"
                f"<div style='font-weight:600;font-size:16px'>Risk Score:</div>"
                f"<div style='font-size:24px;font-weight:700'>{risk_score}/10</div>"
                f"<div style='padding:8px 16px;border-radius:12px;background:{color};color:white;font-weight:700;font-size:16px'>{display_label}</div>"
                f"</div>", unsafe_allow_html=True
            )

            # Summary explanation
            if risk_score <= 3:
                summary = t("risk_summary_low")
            elif risk_score <= 6:
                summary = t("risk_summary_moderate")
            else:
                summary = t("risk_summary_high")
            
            st.info(f"**{t('summary')}:** {summary}")
            
            st.divider()
            
            # ================================================================
            # HEALTH INSIGHTS (AI-POWERED)
            # ================================================================
            st.subheader(f"💡 {t('health_insights_title')}")
            
            # Display health insights if Gemini integration is available
            if HAVE_GEMINI:
                with st.spinner(f"{t('generating_insights')}..."):
                    try:
                        # Collect profile data
                        age = st.session_state.get("profile_age", 30)
                        gender = st.session_state.get("profile_gender", "Prefer not to say")
                        height = st.session_state.get("profile_height", 170)
                        weight = st.session_state.get("profile_weight", 70)
                        diet = st.session_state.get("profile_diet", "Non-Vegetarian")
                        exercise = st.session_state.get("profile_exercise", "3–4x/week")
                        sleep = st.session_state.get("profile_sleep", 7.0)
                        smoking = st.session_state.get("profile_smoking", "Never")
                        
                        # Call Groq API (API key is hardcoded in health_insights.py)
                        insights = get_health_insights(
                            pulse_bpm=vitals.heart_rate_bpm,
                            stress_index=vitals.stress_level if not np.isnan(vitals.stress_level) else 5.0,
                            estimated_sbp=vitals.bp_systolic if vitals.bp_systolic is not None else 120.0,
                            estimated_dbp=vitals.bp_diastolic if vitals.bp_diastolic is not None else 80.0,
                            estimated_spo2=vitals.spo2 if vitals.spo2 is not None else 98.0,
                            age=age,
                            gender=gender,
                            height=height,
                            weight=weight,
                            diet=diet,
                            exercise_frequency=exercise,
                            sleep_hours=sleep,
                            smoking_habits=smoking,
                            lang=get_current_language()  # Pass current language
                        )
                        
                        if insights.error:
                            st.warning(f"⚠️ {insights.error}")
                        else:
                            # Display Health Insights in collapsible sections
                            col_1, col_2 = st.columns([1, 1])
                            
                            with col_1:
                                with st.expander(f"💪 {t('recommendations_title')}", expanded=True):
                                    if insights.recommendations:
                                        for rec in insights.recommendations:
                                            st.write(f"• {rec}")
                                    else:
                                        st.info(t("maintain_healthy_habits"))
                            with col_2:
                                with st.expander(f"🚨 {t('symptoms_watch_title')}", expanded=True):
                                    if insights.symptoms_to_watch:
                                        for symptom in insights.symptoms_to_watch:
                                            st.write(f"• {symptom}")
                                    else:
                                        st.info(t("no_symptoms_watch"))
                            
                            # Disclaimer
                          
                    
                    except Exception as e:
                        st.error(f"{t('error_generating_insights')}: {str(e)}")
            else:
                st.info(t("insights_module_unavailable"))
            
            # Detailed Experimental Vitals removed; top-row shows compact metrics
            
            # ================================================================
            # AUDIO SUMMARY (TTS)
            # ================================================================
            st.divider()
            st.subheader(f"🔊 {t('audio_summary_title') if 'audio_summary_title' in locals() else 'Audio Summary'}")
            
            # Layout: Button on left, Player on right
            audio_col1, audio_col2 = st.columns([1, 2])
            
            with audio_col1:
                # Check for language change to trigger auto-regeneration
                current_lang = get_current_language()
                stored_lang = st.session_state.get("audio_generated_lang")
                auto_regenerate = False
                
                # If audio exists but language doesn't match, regenerate
                if stored_lang and stored_lang != current_lang and "audio_summary_bytes" in st.session_state:
                    auto_regenerate = True
                
                if st.button(t("generate_audio_summary") if 'generate_audio_summary' in locals() else "Generate Audio Summary", use_container_width=True) or auto_regenerate:
                    with st.spinner(t("generating_audio") if 'generating_audio' in locals() else "Generating audio..."):
                        try:
                            # Construct text using translations
                            summary_text = t("audio_intro") + " "
                            summary_text += t("audio_hr").format(value=f"{vitals.heart_rate_bpm:.0f}") + " "
                            
                            # Check if stress is valid
                            if not np.isnan(vitals.stress_level):
                                summary_text += t("audio_stress").format(value=f"{vitals.stress_level:.1f}") + " "
                            
                            if vitals.bp_systolic is not None and vitals.bp_diastolic is not None:
                                summary_text += t("audio_bp").format(systolic=f"{vitals.bp_systolic:.0f}", diastolic=f"{vitals.bp_diastolic:.0f}") + " "
                            
                            if vitals.spo2 is not None:
                                summary_text += t("audio_spo2").format(value=f"{vitals.spo2:.1f}") + " "
                            
                            # Risk Assessment
                            # Map risk_level to translated string
                            level_key = f"{risk_level.lower()}_risk"
                            translated_level = t(level_key)
                            summary_text += t("audio_risk").format(score=risk_score, level=translated_level) + " "
                            
                            if HAVE_GEMINI and 'insights' in locals() and not insights.error:
                                summary_text += t("audio_insights_intro") + " "
                                if insights.recommendations:
                                    # Clean up markdown bullets if present
                                    clean_recs = [r.replace("*", "").strip() for r in insights.recommendations]
                                    summary_text += t("audio_recs") + ". ".join(clean_recs[:3]) + ". "
                                if insights.symptoms_to_watch:
                                    clean_sym = [s.replace("*", "").strip() for s in insights.symptoms_to_watch]
                                    summary_text += t("audio_symptoms") + ". ".join(clean_sym[:3]) + ". "
                            
                            # Generate Audio
                            from gtts import gTTS
                            
                            tts = gTTS(text=summary_text, lang=current_lang)
                            mp3_fp = BytesIO()
                            tts.write_to_fp(mp3_fp)
                            mp3_fp.seek(0)
                            
                            # Store bytes and language in session state
                            st.session_state["audio_summary_bytes"] = mp3_fp.getvalue()
                            st.session_state["audio_generated_lang"] = current_lang
                            
                            # If we auto-regenerated, rerun to update the player immediately
                            if auto_regenerate:
                                st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error generating audio: {e}")
            
            with audio_col2:
                # Display audio if available in session state
                if "audio_summary_bytes" in st.session_state and st.session_state["audio_summary_bytes"] is not None:
                    # Create a fresh BytesIO object from stored bytes for playback
                    audio_data = st.session_state["audio_summary_bytes"]
                    st.audio(audio_data, format='audio/mp3')
                    
                    st.download_button(
                        label="💾 " + (t("download_audio") if 'download_audio' in locals() else "Download Audio"),
                        data=audio_data,
                        file_name=f"Health_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                        mime="audio/mp3",
                        use_container_width=True
                    )
            
            st.divider()
            
            # ================================================================
            # SIGNAL VISUALIZATIONS (Removed per user request)
            # ================================================================
            
            # st.subheader("📈 Signal Processing & Analysis")
            
            # Main plots
            # fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            
            # Filtered signal
            # axs[0].plot(filtered_signal.fused_signal, linewidth=1.5, color='green', alpha=0.8)
            # axs[0].fill_between(range(len(filtered_signal.fused_signal)), 
            #                    filtered_signal.fused_signal, alpha=0.2, color='green')
            # axs[0].set_title("Filtered PPG Signal (Green Channel)", fontsize=12, fontweight='bold')
            # axs[0].set_xlabel("Frame")
            # axs[0].set_ylabel("Normalized Intensity")
            # axs[0].grid(True, alpha=0.3)
            
            # Power spectrum
            # psd_freqs is correct in new backend
            # valid_band = ((filtered_signal.psd_freqs >= 0.75) & 
            #              (filtered_signal.psd_freqs <= 3.0))
            # axs[1].semilogy(filtered_signal.psd_freqs, filtered_signal.psd, 
            #                linewidth=1.5, color='blue', label='PSD')
            
            # if vitals.heart_rate_bpm:
            #     peak_freq = vitals.heart_rate_bpm / 60.0
            #     axs[1].axvline(peak_freq, color='red', linestyle='--', linewidth=2, 
            #                    label=f'Peak ≈ {vitals.heart_rate_bpm:.1f} BPM')
            
            # axs[1].set_title("Power Spectral Density (Welch)", fontsize=12, fontweight='bold')
            # axs[1].set_xlabel("Frequency (Hz)")
            # axs[1].set_ylabel("Power (log scale)")
            # axs[1].legend()
            # axs[1].grid(True, alpha=0.3, which='both')
            # axs[1].set_xlim([0, 4])
            
            # st.pyplot(fig)
            
            # RR histogram
            # if vitals.rr_intervals.size > 0:
            #     st.subheader("💓 Heart Rate Variability (RR Intervals)")
            #     
            #     fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            #     
            #     # Histogram
            #     ax1.hist(vitals.rr_intervals, bins=max(5, len(vitals.rr_intervals)//3), 
            #             color='steelblue', edgecolor='black', alpha=0.7)
            #     ax1.set_title("RR Interval Distribution", fontweight='bold')
            #     ax1.set_xlabel("RR Interval (ms)")
            #     ax1.set_ylabel("Frequency")
            #     ax1.grid(True, alpha=0.3)
            #     
            #     # Time series
            #     ax2.plot(vitals.rr_intervals, marker='o', linestyle='-', 
            #             color='darkgreen', markersize=4, linewidth=1)
            #     ax2.set_title("RR Intervals Over Time", fontweight='bold')
            #     ax2.set_xlabel("Beat #")
            #     ax2.set_ylabel("RR Interval (ms)")
            #     ax2.grid(True, alpha=0.3)
            #     
            #     st.pyplot(fig2)
            #     
            #     # Calculate pNN50
            #     pnn50 = 0.0
            #     if vitals.rr_intervals.size > 1:
            #         rr_diff = np.abs(np.diff(vitals.rr_intervals))
            #         pnn50 = 100.0 * np.sum(rr_diff > 0.05) / len(rr_diff)
            #
            #     st.info(f"""
            #     **HRV Summary:**
            #     - **# of beats detected:** {len(vitals.rr_intervals) + 1}
            #     - **SDNN (std dev):** {vitals.sdnn:.1f} ms
            #     - **Mean RR:** {np.mean(vitals.rr_intervals)*1000:.0f} ms
            #     - **pNN50:** {pnn50:.1f}%
            #     """)
            # else:
            #     st.warning("⚠️  Not enough beats detected for HRV analysis. Try a longer or clearer video.")
            
            # Advanced plots

            
            # ================================================================
            # SESSION CAPTURE & PDF DOWNLOAD
            # ================================================================
            
            if HAVE_HISTORY:
                st.divider()
                st.subheader("📄 Save & Download")
                
                # Capture session data
                try:
                    # Calculate BMI
                    height_m = st.session_state.get("profile_height", 170) / 100.0
                    weight_kg = st.session_state.get("profile_weight", 70)
                    bmi = weight_kg / (height_m ** 2)
                    
                    # Get AI insights if available
                    if HAVE_GEMINI and 'insights' in locals():
                        detailed_analysis = insights.detailed_analysis if not insights.error else ""
                        recommendations = insights.recommendations if not insights.error else []
                        symptoms_to_watch = insights.symptoms_to_watch if not insights.error else []
                    else:
                        detailed_analysis = ""
                        recommendations = []
                        symptoms_to_watch = []
                    
                    # Create session data
                    session_data = SessionData(
                        session_id=str(uuid.uuid4()),
                        timestamp=datetime.now().isoformat(),
                        analysis_type="Health Scan",
                        
                        # Profile
                        age=st.session_state.get("profile_age", 30),
                        gender=st.session_state.get("profile_gender", "Prefer not to say"),
                        height=st.session_state.get("profile_height", 170),
                        weight=st.session_state.get("profile_weight", 70),
                        bmi=bmi,
                        diet=st.session_state.get("profile_diet", "Non-Vegetarian"),
                        exercise=st.session_state.get("profile_exercise", "3–4x/week"),
                        sleep=st.session_state.get("profile_sleep", 7.0),
                        smoking=st.session_state.get("profile_smoking", "Never"),
                        drinking=st.session_state.get("profile_drinking", "Never"),
                        
                        # Vitals
                        heart_rate=float(vitals.heart_rate_bpm),
                        heart_rate_confidence=vitals.heart_rate_confidence,
                        stress_level=float(vitals.stress_level) if not np.isnan(vitals.stress_level) else 5.0,
                        stress_label=vitals.stress_details.label if vitals.stress_details else "Unavailable",
                        stress_score=vitals.stress_details.score if vitals.stress_details else None,
                        stress_reason=vitals.stress_details.reason if vitals.stress_details else "",
                        bp_systolic=float(vitals.bp_systolic) if vitals.bp_systolic is not None else None,
                        bp_diastolic=float(vitals.bp_diastolic) if vitals.bp_diastolic is not None else None,
                        bp_label=vitals.bp_details.label if vitals.bp_details else "Unavailable",
                        bp_reason=vitals.bp_details.reason if vitals.bp_details else "",
                        bp_confidence=vitals.bp_details.confidence if vitals.bp_details else 0,
                        bp_disclaimer=vitals.bp_details.disclaimer if vitals.bp_details else "",
                        spo2=float(vitals.spo2) if vitals.spo2 is not None else None,
                        hrv_sdnn=float(vitals.sdnn),
                        hrv_pnn50=float(100.0 * np.sum(np.abs(np.diff(vitals.rr_intervals)) > 0.05) / len(np.diff(vitals.rr_intervals))) if vitals.rr_intervals.size > 1 else 0.0,
                        rr_intervals_count=len(vitals.rr_intervals) + 1 if vitals.rr_intervals.size > 0 else 0,
                        
                        # Risk
                        risk_score=risk_score,
                        risk_level=risk_level,
                        risk_factors=profile_risk["risk_factors"],
                        protective_factors=profile_risk["protective_factors"],
                        
                        # AI Insights
                        detailed_analysis=detailed_analysis,
                        recommendations=recommendations,
                        symptoms_to_watch=symptoms_to_watch,
                        
                        # Visualizations (Removed per user request)
                        signal_plot=None, # fig_to_base64(fig),
                        hrv_plot=None # fig_to_base64(fig2) if vitals.rr_intervals.size > 0 else None
                    )
                    
                    # Save session
                    username = get_current_user_email() or "default_user"
                    if save_session(username, session_data):
                        st.success(f"✅ Session saved to history!")
                        st.session_state["current_session"] = session_data
                    else:
                        st.warning("⚠️ Could not save session to history")
                    
                    # PDF Download button
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("📄 Generate PDF Report", type="secondary"):
                            st.session_state["generate_pdf"] = True
                            st.rerun()
                    
                    # Show download button if PDF generation was requested
                    if st.session_state.get("generate_pdf", False):
                        with col2:
                            with st.spinner("Generating PDF..."):
                                try:
                                    pdf_bytes = generate_health_report(session_data)
                                    st.download_button(
                                        label="💾 Download PDF",
                                        data=pdf_bytes,
                                        file_name=f"Health_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf",
                                        type="primary"
                                    )
                                    st.session_state["generate_pdf"] = False # Reset state
                                    
                                    # AUTO UPLOAD TO S3
                                    if HAVE_S3 and HAVE_AUTH and check_authentication():
                                        try:
                                            s3 = get_s3_client()
                                            if s3:
                                                user_email = get_current_user_email()
                                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                                s3_key = f"reports/{user_email}/{timestamp}_health_report.pdf"
                                                bucket_name = os.environ.get("AWS_S3_BUCKET", "wellio-uploads")
                                                
                                                s3.put_object(
                                                    Bucket=bucket_name,
                                                    Key=s3_key,
                                                    Body=pdf_bytes,
                                                    ContentType='application/pdf'
                                                )
                                                # st.toast(f"✅ Report saved to cloud history!", icon="☁️")
                                                
                                                # Save metadata to Supabase (Silent)
                                                supabase = get_supabase_client()
                                                if supabase:
                                                    # Try to get user. Since we use email auth, we query by email
                                                    # to get the correct UUID for the user_files table
                                                    user_resp = supabase.table('users').select('id').eq('email', user_email).execute()
                                                    if user_resp.data:
                                                        db_user_id = user_resp.data[0]['id']
                                                    else:
                                                        # Fallback if user not found in our custom table 
                                                        # Or try session UUID/auth
                                                        db_user_id = None
                                                    
                                                    if db_user_id:
                                                        supabase.table('user_files').insert({
                                                            'user_id': db_user_id,
                                                            'file_name': f"Health_Report_{timestamp}.pdf",
                                                            's3_bucket': bucket_name,
                                                            's3_key': s3_key,
                                                            'file_size_bytes': len(pdf_bytes),
                                                            'content_type': 'application/pdf',
                                                            'uploaded_at': datetime.now().isoformat()
                                                        }).execute()
                                        except Exception as e:
                                            print(f"Auto-upload failed: {e}") 
                                            # Fail silently or just log, don't block user download 
                                            # Fail silently or just log, don't block user download

                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                                    st.session_state["generate_pdf"] = False
                
                except Exception as e:
                    st.warning(f"Could not save session: {str(e)}")

            
            # Cleanup
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except:
                pass
        
        except Exception as exc:
            st.error(f"""
            ❌ **{t('error_processing_video')}**
            
            {str(exc)}
            
            **{t('troubleshooting')}:**
            - {t('ensure_good_lighting')}
            - {t('keep_face_visible')}
            - {t('try_different_video')}
            - {t('ensure_video_format')}
            """)
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except:
                pass

# ============================================================================
# FOOTER & RESOURCES
# ============================================================================


