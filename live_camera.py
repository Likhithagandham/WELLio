"""
Native Cloud-Safe Camera Module for Wellio
=========================================

Uses Streamlit-native components for video capture/upload to ensure 
compatibility with Streamlit Cloud while maintaining an interactive feel.
"""

import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
import time

def live_camera_interface():
    """
    Renders a native Streamlit 'Capture Wizard' for video input.
    Returns path to video file if upload is complete, else None.
    """
    
    st.markdown("### 📸 Vital Signs Capture Wizard")
    
    # Step-by-step guidance
    step = st.radio(
        "Capture Steps",
        ["1. Prepare", "2. Capture & Upload", "3. Analyze"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if "Prepare" in step:
        st.info("💡 **Best Results**: Sit in a well-lit area, keep your face steady, and ensure your forehead/cheeks are clearly visible.")
        st.image("https://img.icons8.com/fluency/96/person-female--v1.png", width=60) # Placeholder or local asset if available
        st.write("When ready, move to the next step to record or upload a short clip.")
        
    elif "Capture & Upload" in step:
        st.write("#### 📹 Record or Select Video")
        st.caption("Please provide a **10-15 second** close-up video of your face.")
        
        # Native File Uploader
        # Note: In mobile browsers, st.file_uploader(capture="user") can trigger native camera app
        uploaded_file = st.file_uploader(
            "Upload face video", 
            type=["mp4", "mov", "avi", "webm"],
            help="Most smartphones allow you to record directly when clicking 'Browse files'."
        )
        
        if uploaded_file is not None:
            st.success("✅ Video received!")
            
            # Save to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            st.session_state["raw_video_path"] = tfile.name
            st.info("Move to the **Analyze** step to start the vitals estimation.")
            
    elif "Analyze" in step:
        if "raw_video_path" in st.session_state and st.session_state["raw_video_path"]:
            st.write("#### ⚙️ Final Review")
            st.video(st.session_state["raw_video_path"])
            
            if st.button("🚀 Start Interactive Analysis", type="primary"):
                # Simulate "Live" extraction feedback
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(101):
                    time.sleep(0.02) # Fast simulation
                    progress_bar.progress(i)
                    if i < 30: status_text.text("🔍 Detecting face region...")
                    elif i < 70: status_text.text("💓 Extracting signal from forehead...")
                    else: status_text.text("📊 Processing vitals...")
                
                return st.session_state["raw_video_path"]
        else:
            st.warning("Please go back and upload a video first.")
            
    return None

def process_recorded_video(video_path: str):
    """
    Pass-through for the existing backend pipeline.
    """
    return video_path
