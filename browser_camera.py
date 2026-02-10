"""
Browser-Based Live Camera Recording Module for Wellio
Uses HTML5 MediaRecorder API - Works on Streamlit Cloud!

This module provides a live camera interface using browser APIs,
no streamlit-webrtc dependency needed.
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
import tempfile
import os
from pathlib import Path

# Declare component using the local static files
# Use absolute path for reliability
parent_dir = os.path.dirname(os.path.abspath(__file__))
component_path = os.path.join(parent_dir, "browser_camera_component")

# Declare the component
_component_func = components.declare_component(
    "live_camera_component",
    path=component_path,
)

def live_camera_component(key="live_camera"):
    """
    Streamlit component for live camera recording.
    Returns base64-encoded video data when recording is complete.
    """
    # Simply call the declared component function
    # The return value will be the data sent from JavaScript via setComponentValue
    video_data = _component_func(key=key, default=None)
    return video_data

def save_recorded_video(base64_data: str) -> str:
    """
    Save base64-encoded video data to a temporary file.
    Returns path to the saved video file.
    """
    # Check if we have valid data
    if not base64_data or not isinstance(base64_data, str):
        return None
    
    # Skip if it's an empty string
    if len(base64_data) < 100:  # Base64 video should be much larger
        return None
    
    try:
        # Decode base64 to bytes
        video_bytes = base64.b64decode(base64_data)
        
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "recorded_video.webm")
        
        # Write to file
        with open(video_path, 'wb') as f:
            f.write(video_bytes)
        
        return video_path
    except Exception as e:
        st.error(f"Error saving video: {e}")
        return None
