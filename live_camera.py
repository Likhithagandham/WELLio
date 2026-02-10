"""
In-Memory Camera Module for Wellio
==================================

Purely native Streamlit implementation.
All data remains in RAM. No file I/O or temp storage.
Uses PyAV to decode video streams in-memory.
"""

import streamlit as st
import numpy as np
import av
import io
import time
from typing import List, Optional, Tuple

def live_camera_interface():
    """
    Renders an in-memory capture interface.
    Returns List[np.ndarray] and FPS if successful, else None.
    """
    
    # Guidance
    with st.expander("📝 Instructions for In-Memory Capture"):
        st.write("""
        1. Click the button below to start your camera.
        2. Record a **10-15 second** video of your face.
        3. The data will be processed strictly in your device's RAM.
        """)
    
    # Custom Live Recorder Component (Cloud Safe, MediaRecorder API)
    from video_recorder import video_recorder, process_recorder_output
    
    recorder_output = video_recorder(key="live_video_recorder")
    
    if recorder_output:
        st.info("✅ Clip captured! Processing frames in memory...")
        return process_recorder_output(recorder_output)
            
    return None

def process_upload_in_memory(uploaded_file) -> Optional[Tuple[List[np.ndarray], float]]:
    """
    Decodes the uploaded BytesIO stream directly into a list of NumPy frames.
    """
    status = st.empty()
    progress = st.progress(0)
    
    try:
        status.info("📽️ Reading stream into memory...")
        # PyAV can open file-like objects
        container = av.open(uploaded_file)
        
        frames = []
        fps = 0.0
        
        # Get stream and FPS
        v_stream = container.streams.video[0]
        # Some streams don't report average_rate, fallback to r_frame_rate
        fps = float(v_stream.average_rate or v_stream.r_frame_rate)
        if fps < 1: fps = 30.0 # Fallback
        
        # Extract frames
        stream_total_frames = v_stream.frames or 300 # Assume ~10s at 30fps if unknown
        
        for i, frame in enumerate(container.decode(video=0)):
            # Convert to NumPy BGR (Core logic expects BGR)
            # frame.to_ndarray returns RGB or Gray, we want BGR for cv2-like logic
            img = frame.to_ndarray(format='bgr24')
            frames.append(img)
            
            # Update progress periodically
            if i % 10 == 0:
                p_val = min(1.0, i / stream_total_frames)
                progress.progress(p_val)
                status.info(f"🎞️ Extracting frames... {i} collected")
        
        container.close()
        
        if len(frames) < 60: # Less than ~2 seconds
            st.error("Captured video too short. Please record for at least 10 seconds.")
            return None
            
        status.success(f"✅ {len(frames)} frames buffered in RAM. Starting analysis...")
        st.session_state["last_processed_id"] = uploaded_file.id
        
        # Final analysis step with interactive status
        time.sleep(0.5)
        return frames, fps

    except Exception as e:
        st.error(f"In-memory processing failed: {e}")
        return None

def process_recorded_video(data: Tuple[List[np.ndarray], float]):
    """
    Helper to bridge the interface with the backend.
    """
    return data
