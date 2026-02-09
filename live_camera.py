"""
Live Camera Recording Module for Wellio rPPG Application
(Streamlit WebRTC Implementation)

This module provides a live camera interface with face detection guidance,
automatic recording, and integration with the existing video analysis pipeline.
"""

import av
import cv2
import numpy as np
import time
import threading
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from typing import Optional, Tuple, List
import tempfile
import os

# Import translations
from translations import get_text as t

# Helper class for thread-safe value storage
class SharedState:
    def __init__(self):
        self.face_detected = False
        self.is_aligned = False
        self.guidance_text = ""
        self.guidance_color = (255, 255, 255) # RGB
        self.recording_active = False
        self.recording_start_time = 0
        self.recorded_frames = []
        self.process_complete = False
        self.countdown_active = False
        self.countdown_start = 0
        self.alignment_stable_start = 0

class FaceGuidanceProcessor(VideoProcessorBase):
    def __init__(self, language="en"):
        self.language = language
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.shared_state = SharedState()
        self.lock = threading.Lock()
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        
        # Mirror image for better UX
        img = cv2.flip(img, 1)
        
        # Calculate oval parameters
        center_x, center_y = width // 2, height // 2
        radius_x = int(width * 0.25)
        radius_y = int(height * 0.35)
        
        # Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Status update
        current_guidance = ""
        current_color = (0, 165, 255) # Orange (BGR)
        aligned = False
        
        with self.lock:
            if len(faces) == 0:
                current_guidance = "Face not detected" # Will translate in UI
                self.shared_state.alignment_stable_start = 0
            else:
                # Get largest face
                largest_face = max(faces, key=lambda r: r[2] * r[3])
                x, y, w, h = largest_face
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Check alignment
                offset_x = abs(face_center_x - center_x)
                offset_y = abs(face_center_y - center_y)
                
                # Heuristics
                if offset_x > radius_x * 0.5 or offset_y > radius_y * 0.5:
                    current_guidance = "Center your face"
                    self.shared_state.alignment_stable_start = 0
                elif w < radius_x * 1.0:
                    current_guidance = "Move closer"
                    self.shared_state.alignment_stable_start = 0
                elif w > radius_x * 1.8:
                    current_guidance = "Move back"
                    self.shared_state.alignment_stable_start = 0
                else:
                    current_guidance = "Perfect! Stay still"
                    current_color = (0, 255, 0) # Green
                    aligned = True
            
            # Logic for Auto-Recording
            now = time.time()
            
            if not self.shared_state.recording_active and not self.shared_state.process_complete:
                if aligned:
                    if self.shared_state.alignment_stable_start == 0:
                        self.shared_state.alignment_stable_start = now
                    elif now - self.shared_state.alignment_stable_start > 2.0: # 2 sec stability
                        if not self.shared_state.countdown_active:
                            self.shared_state.countdown_active = True
                            self.shared_state.countdown_start = now
                else:
                    self.shared_state.alignment_stable_start = 0
                    self.shared_state.countdown_active = False
                
                # Handle Countdown
                if self.shared_state.countdown_active:
                    elapsed = now - self.shared_state.countdown_start
                    if elapsed < 3.0:
                        count = 3 - int(elapsed)
                        current_guidance = f"Starting in {count}..."
                    else:
                        self.shared_state.recording_active = True
                        self.shared_state.recording_start_time = now
                        self.shared_state.countdown_active = False
            
            elif self.shared_state.recording_active:
                # Record frame (keep RGB for simplicity in saving later, but cv2 uses BGR)
                # We will store BGR frames
                self.shared_state.recorded_frames.append(img.copy())
                
                elapsed = now - self.shared_state.recording_start_time
                remaining = 15 - int(elapsed)
                current_guidance = f"Recording... {remaining}s"
                current_color = (0, 0, 255) # Red
                
                if elapsed >= 15.0:
                    self.shared_state.recording_active = False
                    self.shared_state.process_complete = True
                    current_guidance = "Recording Complete!"
            
            elif self.shared_state.process_complete:
                current_guidance = "Processing..."
                
            # Draw overlay
            # Oval
            color = current_color
            cv2.ellipse(img, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, color, 4)
            
            # Text (Draw simplistic text on frame, but mainly use UI)
            # Actually, drawing on frame is better for sync
            cv2.putText(img, current_guidance, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Update state for UI to read (if needed)
            self.shared_state.guidance_text = current_guidance

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def get_camera_component_html(language="en"):
    # This function is deprecated with Webrtc approach but keeping interface signature if needed
    pass

def live_camera_interface():
    """
    Renders the Webrtc streamer and handles the recording logic.
    Returns path to video file if recording is complete, else None.
    """
    
    ctx = webrtc_streamer(
        key="live_camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=FaceGuidanceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.video_processor:
        # Access the processor logic to check status
        processor = ctx.video_processor
        
        # Wait for the recording to accept/complete or the stream to stop
        # This keeps the script running so we can detect the completion event automatically
        placeholder = st.empty()
        
        while ctx.state.playing:
            if processor.shared_state.process_complete:
                placeholder.success("Recording complete! Processing...")
                
                # Save the frames to a file
                frames = processor.shared_state.recorded_frames
                if frames:
                    tmp_dir = tempfile.mkdtemp()
                    tmp_path = os.path.join(tmp_dir, "live_recording.mp4")
                    
                    # Get dimensions from first frame
                    height, width, layers = frames[0].shape
                    
                    # MP4V codec is widely supported
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(tmp_path, fourcc, 20.0, (width, height))
                    
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    
                    return tmp_path
                else:
                    st.error("No frames captured.")
                    return None
            
            # Update status message if recording (optional UX improvement)
            if processor.shared_state.recording_active:
                elapsed = time.time() - processor.shared_state.recording_start_time
                remaining = max(0, 15 - int(elapsed))
                placeholder.info(f"Recording... {remaining}s")
            elif processor.shared_state.countdown_active:
                elapsed = time.time() - processor.shared_state.countdown_start
                count = max(1, 3 - int(elapsed))
                placeholder.warning(f"Starting in {count}...")
            elif processor.shared_state.alignment_stable_start > 0:
                 placeholder.info("Face aligned. Hold still...")
            else:
                 # Clean state or instructions
                 time.sleep(0.1)
            
            time.sleep(0.1)
                
    return None

def process_recorded_video(video_path: str):
    # Pass-through since we generate file directly
    return video_path
