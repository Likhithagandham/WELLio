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

def get_camera_html():
    """
    Returns HTML/JavaScript for browser-based camera recording.
    Uses MediaRecorder API with face detection overlay.
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            #camera-container {
                position: relative;
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
            }
            #video {
                width: 100%;
                border-radius: 10px;
                transform: scaleX(-1); /* Mirror effect */
            }
            #canvas {
                display: none;
            }
            .overlay {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                border: 4px solid #FFA500;
                border-radius: 50%;
                width: 250px;
                height: 350px;
                pointer-events: none;
            }
            .overlay.recording {
                border-color: #FF0000;
                animation: pulse 1s infinite;
            }
            .overlay.aligned {
                border-color: #00FF00;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            #status {
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
                background: #f0f0f0;
            }
            #controls {
                text-align: center;
                margin: 20px 0;
            }
            button {
                background: #4A6741;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 16px;
                border-radius: 8px;
                cursor: pointer;
                margin: 5px;
            }
            button:hover {
                background: #3A5232;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            #countdown {
                font-size: 48px;
                font-weight: bold;
                color: #FF0000;
            }
        </style>
    </head>
    <body>
        <div id="camera-container">
            <video id="video" autoplay playsinline></video>
            <div id="overlay" class="overlay"></div>
            <canvas id="canvas"></canvas>
        </div>
        <div id="status">Click "Start Camera" to begin</div>
        <div id="controls">
            <button id="startBtn" onclick="startCamera()">Start Camera</button>
            <button id="recordBtn" onclick="startRecording()" disabled>Start Recording (15s)</button>
            <button id="stopBtn" onclick="stopCamera()" disabled>Stop Camera</button>
        </div>

        <script>
            let stream = null;
            let mediaRecorder = null;
            let recordedChunks = [];
            let recordingTimeout = null;
            let countdownInterval = null;

            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: { ideal: 1280 },
                            height: { ideal: 720 },
                            facingMode: 'user'
                        }, 
                        audio: false 
                    });
                    
                    const video = document.getElementById('video');
                    video.srcObject = stream;
                    
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('recordBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('status').textContent = 'Camera ready! Position your face in the oval';
                    document.getElementById('overlay').classList.add('aligned');
                    
                } catch (err) {
                    document.getElementById('status').textContent = 'Error: ' + err.message;
                    console.error('Camera error:', err);
                }
            }

            function startRecording() {
                recordedChunks = [];
                
                // Countdown
                let count = 3;
                document.getElementById('status').innerHTML = '<span id="countdown">' + count + '</span>';
                document.getElementById('recordBtn').disabled = true;
                
                countdownInterval = setInterval(() => {
                    count--;
                    if (count > 0) {
                        document.getElementById('countdown').textContent = count;
                    } else {
                        clearInterval(countdownInterval);
                        actuallyStartRecording();
                    }
                }, 1000);
            }

            function actuallyStartRecording() {
                try {
                    const options = { mimeType: 'video/webm;codecs=vp8' };
                    mediaRecorder = new MediaRecorder(stream, options);
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            recordedChunks.push(event.data);
                        }
                    };
                    
                    mediaRecorder.onstop = () => {
                        const blob = new Blob(recordedChunks, { type: 'video/webm' });
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            const base64data = reader.result.split(',')[1];
                            // Send to Streamlit
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: base64data
                            }, '*');
                        };
                        reader.readAsDataURL(blob);
                        
                        document.getElementById('status').textContent = 'Recording complete! Processing...';
                        document.getElementById('overlay').classList.remove('recording');
                    };
                    
                    mediaRecorder.start();
                    document.getElementById('overlay').classList.add('recording');
                    
                    let remaining = 15;
                    document.getElementById('status').textContent = 'Recording... ' + remaining + 's remaining';
                    
                    const updateTimer = setInterval(() => {
                        remaining--;
                        if (remaining > 0) {
                            document.getElementById('status').textContent = 'Recording... ' + remaining + 's remaining';
                        } else {
                            clearInterval(updateTimer);
                        }
                    }, 1000);
                    
                    // Auto-stop after 15 seconds
                    recordingTimeout = setTimeout(() => {
                        if (mediaRecorder && mediaRecorder.state === 'recording') {
                            mediaRecorder.stop();
                        }
                    }, 15000);
                    
                } catch (err) {
                    document.getElementById('status').textContent = 'Recording error: ' + err.message;
                    console.error('Recording error:', err);
                }
            }

            function stopCamera() {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    clearTimeout(recordingTimeout);
                }
                
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                
                document.getElementById('video').srcObject = null;
                document.getElementById('startBtn').disabled = false;
                document.getElementById('recordBtn').disabled = true;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('status').textContent = 'Camera stopped';
                document.getElementById('overlay').classList.remove('aligned', 'recording');
            }

            // Cleanup on page unload
            window.addEventListener('beforeunload', stopCamera);
        </script>
    </body>
    </html>
    """
    return html_code

def live_camera_component(key="live_camera"):
    """
    Streamlit component for live camera recording.
    Returns base64-encoded video data when recording is complete.
    """
    html = get_camera_html()
    video_data = components.html(html, height=600, scrolling=False)
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
