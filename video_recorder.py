import os
import streamlit.components.v1 as components
import base64
import io
import av
import numpy as np
from typing import List, Tuple, Optional

# Absolute path to the component directory
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Standalone recorder directory
COMPONENT_DIR = os.path.abspath(os.path.join(PARENT_DIR, "standalone_recorder"))

# Declare the component
if not os.path.exists(os.path.join(COMPONENT_DIR, "index.html")):
    _component_func = None
    _asset_error = f"Standalone recorder assets not found at {COMPONENT_DIR}"
else:
    _component_func = components.declare_component(
        "video_recorder",
        path=COMPONENT_DIR
    )
    _asset_error = None

def video_recorder(key: Optional[str] = None):
    """
    Renders the custom standalone video recorder as a Streamlit component.
    """
    if _component_func is None:
        import streamlit as st
        st.error(_asset_error)
        return None
    return _component_func(key=key, default=None)

def process_recorder_output(recorder_output) -> Optional[Tuple[List[np.ndarray], float]]:
    """
    Decodes the base64 output from the recorder component into a list of NumPy frames and FPS.
    """
    if not recorder_output or 'data' not in recorder_output:
        return None

    try:
        # Strip the data:video/...;base64, prefix
        b64_data = recorder_output['data'].split(',')[1]
        video_bytes = base64.b64decode(b64_data)
        
        # Open with PyAV
        container = av.open(io.BytesIO(video_bytes))
        frames = []
        
        # Get FPS
        video_stream = container.streams.video[0]
        # Our JS sets captureStream(30), but MediaRecorder might vary slightly
        # We'll use the container's reported rate or fallback to 30.
        fps = float(video_stream.average_rate or video_stream.r_frame_rate)
        if fps < 1: fps = 30.0
        
        # Decode frames
        for frame in container.decode(video=0):
            # Convert to BGR for backend consistency
            img = frame.to_ndarray(format='bgr24')
            frames.append(img)
            
        container.close()
        
        if not frames:
            return None
            
        return frames, fps

    except Exception as e:
        import streamlit as st
        st.error(f"Error processing recorded video: {e}")
        return None
