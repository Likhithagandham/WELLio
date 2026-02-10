import os
import streamlit.components.v1 as components
import base64
import io
import av
import numpy as np
from typing import List, Tuple, Optional

# Absolute path to the component directory
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPONENT_DIR = os.path.join(PARENT_DIR, "components", "video_recorder")

# Declare the component
_component_func = components.declare_component(
    "video_recorder",
    path=COMPONENT_DIR
)

def video_recorder(key=None):
    """
    Renders the custom video recorder component.
    Returns a dictionary with 'data' (base64) and 'type' if recorded, else None.
    """
    return _component_func(key=key, default=None)

def process_recorder_output(recorder_output) -> Optional[Tuple[List[np.ndarray], float]]:
    """
    Converts the base64 output from the recorder component into a list of NumPy frames and FPS.
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
        fps = float(video_stream.average_rate or video_stream.r_frame_rate)
        
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
