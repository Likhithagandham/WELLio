"""
Unified rPPG Backend (Research / Demo Only)
===========================================

- Multi-ROI rPPG (Forehead + Cheeks)
- MediaPipe landmarks with Haar fallback
- ROI fusion via PSD quality
- Motion & lighting aware confidence
- HR, HRV (SDNN), Stress (heuristic)
- Experimental BP & SpO2
- Risk assessment

NOT CLINICALLY VALIDATED
"""

import os
import cv2
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from scipy.signal import butter, filtfilt, welch, find_peaks, detrend

warnings.filterwarnings("ignore")

# ============================================================
# MEDIAPIPE CHECK
# ============================================================

USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    _ = mp.solutions.face_mesh
    USE_MEDIAPIPE = True
except Exception:
    USE_MEDIAPIPE = False

# ALIAS FOR COMPATIBILITY
HAVE_MEDIAPIPE = USE_MEDIAPIPE

# ============================================================
# ROI LANDMARKS
# ============================================================

ROI_FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365]
ROI_LEFT_CHEEK = [50, 101, 118, 119, 120, 47, 126, 142, 203, 206]
ROI_RIGHT_CHEEK = [280, 330, 347, 348, 349, 277, 355, 371, 423, 426]

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MultiROISignals:
    g_forehead: np.ndarray
    g_left: np.ndarray
    g_right: np.ndarray
    fps: float
    face_centers: np.ndarray
    brightness: np.ndarray
    valid_frames: int
    total_frames: int

@dataclass
class FilteredSignal:
    fused_signal: np.ndarray
    fps: float
    psd_freqs: np.ndarray
    psd: np.ndarray
    confidence_percent: int
    signal_quality_score: float

@dataclass
class VitalsEstimate:
    heart_rate_bpm: Optional[float]
    heart_rate_valid: bool
    rr_intervals: np.ndarray
    sdnn: Optional[float]
    stress_level: Optional[float]
    bp_systolic: Optional[float]
    bp_diastolic: Optional[float]
    spo2: Optional[float]

@dataclass
class RiskAssessment:
    risk_score: int
    risk_level: str
    alerts: List[str]
    recommendation: str

# ============================================================
# UTILS
# ============================================================

def polygon_from_landmarks(lm, idxs, w, h):
    return np.array(
        [[int(lm[i].x * w), int(lm[i].y * h)] for i in idxs],
        dtype=np.int32
    )

def mean_rgb_in_polygon(frame_bgr, poly):
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb[mask == 255]
    return pixels.mean(axis=0) if pixels.size else (np.nan, np.nan, np.nan)

def normalize(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-9
    return (x - med) / mad

def bandpass(sig, fs, low=0.7, high=4.0):
    nyq = 0.5 * fs
    b, a = butter(3, [low/nyq, high/nyq], btype="band")
    sig = np.nan_to_num(sig, nan=np.nanmedian(sig))
    return filtfilt(b, a, sig)

def welch_hr(sig, fs):
    freqs, psd = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band):
        return None, 0.0, freqs, psd
    p = psd[band]
    f = freqs[band]
    i = np.argmax(p)
    return f[i]*60.0, p[i]/(np.sum(p)+1e-9), freqs, psd

def rr_from_peaks(sig, fs, hr):
    if hr is None:
        return np.array([])
    min_dist = int(0.5 * (60/hr) * fs)
    peaks, _ = find_peaks(sig, distance=min_dist)
    rr = np.diff(peaks)/fs
    return rr[(rr > 0.3) & (rr < 2.0)]

# ============================================================
# SIGNAL EXTRACTION
# ============================================================

def extract_signals(video_path: str) -> MultiROISignals:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fs = cap.get(cv2.CAP_PROP_FPS)
    if fs <= 1 or fs > 120:
        fs = 30.0

    g_f, g_l, g_r = [], [], []
    centers, brightness = [], []
    valid = total = 0

    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1) if USE_MEDIAPIPE else None
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))

        face_found = False

        if USE_MEDIAPIPE and mp_face:
            res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                cx, cy = int(lm[1].x*w), int(lm[1].y*h)
                centers.append([cx, cy])
                pf = polygon_from_landmarks(lm, ROI_FOREHEAD, w, h)
                pl = polygon_from_landmarks(lm, ROI_LEFT_CHEEK, w, h)
                pr = polygon_from_landmarks(lm, ROI_RIGHT_CHEEK, w, h)
                _, gf, _ = mean_rgb_in_polygon(frame, pf)
                _, gl, _ = mean_rgb_in_polygon(frame, pl)
                _, gr, _ = mean_rgb_in_polygon(frame, pr)
                g_f.append(gf); g_l.append(gl); g_r.append(gr)
                face_found = True

        if not face_found and not (USE_MEDIAPIPE and mp_face):
             # Fallback to Haar if MP not used or configured
             faces = haar.detectMultiScale(gray, 1.3, 5)
             if len(faces) > 0:
                 x,y,w_box,h_box = faces[0]
                 # Simple center extraction for Haar fallback
                 roi = frame[y:y+h_box, x:x+w_box]
                 g = np.mean(roi[:,:,1])
                 g_f.append(g); g_l.append(g); g_r.append(g) # duplicate for now
                 centers.append([x+w_box//2, y+h_box//2])
                 face_found = True

        if not face_found:
            g_f.append(np.nan); g_l.append(np.nan); g_r.append(np.nan)
        else:
            valid += 1

    cap.release()
    if mp_face:
        mp_face.close()

    return MultiROISignals(
        np.array(g_f), np.array(g_l), np.array(g_r),
        fs,
        np.array(centers) if centers else np.zeros((0,2)),
        np.array(brightness),
        valid,
        total
    )

# ============================================================
# PROCESSING + FUSION
# ============================================================

def process_signals(sig: MultiROISignals) -> FilteredSignal:
    n_f = normalize(sig.g_forehead)
    n_l = normalize(sig.g_left)
    n_r = normalize(sig.g_right)

    f_f = bandpass(n_f, sig.fps)
    f_l = bandpass(n_l, sig.fps)
    f_r = bandpass(n_r, sig.fps)

    hr_f, q_f, _, _ = welch_hr(f_f, sig.fps)
    hr_l, q_l, _, _ = welch_hr(f_l, sig.fps)
    hr_r, q_r, _, _ = welch_hr(f_r, sig.fps)

    q = np.array([q_f, q_l, q_r])
    # Handle NaN/None
    q = np.nan_to_num(q)
    
    if q.sum() > 0:
        w = q/q.sum()
    else:
        w = np.array([1/3]*3)

    fused = w[0]*f_f + w[1]*f_l + w[2]*f_r
    hr, peak_ratio, freqs, psd = welch_hr(fused, sig.fps)

    motion = 0
    if len(sig.face_centers) > 5 and len(sig.face_centers) == len(sig.g_forehead):
        # basic motion metric
        diffs = np.diff(sig.face_centers, axis=0)
        motion = np.median(np.linalg.norm(diffs, axis=1))
    
    lighting = 0
    if len(sig.brightness) > 5:
        lighting = np.std(np.diff(sig.brightness))

    confidence = int(
        0.5*min((peak_ratio if peak_ratio else 0)*250,100) +
        0.3*max(0,100-motion*12) +
        0.2*max(0,100-lighting*8)
    )
    quality = round((confidence/100)*10,1)

    return FilteredSignal(fused, sig.fps, freqs, psd, confidence, quality)

# ============================================================
# VITALS
# ============================================================

def estimate_vitals(sig: MultiROISignals, filt: FilteredSignal) -> VitalsEstimate:
    hr, _, _, _ = welch_hr(filt.fused_signal, sig.fps)
    hr_valid = hr is not None and 40 <= hr <= 180 and filt.confidence_percent >= 40 

    rr = rr_from_peaks(filt.fused_signal, sig.fps, hr) if hr_valid else np.array([])
    sdnn = float(np.std(rr)*1000) if len(rr)>=5 else None

    stress = None
    if sdnn:
        stress = 8 if sdnn < 20 else 4 if sdnn < 50 else 1

    bp_s = 120.0
    if hr and hr < 80:
        bp_s = 120.0
    else:
        bp_s = 135.0
        
    bp_d = bp_s - 40.0

    spo2 = None
    if sig.g_forehead.size > 0:
        val = np.std(sig.g_forehead)/ (np.mean(sig.g_forehead) + 1e-9)
        spo2 = float(np.clip(104 - 18*val, 70, 99))

    return VitalsEstimate(hr, hr_valid, rr, sdnn, stress, bp_s, bp_d, spo2)

# ============================================================
# RISK
# ============================================================

def assess_risk(v: VitalsEstimate) -> RiskAssessment:
    score = 0
    alerts = []

    if v.heart_rate_bpm:
        if v.heart_rate_bpm > 140:
            alerts.append("High HR")
            score += 2
        if v.heart_rate_bpm < 45:
            alerts.append("Low HR")
            score += 2

    if v.sdnn and v.sdnn < 20:
        alerts.append("Low HRV")
        score += 2

    if v.spo2 and v.spo2 < 92:
        alerts.append("Low SpO2")
        score += 3

    level = "HIGH" if score>=5 else "MODERATE" if score>=2 else "LOW"
    rec = "Consult medical professional" if level!="LOW" else "Vitals look stable (experimental)"

    return RiskAssessment(score, level, alerts, rec)

# ============================================================
# MAIN API FUNCTION
# ============================================================

def estimate_vitals_from_video(video_path: str, use_mediapipe: bool = True):
    # Ignoring use_mediapipe arg since global toggle controls it in this script
    # but keeping signature for compatibility
    signals = extract_signals(video_path)
    filtered = process_signals(signals)
    vitals = estimate_vitals(signals, filtered)
    risk = assess_risk(vitals)
    return vitals, filtered, risk
