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
class StressResult:
    label: str
    level: Optional[str]
    score: Optional[int]
    reason: str
    signals_used: List[str]

@dataclass
class BPResult:
    systolic: int
    diastolic: int
    label: str
    reason: str
    confidence: int
    disclaimer: str = "This blood pressure value is an experimental estimate derived from optical pulse signals and physiological correlations. It is not a medical measurement."

@dataclass
class VitalsEstimate:
    heart_rate_bpm: Optional[float]
    heart_rate_valid: bool
    rr_intervals: np.ndarray
    sdnn: Optional[float]
    stress_level: Optional[float] # Kept for compatibility (0-10)
    stress_details: Optional[StressResult] # New structured details
    bp_systolic: Optional[float]
    bp_diastolic: Optional[float]
    bp_details: Optional[BPResult] # New structured details
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

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(frame)
    cap.release()
    
    return extract_signals_from_frames(frames, fs)

def extract_signals_from_frames(frames: List[np.ndarray], fs: float) -> MultiROISignals:
    if not frames:
        raise RuntimeError("No frames to process")

    g_f, g_l, g_r = [], [], []
    centers, brightness = [], []
    valid = total = 0

    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1) if USE_MEDIAPIPE else None
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for frame in frames:
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

        if not face_found:
             # Fallback to Haar
             faces = haar.detectMultiScale(gray, 1.3, 5)
             if len(faces) > 0:
                 x,y,w_box,h_box = faces[0]
                 roi = frame[y:y+h_box, x:x+w_box]
                 g = np.mean(roi[:,:,1])
                 g_f.append(g); g_l.append(g); g_r.append(g)
                 centers.append([x+w_box//2, y+h_box//2])
                 face_found = True

        if not face_found:
            g_f.append(np.nan); g_l.append(np.nan); g_r.append(np.nan)
        else:
            valid += 1

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

    raw_conf = (
        0.5*min((peak_ratio if (peak_ratio and not np.isnan(peak_ratio)) else 0)*250, 100) +
        0.3*max(0, 100-(motion if not np.isnan(motion) else 0)*12) +
        0.2*max(0, 100-(lighting if not np.isnan(lighting) else 0)*8)
    )
    confidence = int(raw_conf) if not np.isnan(raw_conf) else 0
    quality = round((confidence/100)*10,1)

    return FilteredSignal(fused, sig.fps, freqs, psd, confidence, quality)

def calculate_stress_score(hrv_sdnn: Optional[float], rr_interval_ms: Optional[float]) -> StressResult:
    """
    Compute stress score based on HRV (SDNN) and RR Interval.
    
    Thresholds:
    - HRV (SDNN): Normal >= 30, Low 20-29, Very Low < 20
    - RR Interval: Normal 600-1000, Elevated 500-599, Highly Elevated < 500
    
    Scoring:
    - Normal (Score 2): HRV Normal AND RR Normal
    - Mild Stress (Score 5): HRV Low OR RR Elevated
    - High Stress (Score 8): HRV Very Low AND RR Highly Elevated
    """
    signals_used = []
    if hrv_sdnn is not None: signals_used.append("HRV")
    if rr_interval_ms is not None: signals_used.append("RR")
    
    if not signals_used:
        return StressResult("Unavailable", None, None, "Insufficient signal for stress computation", [])
        
    # Classify HRV
    hrv_status = "Normal"
    if hrv_sdnn is not None:
        if hrv_sdnn < 20: hrv_status = "Very Low"
        elif hrv_sdnn < 30: hrv_status = "Low"
        else: hrv_status = "Normal"
        
    # Classify RR
    rr_status = "Normal"
    if rr_interval_ms is not None:
        if rr_interval_ms < 500: rr_status = "Highly Elevated"
        elif rr_interval_ms < 600: rr_status = "Elevated"
        else: rr_status = "Normal"
        
    # Compute Score
    score = 2
    label = "Normal"
    level = "NORMAL"
    reason = "Vitals within normal range"
    
    # Logic
    if hrv_status == "Very Low" and rr_status == "Highly Elevated":
        score = 8
        label = "High Stress"
        level = "HIGH"
        reason = "Very low HRV and highly elevated RR interval detected"
    elif hrv_status == "Low" or rr_status == "Elevated" or hrv_status == "Very Low" or rr_status == "Highly Elevated":
        # Any deviation that isn't the worst case falls here (conservative)
        # Re-evaluating exact logic from prompt:
        # Mild: HRV ↓ OR RR ↑
        # High: HRV ↓↓ AND RR ↑↑
        # So "Very Low" OR "Highly Elevated" alone is at least Mild.
        score = 5
        label = "Mild Stress"
        level = "MILD"
        reason = "Low HRV or elevated RR interval detected"
        
    # Refine High Stress to match "AND" condition strictly if both present
    if "HRV" in signals_used and "RR" in signals_used:
        if hrv_status == "Very Low" and rr_status == "Highly Elevated":
            score = 8
            label = "High Stress"
            level = "HIGH"
            reason = "Very low HRV and highly elevated RR interval detected"
    
    # Handle single signal cases
    # If only HRV available:
    if "HRV" in signals_used and "RR" not in signals_used:
        if hrv_status == "Very Low":
             score = 8; label="High Stress"; level="HIGH"; reason="Very low HRV detected (RR unavailable)"
        elif hrv_status == "Low":
             score = 5; label="Mild Stress"; level="MILD"; reason="Low HRV detected (RR unavailable)"
             
    # If only RR available
    if "RR" in signals_used and "HRV" not in signals_used:
        if rr_status == "Highly Elevated":
            score = 8; label="High Stress"; level="HIGH"; reason="Highly elevated RR interval detected (HRV unavailable)"
        elif rr_status == "Elevated":
            score = 5; label="Mild Stress"; level="MILD"; reason="Elevated RR interval detected (HRV unavailable)"

    return StressResult(label, level, score, reason, signals_used)

def get_bp_category(sbp: Optional[float], dbp: Optional[float]) -> str:
    """
    Get BP Label based on Max Severity Rule.
    
    Severity Order: Low < Normal < High-Normal < High
    
    Step 1: Classify SBP
    < 90       -> Low (0)
    90-119     -> Normal (1)
    120-139    -> High-Normal (2)
    >= 140     -> High (3)
    
    Step 2: Classify DBP
    < 60       -> Low (0)
    60-79      -> Normal (1)
    80-89      -> High-Normal (2)
    >= 90      -> High (3)
    
    Step 3: Label = Category with Max Severity
    """
    if sbp is None and dbp is None:
        return "Unavailable"

    # Define categories and severity
    CAT_LOW = 0
    CAT_NORMAL = 1
    CAT_HIGH_NORMAL = 2
    CAT_HIGH = 3
    
    severity_map = {
        CAT_LOW: "⬇️ Low",
        CAT_NORMAL: "✅ Normal",
        CAT_HIGH_NORMAL: "⬆️ High-Normal",
        CAT_HIGH: "⬆️ High"
    }

    # Classify SBP
    s_sev = -1
    if sbp is not None:
        if sbp < 90: s_sev = CAT_LOW
        elif sbp < 120: s_sev = CAT_NORMAL
        elif sbp < 140: s_sev = CAT_HIGH_NORMAL
        else: s_sev = CAT_HIGH

    # Classify DBP
    d_sev = -1
    if dbp is not None:
        if dbp < 60: d_sev = CAT_LOW
        elif dbp < 80: d_sev = CAT_NORMAL
        elif dbp < 90: d_sev = CAT_HIGH_NORMAL
        else: d_sev = CAT_HIGH

    # Combine
    final_sev = -1
    
    if s_sev != -1 and d_sev != -1:
        final_sev = max(s_sev, d_sev)
    elif s_sev != -1:
        final_sev = s_sev
    elif d_sev != -1:
        final_sev = d_sev
        
    return severity_map.get(final_sev, "Unavailable")

def calculate_bp(heart_rate: Optional[float], hrv_sdnn: Optional[float], 
                 stress_level: Optional[str], rr_interval_ms: Optional[float], 
                 confidence: int) -> BPResult:
    """
    Heuristic BP estimation. NOT A MEDICAL MEASUREMENT.
    Base: 115/75
    Modifiers: HR, HRV, Stress, RR
    """
    # 1. Base Values
    sbp = 115
    dbp = 75
    reasons = []

    # 2. Modifiers
    
    # Heart Rate
    if heart_rate and not np.isnan(heart_rate):
        if heart_rate > 100:
            sbp += 10; dbp += 5
            reasons.append("Elevated HR")
        elif heart_rate < 50:
            sbp -= 10; dbp -= 5
            reasons.append("Low HR")

    # HRV
    if hrv_sdnn and not np.isnan(hrv_sdnn):
        if hrv_sdnn < 20:
            sbp += 10
            reasons.append("Very Low HRV")
        elif hrv_sdnn < 30: # 20-29
            sbp += 5
            reasons.append("Low HRV")

    # Stress Level (from backend calculation)
    if stress_level:
        if stress_level == "MILD":
            sbp += 5; dbp += 3
            reasons.append("Mild Stress")
        elif stress_level == "HIGH":
            sbp += 15; dbp += 8
            reasons.append("High Stress")

    # RR Interval
    if rr_interval_ms and not np.isnan(rr_interval_ms):
        if rr_interval_ms < 500:
            sbp += 10
            reasons.append("Very Low RR")
        elif rr_interval_ms < 600:
            sbp += 5
            reasons.append("Low RR")

    # 3. Clamping
    sbp = max(80, min(200, sbp))
    dbp = max(50, min(130, dbp))

    # 4. Labeling
    label = get_bp_category(sbp, dbp)

    reason = "Experimental estimate based on physiological metrics."
    if reasons:
        reason = f"Estimate adjusted due to: {', '.join(reasons)}."

    # Signal Confidence Handling
    if confidence < 55:
        reason += " [Low signal confidence - unreliable]"

    return BPResult(
        systolic=int(sbp), 
        diastolic=int(dbp), 
        label=label, 
        reason=reason, 
        confidence=confidence
    )

# ============================================================
# VITALS
# ============================================================

def estimate_vitals(sig: MultiROISignals, filt: FilteredSignal) -> VitalsEstimate:
    hr, _, _, _ = welch_hr(filt.fused_signal, sig.fps)
    hr_valid = hr is not None and not np.isnan(hr) and 40 <= hr <= 180 and filt.confidence_percent >= 40 

    rr = rr_from_peaks(filt.fused_signal, sig.fps, hr) if hr_valid else np.array([])
    sdnn = float(np.std(rr)*1000) if len(rr)>=5 else None

    # New Stress Logic
    stress_res = calculate_stress_score(sdnn, np.mean(rr)*1000 if len(rr)>0 else None)
    stress_level = float(stress_res.score) if stress_res.score is not None else None

    # New BP Logic
    conf_val = int(filt.confidence_percent) if not np.isnan(filt.confidence_percent) else 0
    bp_res = calculate_bp(
        heart_rate=hr,
        hrv_sdnn=sdnn,
        stress_level=stress_res.level, # "NORMAL", "MILD", "HIGH"
        rr_interval_ms=np.mean(rr)*1000 if len(rr)>0 else None,
        confidence=conf_val
    )
    
    bp_s = float(bp_res.systolic)
    bp_d = float(bp_res.diastolic)

    spo2 = None
    if sig.g_forehead.size > 0:
        mean_g = np.mean(sig.g_forehead)
        if mean_g > 0:
            val = np.std(sig.g_forehead)/ (mean_g + 1e-9)
            spo2 = float(np.clip(104 - 18*val, 70, 99))

    return VitalsEstimate(hr, hr_valid, rr, sdnn, stress_level, stress_res, bp_s, bp_d, bp_res, spo2)

# ============================================================
# RISK
# ============================================================

def assess_risk(v: VitalsEstimate) -> RiskAssessment:
    score = 0
    alerts = []

    if v.heart_rate_bpm and not np.isnan(v.heart_rate_bpm):
        if v.heart_rate_bpm > 140:
            alerts.append("High HR")
            score += 2
        if v.heart_rate_bpm < 45:
            alerts.append("Low HR")
            score += 2

    if v.sdnn and not np.isnan(v.sdnn) and v.sdnn < 20:
        alerts.append("Low HRV")
        score += 2

    if v.spo2 and not np.isnan(v.spo2) and v.spo2 < 92:
        alerts.append("Low SpO2")
        score += 3

    level = "HIGH" if score>=5 else "MODERATE" if score>=2 else "LOW"
    rec = "Consult medical professional" if level!="LOW" else "Vitals look stable (experimental)"

    return RiskAssessment(score, level, alerts, rec)

# ============================================================
# MAIN API FUNCTION
# ============================================================

def estimate_vitals_from_video(video_path: str, use_mediapipe: bool = True):
    signals = extract_signals(video_path)
    return process_and_assess(signals)

def estimate_vitals_from_frames(frames: List[np.ndarray], fs: float):
    signals = extract_signals_from_frames(frames, fs)
    return process_and_assess(signals)

def process_and_assess(signals: MultiROISignals):
    filtered = process_signals(signals)
    vitals = estimate_vitals(signals, filtered)
    risk = assess_risk(vitals)
    return vitals, filtered, risk
