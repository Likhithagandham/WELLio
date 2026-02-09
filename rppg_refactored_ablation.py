"""
Remote Photoplethysmography (rPPG) System - Refactored Modular Version
========================================================================

DISCLAIMER:
-----------
This is a research/demo prototype. Outputs are experimental and NOT clinically validated.
Do NOT use for medical diagnosis or treatment. Consult a healthcare professional.

Pipeline:
1. Face Detection (MediaPipe / Haar)
2. Signal Extraction (green channel temporal)
3. Signal Processing (detrending, interpolation, filtering)
4. Vitals Estimation (HR, HRV, experimental BP)
5. Risk Scoring (heuristic, for demo only)
6. Visualization (plots, alerts)

Architecture:
- signal_extraction.py: ROI extraction, channel averaging
- signal_processor.py: Filtering, detrending, quality checks
- vitals_estimator.py: HR, RR, HRV, experimental BP
- risk_scorer.py: Heuristic risk rules
- ui_streamlit.py: Streamlit interface
- api_fastapi.py: FastAPI backend (for web/mobile)

"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import cv2

try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = False  # Disable MediaPipe due to compatibility issues
except ImportError:
    HAVE_MEDIAPIPE = False

from scipy.signal import butter, filtfilt, find_peaks, welch, detrend, periodogram
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================

class SignalConfig:
    """Signal processing hyperparameters"""
    BANDPASS_LOW = 0.75  # Hz (~45 BPM)
    BANDPASS_HIGH = 3.0  # Hz (~180 BPM)
    BANDPASS_ORDER = 4
    FPS_DEFAULT = 30  # fallback FPS
    MIN_DETECTION_RATIO = 0.25  # min 25% of frames must have face
    NaN_INTERPOLATION_LIMIT = 0.5  # max 50% NaN allowed before warning

class HRConfig:
    """Heart rate estimation parameters"""
    MIN_HR = 30  # BPM
    MAX_HR = 220  # BPM
    MIN_PEAKS_FOR_HRV = 3  # need at least 3 peaks for RR variance
    PEAK_MIN_DISTANCE = 0.4  # seconds between heartbeats (150 BPM max)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ROIBbox:
    """Bounding box for ROI"""
    x: int
    y: int
    w: int
    h: int
    
    def is_valid(self, img_h: int, img_w: int) -> bool:
        """Check if bbox is within image bounds and non-zero size"""
        return (self.x >= 0 and self.y >= 0 and 
                self.w > 2 and self.h > 2 and
                self.x + self.w <= img_w and 
                self.y + self.h <= img_h)


@dataclass
class SignalData:
    """Extracted and preprocessed signals"""
    green: np.ndarray  # green channel over time
    red: np.ndarray    # red channel over time
    fps: float
    detected_frames: int
    total_frames: int
    

@dataclass
class FilteredSignal:
    """Filtered signal with metadata"""
    signal: np.ndarray  # filtered signal
    sampling_rate: float
    psd: np.ndarray     # power spectral density
    psd_freqs: np.ndarray  # frequency axis
    snr: float          # signal-to-noise ratio (power in HR band / outside)
    quality_flags: Dict[str, bool]  # quality checks (motion_high, nan_high, etc.)


@dataclass
class VitalsEstimate:
    """Vitals estimation output"""
    heart_rate_bpm: float
    heart_rate_confidence: str  # "HIGH", "MEDIUM", "LOW"
    rr_intervals: np.ndarray  # in milliseconds
    sdnn: float  # std of RR intervals (ms)
    pnn50: float  # percentage of RR intervals > 50ms difference
    stress_level: Optional[float]  # 0-10, experimental
    bp_systolic: Optional[float]
    bp_diastolic: Optional[float]
    bp_note: str  # disclaimer
    spo2: Optional[float]
    spo2_note: str  # disclaimer
    

@dataclass 
class RiskAssessment:
    """Risk scoring output"""
    risk_score: int
    risk_level: str  # "LOW", "MODERATE", "HIGH"
    alerts: List[str]  # list of concerns
    recommendation: str


# ============================================================================
# MODULE 1: FACE DETECTION & ROI EXTRACTION
# ============================================================================

class FaceDetector:
    """Unified face detection (MediaPipe preferred, Haar fallback)"""
    
    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and HAVE_MEDIAPIPE
        self.mp_face = None
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.use_mediapipe:
            mp_face_mesh = mp.solutions.face_mesh
            self.mp_face = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )
    
    def get_forehead_roi(self, face_bbox: Tuple[int, int, int, int]) -> ROIBbox:
        """Extract forehead region from face bounding box"""
        x, y, w, h = face_bbox
        fx = int(x + w * 0.3)
        fy = int(y + h * 0.08)  # forehead is ~8% down
        fw = int(w * 0.4)  # ~40% of face width
        fh = int(h * 0.18)  # ~18% of face height
        return ROIBbox(fx, fy, fw, fh)
    
    def mediapipe_forehead_bbox(self, landmarks, img_w: int, img_h: int) -> Optional[ROIBbox]:
        """Compute forehead ROI from MediaPipe landmarks"""
        try:
            xs = np.array([lm.x for lm in landmarks])
            ys = np.array([lm.y for lm in landmarks])
            min_x, max_x = xs.min() * img_w, xs.max() * img_w
            min_y, max_y = ys.min() * img_h, ys.max() * img_h
            face_bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
            return self.get_forehead_roi(face_bbox)
        except Exception:
            return None
    
    def detect(self, frame: np.ndarray) -> Optional[ROIBbox]:
        """Detect face and return forehead ROI"""
        h_img, w_img = frame.shape[:2]
        
        # Try MediaPipe first
        if self.mp_face is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                roi = self.mediapipe_forehead_bbox(lm, w_img, h_img)
                if roi and roi.is_valid(h_img, w_img):
                    return roi
        
        # Fallback to Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
        )
        if len(faces) > 0:
            face = max(faces, key=lambda r: r[2] * r[3])  # largest face
            roi = self.get_forehead_roi(face)
            if roi.is_valid(h_img, w_img):
                return roi
        
        return None
    
    def close(self):
        """Cleanup resources"""
        if self.mp_face:
            self.mp_face.close()


# ============================================================================
# MODULE 2: SIGNAL EXTRACTION FROM VIDEO
# ============================================================================

class SignalExtractor:
    """Extract temporal color signals from ROI"""
    
    def __init__(self, detector: FaceDetector):
        self.detector = detector
    
    def extract(self, video_path: str, progress_callback=None) -> SignalData:
        """
        Extract green and red channel signals from video.
        
        Args:
            video_path: path to video file
            progress_callback: optional function(current, total) for UI updates
            
        Returns:
            SignalData with green, red, fps, detection stats
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = SignalConfig.FPS_DEFAULT
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        
        greens, reds = [], []
        detected_frames = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            roi = self.detector.detect(frame)
            if roi is not None:
                x, y, w, h = roi.x, roi.y, roi.w, roi.h
                patch = frame[y:y+h, x:x+w]
                if patch.size > 0:
                    # Extract channels (BGR in OpenCV)
                    greens.append(float(np.mean(patch[:, :, 1])))
                    reds.append(float(np.mean(patch[:, :, 2])))
                    detected_frames += 1
                else:
                    greens.append(np.nan)
                    reds.append(np.nan)
            else:
                greens.append(np.nan)
                reds.append(np.nan)
            
            frame_idx += 1
            if progress_callback and total_frames:
                progress_callback(frame_idx, total_frames)
        
        cap.release()
        
        # Check detection quality
        if total_frames and detected_frames < total_frames * SignalConfig.MIN_DETECTION_RATIO:
            raise ValueError(
                f"Face detected in only {detected_frames}/{total_frames} frames "
                f"({100*detected_frames/total_frames:.1f}%). "
                "Try better lighting or adjust camera position."
            )
        
        return SignalData(
            green=np.array(greens, dtype=float),
            red=np.array(reds, dtype=float),
            fps=fps,
            detected_frames=detected_frames,
            total_frames=frame_idx or 0
        )


# ============================================================================
# MODULE 3: SIGNAL PROCESSING (Filtering, Detrending, Quality)
# ============================================================================

class SignalProcessor:
    """Preprocess signals: interpolation, detrending, filtering, quality checks"""
    
    @staticmethod
    def interpolate_nans(signal: np.ndarray, warn_threshold: float = 0.3) -> np.ndarray:
        """
        Interpolate NaN values in signal.
        
        Warns if >threshold of signal is NaN.
        Returns interpolated signal.
        """
        nan_ratio = np.isnan(signal).sum() / len(signal)
        if nan_ratio > warn_threshold:
            print(f"[WARNING] Signal has {100*nan_ratio:.1f}% NaN values. "
                  "Face detection may be unreliable.")
        
        if not np.isnan(signal).any():
            return signal
        
        # Use pandas interpolation for robustness
        series = pd.Series(signal)
        series = series.interpolate(method='linear', limit_direction='both')
        series = series.fillna(method='bfill').fillna(method='ffill')
        return series.values
    
    @staticmethod
    def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, 
                                fs: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth bandpass filter"""
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 1e-6)
        high = min(highcut / nyq, 0.999)
        
        if low >= high:
            raise ValueError(f"Invalid bandpass range: {lowcut}–{highcut} Hz at {fs} Hz sampling")
        
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    @staticmethod
    def compute_signal_quality(signal: np.ndarray, fps: float, 
                               bandpass_low: float, bandpass_high: float) -> Dict[str, bool]:
        """
        Compute quality flags for signal.
        
        Returns dict of quality indicators.
        """
        flags = {}
        
        # Check for low variance
        if np.std(signal) < 0.01:
            flags['low_variance'] = True
        else:
            flags['low_variance'] = False
        
        # Check for high motion (large jumps)
        jumps = np.abs(np.diff(signal))
        if np.percentile(jumps, 95) > 2.0 * np.std(signal):
            flags['high_motion'] = True
        else:
            flags['high_motion'] = False
        
        return flags
    
    @staticmethod
    def process(signal_data: SignalData) -> FilteredSignal:
        """
        Full preprocessing pipeline: interpolate → detrend → normalize → filter.
        
        Returns FilteredSignal with quality metrics.
        """
        # Interpolate NaNs
        green = SignalProcessor.interpolate_nans(signal_data.green)
        
        # Detrend
        green = detrend(green)
        
        # Normalize
        std_green = np.std(green)
        if std_green < 1e-8:
            raise ValueError("Processed signal has near-zero variance. Cannot estimate heart rate.")
        green = green / std_green
        
        # Bandpass filter
        try:
            filtered = SignalProcessor.butter_bandpass_filter(
                green, 
                SignalConfig.BANDPASS_LOW, 
                SignalConfig.BANDPASS_HIGH,
                signal_data.fps,
                order=SignalConfig.BANDPASS_ORDER
            )
        except Exception as e:
            raise RuntimeError(f"Filtering failed: {e}")
        
        # Welch PSD
        nperseg = min(256, len(filtered))
        freqs, psd = welch(filtered, fs=signal_data.fps, nperseg=nperseg)
        
        # Compute SNR: power in HR band vs. outside
        in_band = (freqs >= SignalConfig.BANDPASS_LOW) & (freqs <= SignalConfig.BANDPASS_HIGH)
        if np.any(in_band):
            power_in = np.sum(psd[in_band])
            power_out = np.sum(psd[~in_band]) + 1e-8
            snr = power_in / power_out
        else:
            snr = 0.0
        
        # Quality flags
        quality_flags = SignalProcessor.compute_signal_quality(
            filtered, signal_data.fps,
            SignalConfig.BANDPASS_LOW, SignalConfig.BANDPASS_HIGH
        )
        
        return FilteredSignal(
            signal=filtered,
            sampling_rate=signal_data.fps,
            psd=psd,
            psd_freqs=freqs,
            snr=snr,
            quality_flags=quality_flags
        )


# ============================================================================
# MODULE 4: VITALS ESTIMATION
# ============================================================================

class VitalsEstimator:
    """Estimate heart rate, HRV, and experimental vitals"""
    
    @staticmethod
    def estimate_heart_rate(filtered_signal: FilteredSignal) -> Tuple[float, str]:
        """
        Estimate heart rate from filtered signal via Welch PSD.
        
        Returns:
            (bpm, confidence_level)
        """
        valid_band = ((filtered_signal.psd_freqs >= SignalConfig.BANDPASS_LOW) & 
                      (filtered_signal.psd_freqs <= SignalConfig.BANDPASS_HIGH))
        
        if not np.any(valid_band):
            raise ValueError("No valid frequency bins in heart rate band.")
        
        peak_idx = np.argmax(filtered_signal.psd[valid_band])
        peak_freq = filtered_signal.psd_freqs[valid_band][peak_idx]
        bpm = peak_freq * 60.0
        
        # Sanity check
        if bpm < HRConfig.MIN_HR or bpm > HRConfig.MAX_HR:
            raise ValueError(
                f"Estimated BPM {bpm:.1f} out of plausible range "
                f"({HRConfig.MIN_HR}–{HRConfig.MAX_HR}). Check input video."
            )
        
        # Confidence based on SNR
        if filtered_signal.snr > 2.0:
            confidence = "HIGH"
        elif filtered_signal.snr > 1.0:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return bpm, confidence
    
    @staticmethod
    def estimate_rr_intervals(filtered_signal: FilteredSignal, fps: float) -> np.ndarray:
        """
        Detect peaks in filtered signal and compute RR intervals (in ms).
        
        Returns:
            array of RR intervals in milliseconds
        """
        signal = filtered_signal.signal
        
        # Peak detection parameters
        min_distance = int(max(1, HRConfig.PEAK_MIN_DISTANCE * fps))
        prominence = max(0.3 * np.std(signal), 0.01)
        
        peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
        
        # If few peaks, try less strict criteria
        if len(peaks) < HRConfig.MIN_PEAKS_FOR_HRV:
            peaks, _ = find_peaks(signal, distance=int(0.3 * fps))
        
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / fps * 1000.0  # convert to ms
        else:
            rr_intervals = np.array([])
        
        return rr_intervals
    
    @staticmethod
    def compute_hrv_metrics(rr_intervals: np.ndarray) -> Tuple[float, float]:
        """
        Compute HRV metrics from RR intervals.
        
        Returns:
            (sdnn, pnn50) – both in appropriate units
        """
        if len(rr_intervals) < HRConfig.MIN_PEAKS_FOR_HRV:
            return np.nan, np.nan
        
        sdnn = float(np.std(rr_intervals))
        
        # pNN50: percentage of consecutive RR intervals differing by >50 ms
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = 100.0 * nn50 / (len(rr_intervals) - 1) if len(rr_intervals) > 1 else 0.0
        
        return sdnn, float(pnn50)
    
    @staticmethod
    def estimate_stress_level(sdnn: float, pnn50: float) -> Optional[float]:
        """
        Heuristic stress estimation (0–10 scale, 0=calm, 10=high stress).
        
        Uses HRV metrics (pNN50 > SDNN alone).
        NOT clinically validated.
        """
        if np.isnan(pnn50):
            return np.nan
        
        # Use pNN50 as primary indicator
        if pnn50 < 5:
            stress = 8.0  # low HRV → high stress
        elif pnn50 < 15:
            stress = 5.5
        elif pnn50 < 30:
            stress = 3.0
        else:
            stress = 1.0  # high HRV → low stress (calm)
        
        return float(np.clip(stress, 0.0, 10.0))
    
    @staticmethod
    def experimental_bp_estimate(bpm: float, sdnn: float) -> Dict[str, any]:
        """
        Experimental BP estimation (NOT clinically validated).
        
        Returns dict with estimate and disclaimer.
        """
        # Simple heuristic based on BPM only (SDNN too weak)
        if bpm < 60:
            sbp_range = (105, 120)
        elif bpm < 80:
            sbp_range = (115, 130)
        else:
            sbp_range = (125, 145)
        
        sbp = (sbp_range[0] + sbp_range[1]) / 2
        # Approximate DBP
        dbp = sbp - 40
        
        return {
            "sbp": sbp,
            "dbp": dbp,
            "sbp_range": sbp_range,
            "note": (
                "EXPERIMENTAL ONLY. ±~15 mmHg error. "
                "NOT validated for clinical use. "
                "Not a replacement for traditional BP measurement."
            )
        }
    
    @staticmethod
    def experimental_spo2_estimate(signal_data: SignalData, filtered_signal: FilteredSignal) -> Dict[str, any]:
        """
        Experimental SpO₂ estimation using red/green AC/DC ratio from extracted signals.

        This is a highly approximate heuristic and NOT clinically validated. It
        uses the pulsatile AC amplitude (bandpassed) divided by DC (mean) for
        red and green channels, forms a ratio R = (AC_red/DC_red) / (AC_green/DC_green),
        and maps R to an SpO₂ range via a simple linear heuristic. The output is
        clamped to [70, 100].
        """
        try:
            # Prepare signals
            red = SignalProcessor.interpolate_nans(signal_data.red)
            green = SignalProcessor.interpolate_nans(signal_data.green)

            # Detrend + bandpass around typical heart band to get AC component
            red_bp = SignalProcessor.butter_bandpass_filter(
                detrend(red), SignalConfig.BANDPASS_LOW, SignalConfig.BANDPASS_HIGH, signal_data.fps, order=SignalConfig.BANDPASS_ORDER
            )
            green_bp = SignalProcessor.butter_bandpass_filter(
                detrend(green), SignalConfig.BANDPASS_LOW, SignalConfig.BANDPASS_HIGH, signal_data.fps, order=SignalConfig.BANDPASS_ORDER
            )

            # AC ~ std of bandpassed signal, DC ~ mean of original (clamped away from zero)
            ac_red = float(np.std(red_bp))
            ac_green = float(np.std(green_bp))
            dc_red = float(np.mean(red)) if abs(float(np.mean(red))) > 1e-6 else 1.0
            dc_green = float(np.mean(green)) if abs(float(np.mean(green))) > 1e-6 else 1.0

            # Ratio (pulse oximetry style) — avoid division by zero
            ratio = (ac_red / dc_red) / (ac_green / dc_green + 1e-12)

            # Map ratio to SpO2 via a conservative linear heuristic.
            # Coefficients chosen to produce plausible outputs for typical video signals.
            spo2 = 104.0 - 18.0 * ratio
            spo2 = float(np.clip(spo2, 70.0, 100.0))

            note = (
                "EXPERIMENTAL ONLY. SpO₂ estimated from visible-light red/green ratio. "
                "Highly approximate — ±10–20% possible. NOT clinically validated."
            )

            return {"spo2": spo2, "note": note}

        except Exception as e:
            return {"spo2": None, "note": f"SpO₂ estimation failed: {e}."}
    
    @staticmethod
    def estimate(filtered_signal: FilteredSignal, signal_data: SignalData) -> VitalsEstimate:
        """
        Full vitals estimation pipeline.
        
        Returns VitalsEstimate with all metrics and disclaimers.
        """
        # Use fps from the SignalData passed in
        fps = signal_data.fps

        # Heart rate
        bpm, hr_confidence = VitalsEstimator.estimate_heart_rate(filtered_signal)

        # RR intervals and HRV
        rr_intervals = VitalsEstimator.estimate_rr_intervals(filtered_signal, fps)
        sdnn, pnn50 = VitalsEstimator.compute_hrv_metrics(rr_intervals)
        
        # Stress
        stress = VitalsEstimator.estimate_stress_level(sdnn, pnn50)
        
        # Experimental vitals
        bp_result = VitalsEstimator.experimental_bp_estimate(bpm, sdnn)
        spo2_result = VitalsEstimator.experimental_spo2_estimate(signal_data, filtered_signal)
        
        return VitalsEstimate(
            heart_rate_bpm=bpm,
            heart_rate_confidence=hr_confidence,
            rr_intervals=rr_intervals,
            sdnn=sdnn,
            pnn50=pnn50,
            stress_level=stress,
            bp_systolic=bp_result["sbp"],
            bp_diastolic=bp_result["dbp"],
            bp_note=bp_result["note"],
            spo2=spo2_result["spo2"],
            spo2_note=spo2_result["note"]
        )


# ============================================================================
# MODULE 5: RISK ASSESSMENT
# ============================================================================

class RiskAssessor:
    """Heuristic risk scoring (experimental, NOT clinical)"""
    
    @staticmethod
    def assess(vitals: VitalsEstimate) -> RiskAssessment:
        """
        Compute cardiac risk score based on vitals.
        
        DISCLAIMER: This is a HEURISTIC for demo/research only.
        NOT a clinical tool. Not for diagnosis.
        """
        alerts = []
        score = 0

        bpm = vitals.heart_rate_bpm
        sdnn = vitals.sdnn

        # Heart rate weighting
        if bpm is not None:
            if bpm > 180:
                alerts.append("Very high heart rate (severe tachycardia)")
                score += 4
            elif bpm > 140:
                alerts.append("High heart rate (tachycardia)")
                score += 3
            elif bpm > 120:
                alerts.append("Elevated heart rate")
                score += 2
            elif bpm < 35:
                alerts.append("Very low heart rate (severe bradycardia)")
                score += 4
            elif bpm < 50:
                alerts.append("Low heart rate (bradycardia)")
                score += 2

        # HRV weighting (SDNN)
        if not np.isnan(sdnn):
            if sdnn < 10:
                alerts.append("Extremely low HRV (very high risk)")
                score += 3
            elif sdnn < 20:
                alerts.append("Low HRV (autonomic stress)")
                score += 2
            elif sdnn < 30:
                score += 1

        # Blood pressure flags (if estimated)
        if vitals.bp_systolic is not None and vitals.bp_diastolic is not None:
            sbp = vitals.bp_systolic
            dbp = vitals.bp_diastolic
            if sbp < 90 or dbp < 60:
                alerts.append("Estimated BP low (hypotension)")
                score += 2
            if sbp > 180 or dbp > 110:
                alerts.append("Estimated BP very high (hypertensive crisis)")
                score += 4
            elif sbp > 160 or dbp > 100:
                alerts.append("Estimated BP high (hypertension)")
                score += 2

        # SpO2 flags (if available)
        if vitals.spo2 is not None:
            if vitals.spo2 < 90:
                alerts.append("Low SpO₂ (hypoxemia)")
                score += 4
            elif vitals.spo2 < 95:
                alerts.append("Borderline SpO₂")
                score += 2

        # Stress flags (secondary)
        if vitals.stress_level is not None and vitals.stress_level > 8:
            alerts.append("High stress level (low HRV)")
            score += 1
        
        # Determine risk level
        # Determine risk level and recommendation
        if score >= 7:
            risk_level = "HIGH"
            recommendation = (
                "⚠️  HIGH RISK INDICATORS (experimental assessment only). "
                "Consider urgent clinical evaluation. This is a research prototype — not a diagnostic tool."
            )
        elif score >= 3:
            risk_level = "MODERATE"
            recommendation = (
                "⚠️  MODERATE RISK INDICATORS. "
                "Follow up with a healthcare professional if symptoms or concerns persist."
            )
        else:
            risk_level = "LOW"
            recommendation = (
                "✓ Low risk indicators (experimental assessment). "
                "Vitals appear within typical ranges for this measurement."
            )
        
        return RiskAssessment(
            risk_score=score,
            risk_level=risk_level,
            alerts=alerts,
            recommendation=recommendation
        )


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def estimate_vitals_from_video(
    video_path: str,
    use_mediapipe: bool = True,
    progress_callback=None
) -> Tuple[VitalsEstimate, FilteredSignal, RiskAssessment]:
    """
    Full pipeline: video → vitals.
    
    Args:
        video_path: path to video file
        use_mediapipe: use MediaPipe for face detection (if available)
        progress_callback: optional function(current, total) for UI
        
    Returns:
        (VitalsEstimate, FilteredSignal, RiskAssessment)
    """
    # Initialize
    detector = FaceDetector(use_mediapipe=use_mediapipe)
    extractor = SignalExtractor(detector)
    
    try:
        # Extract signals
        signal_data = extractor.extract(video_path, progress_callback)

        # Process signals
        filtered_signal = SignalProcessor.process(signal_data)

        # Estimate vitals (pass signal_data so experimental estimators can access raw channels)
        vitals = VitalsEstimator.estimate(filtered_signal, signal_data)
        
        # Risk assessment
        risk = RiskAssessor.assess(vitals)
        
        return vitals, filtered_signal, risk
    
    finally:
        detector.close()


if __name__ == "__main__":
    print("rPPG system refactored. See rppg_streamlit_ui.py for UI, rppg_fastapi.py for API.")
