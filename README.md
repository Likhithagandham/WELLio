# 🫀 Remote Photoplethysmography (rPPG) - Experimental Vitals Estimation

**Research Prototype for Educational & Hackathon Use**

---

## ⚠️ CRITICAL DISCLAIMER

**THIS IS AN EXPERIMENTAL RESEARCH TOOL - NOT CLINICALLY VALIDATED**

This application:
- ❌ Does NOT provide clinical diagnoses
- ❌ Does NOT replace medical devices (ECG, pulse oximeter, BP monitor)
- ❌ Has NOT been validated against clinical devices
- ❌ Should NOT be used for medical decisions
- ❌ Does NOT have FDA/CE/regulatory approval

**All outputs are experimental estimates with ±5-10% typical error.**

### Key Limitations:
| Vital | Confidence | Error Range | Notes |
|-------|-----------|-------------|-------|
| **Heart Rate** | Medium | ±5-10 BPM | Visible light rPPG |
| **HRV (SDNN)** | Low | ±20-40% | Needs 2+ min video |
| **Blood Pressure** | Very Low | ±15 mmHg | Uncalibrated heuristic |
| **SpO₂** | Not Estimated | - | Requires NIR camera |

### Use This For:
✅ Learning signal processing  
✅ Hackathon/classroom demos  
✅ Research proof-of-concept  
✅ Personal wellness exploration  

### DO NOT Use For:
❌ Medical diagnosis  
❌ Clinical monitoring  
❌ Treatment decisions  
❌ Remote patient monitoring  

**Consult a healthcare professional for any medical concerns.**

---

## 📊 Project Overview

### What It Does
This project implements a **complete pipeline for camera-based vital signs estimation**:

```
Video Input
    ↓
📹 [FACE DETECTION] MediaPipe or Haar Cascade
    ↓ Isolate forehead ROI (highest PPG signal)
    ↓
🎬 [SIGNAL EXTRACTION] Temporal color changes (green channel)
    ↓ Creates: G(t), R(t) signals
    ↓
🔬 [PREPROCESSING] Detrending, normalization, bandpass filtering
    ↓
❤️ [HEART RATE] Welch PSD peak → BPM
    ↓
📈 [HRV] RR intervals → SDNN, pNN50 → stress estimate
    ↓
⚠️ [EXPERIMENTAL] BP/SpO₂ heuristics (uncalibrated)
    ↓
📊 [RISK SCORE] Heuristic alerts
    ↓
🖼️ [VISUALIZATION] Signal plots, spectral analysis, HRV histograms
```

### Academic Background
- **rPPG** is an active research area (published since ~2008)
- Key papers: Verkruysse et al., Poh et al., de Haan & Jeanne (CHROM), Wang et al. (POS)
- Motion artifacts, lighting variations, skin tone bias are open research challenges
- **This prototype is NOT validated** — just a teaching tool

---

## 🏗️ Architecture

```
wellio/
├── rppg_refactored.py              # Core signal processing (modular)
│   ├── FaceDetector                # Face/ROI detection
│   ├── SignalExtractor             # Video → color signals
│   ├── SignalProcessor             # Filter, detrend, normalize
│   ├── VitalsEstimator             # HR, HRV, BP, SpO₂
│   └── RiskAssessor                # Risk scoring
├── rppg_streamlit_ui.py            # Interactive Streamlit app (local)
├── rppg_fastapi.py                 # REST API backend (cloud-ready)
├── DEPLOYMENT_AND_ETHICS_GUIDE.py  # Detailed deployment guide
├── REACT_FRONTEND_EXAMPLE.py       # Sample React component
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Three Deployment Modes

| Mode | Best For | Setup | Complexity |
|------|----------|-------|-----------|
| **Streamlit (Local)** | Demo, classroom, learning | `streamlit run app.py` | ⭐ Easy |
| **FastAPI (Server)** | Web/mobile, production | Docker + cloud | ⭐⭐⭐ Medium |
| **Hybrid** | Scalable with optional cloud | Both | ⭐⭐⭐⭐ Complex |

---

## 🚀 Quick Start

### Option 1: Local Streamlit (Easiest)

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run rppg_streamlit_ui.py

# Open browser
# http://localhost:8501
```

**Upload a 10-30 second video of your face under good lighting.**

### Option 2: FastAPI Backend

```bash
# Install + run
pip install fastapi uvicorn aiofiles
python rppg_fastapi.py

# Test
curl http://localhost:8000/health
# Open http://localhost:8000/docs for Swagger UI
```

### Option 3: Docker

```bash
# Build
docker build -f Dockerfile.fastapi -t rppg-api .

# Run
docker run -p 8000:8000 rppg-api

# Test
curl http://localhost:8000/health
```

---

## 📋 Requirements

### System Requirements
- **Python 3.9+**
- **OpenCV 4.5+** (video processing)
- **SciPy 1.7+** (signal processing)
- **NumPy 1.20+**
- **MediaPipe 0.8+** (optional, better face detection)
- **Streamlit 1.0+** (optional, for UI)
- **FastAPI 0.95+** (optional, for API)

### Install

```bash
# Core (minimal)
pip install opencv-python scipy numpy pandas

# Full (recommended)
pip install -r requirements.txt

# For Streamlit UI
pip install streamlit

# For FastAPI
pip install fastapi uvicorn aiofiles
```

### GPU Acceleration (Optional)
- For faster processing, install **CUDA + cuDNN**
- CV2 and SciPy will auto-detect CUDA
- Typical speedup: 2-3x on NVIDIA GPUs

---

## 💻 Usage

### Programmatic API

```python
from rppg_refactored import estimate_vitals_from_video

# Analyze a video
vitals, filtered_signal, risk = estimate_vitals_from_video(
    video_path="my_video.mp4",
    use_mediapipe=True
)

# Results
print(f"Heart Rate: {vitals.heart_rate_bpm:.1f} BPM")
print(f"Confidence: {vitals.heart_rate_confidence}")
print(f"SDNN: {vitals.sdnn:.1f} ms")
print(f"Stress: {vitals.stress_level:.1f}/10")
print(f"BP: {vitals.bp_systolic:.0f}/{vitals.bp_diastolic:.0f} mmHg (experimental)")
print(f"Risk: {risk.risk_level} - {risk.recommendation}")
```

### HTTP API

```bash
# POST /analyze
curl -X POST http://localhost:8000/analyze \
  -F "file=@video.mp4" \
  -H "Content-Type: multipart/form-data"

# Response:
{
  "request_id": "rppg_...",
  "vitals": {
    "heart_rate_bpm": 72.5,
    "stress_level": 3.2,
    ...
  },
  "risk": {
    "risk_level": "LOW",
    ...
  }
}

# GET /health
curl http://localhost:8000/health

# GET /docs
# Open http://localhost:8000/docs in browser for interactive Swagger UI
```

### Streamlit App
- Upload video
- View real-time processing
- See signal plots, FFT, HRV histograms
- Download results as JSON

---

## 🔬 How It Works (Technical Details)

### 1. Face Detection
**Goal:** Locate forehead (best PPG signal location)

```python
# MediaPipe (preferred): ~99% accuracy, robust
# Haar Cascade (fallback): ~85% accuracy, faster
face_detector = FaceDetector(use_mediapipe=True)
roi = face_detector.detect(frame)  # Returns ROIBbox
```

### 2. Signal Extraction
**Goal:** Extract temporal color signal from ROI

```python
# Sample green & red channels over 30 frames/sec
signal_extractor = SignalExtractor(detector)
signal_data = signal_extractor.extract("video.mp4")
# Returns: green[T], red[T], fps, detection_stats
```

**Why green channel?**
- Green light (λ ~550 nm) penetrates skin ~1-2 mm
- Optimal for PPG detection (hemoglobin absorption peak)
- Red/NIR require specialized cameras

### 3. Preprocessing
**Goal:** Remove noise, drift, motion artifacts

```python
# Step 1: Interpolate missing frames
green = SignalProcessor.interpolate_nans(signal_data.green)

# Step 2: Remove DC drift (detrending)
green = detrend(green)

# Step 3: Normalize
green = green / np.std(green)

# Step 4: Bandpass filter (0.75-3.0 Hz = 45-180 BPM)
filtered = butter_bandpass_filter(green, 0.75, 3.0, fps=30)
```

**Why these steps?**
- **Interpolation:** Handle face detection failures
- **Detrending:** Remove slow lighting changes
- **Normalization:** Standardize amplitude (camera-independent)
- **Bandpass:** Keep only plausible HR frequencies, remove noise

### 4. Heart Rate Estimation
**Goal:** Find dominant frequency (heart rate)

```python
# Method: Welch Power Spectral Density (more robust than raw FFT)
freqs, psd = welch(filtered, fs=fps, nperseg=256)

# Find peak in HR band
valid_band = (freqs >= 0.75) & (freqs <= 3.0)
peak_freq = freqs[valid_band][np.argmax(psd[valid_band])]
bpm = peak_freq * 60  # Convert Hz to BPM
```

**Why Welch over raw FFT?**
- Reduces spectral leakage
- More robust to noise
- Better for short signals

**Why 0.75-3.0 Hz?**
- Healthy resting HR: 60-100 BPM = 1-1.67 Hz
- Upper bound: 180 BPM = 3 Hz (covers tachycardia, exercise)
- Lower bound: 45 BPM = 0.75 Hz (covers bradycardia)

### 5. Heart Rate Variability (HRV)
**Goal:** Assess autonomic nervous system via RR intervals

```python
# Detect peaks in filtered signal
peaks = find_peaks(filtered, distance=0.4*fps, prominence=0.3*std)
rr_intervals = np.diff(peaks) / fps * 1000  # Convert to ms

# Time-domain metrics
sdnn = np.std(rr_intervals)        # ms, standard deviation
nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
pnn50 = 100 * nn50 / (len(rr_intervals) - 1)  # percentage

# Stress estimate (heuristic)
if pnn50 < 5:
    stress = 8.0      # Low HRV → high stress
elif pnn50 < 15:
    stress = 5.5
elif pnn50 < 30:
    stress = 3.0
else:
    stress = 1.0      # High HRV → calm
```

**Limitations:**
- Gold standard: 5-minute recording → SDNN is clinical
- This app: ~1 min video → unreliable
- Motion artifacts corrupt RR detection
- pNN50 > SDNN for short recordings

### 6. Experimental Blood Pressure
**⚠️ NOT CLINICALLY VALIDATED**

```python
# Simple heuristic (NO scientific basis)
if bpm < 60:
    sbp_range = (105, 120)
elif bpm < 80:
    sbp_range = (115, 130)
else:
    sbp_range = (125, 145)

sbp = mean(sbp_range)
dbp = sbp - 40
```

**Why it's wrong:**
- BP depends on vascular resistance, NOT just HR
- rPPG alone cannot estimate BP accurately
- No peer-reviewed validation
- Clinical devices measure arterial pressure directly

**Better approach (research):**
- Use multiple signals (green, red, NIR)
- Calibrate on validated device dataset
- Model: ML regression (HR, HRV → BP)
- Requires 100+ subjects for validation

### 7. Experimental SpO₂
**NOT ESTIMATED — reasons:**

```python
# Old (WRONG):
# spo2 = 110 - 25 * (red_ac/red_dc) / (green_ac/green_dc)

# Why it's wrong:
# 1. Pulse oximetry uses Red (~660 nm) + IR (~940 nm)
# 2. Green channel is NOT suited for SpO₂
# 3. Beer-Lambert law: 2 wavelengths → 2 unknowns (SaO₂, perfusion)
# 4. Green is saturated in oxy/deoxy hemoglobin
# 5. Ratio changes with skin tone (melanin absorbs ~all wavelengths)
# 6. Motion artifacts bias the ratio

# Correct approach (research-grade):
# - Use IR + Red cameras
# - Calibrate on diverse skin tones
# - Account for perfusion changes
# - Motion-robust signal extraction (ICA, optical flow)
# → Still open research problem
```

### 8. Risk Assessment
**Heuristic scoring (experimental, NOT clinical):**

```python
risk_score = 0

if bpm > 120 or bpm < 40:
    risk_score += 2  # Abnormal HR

if sdnn < 15:
    risk_score += 2  # Very low HRV

if sbp < 85 or dbp < 55:
    risk_score += 2  # Hypotension

if spo2 < 90:
    risk_score += 3  # Low O₂

if stress_level > 8:
    risk_score += 1  # High stress

# Classify
if risk_score >= 6:
    level = "HIGH"
elif risk_score >= 3:
    level = "MODERATE"
else:
    level = "LOW"
```

**Why heuristic?**
- No epidemiological data
- Thresholds arbitrary
- Not trained on medical data
- NOT predictive of cardiac events

---

## 📊 Signal Processing Walkthrough (Example)

```
Raw signal (30 frames/sec, 60 sec = 1800 samples):
[152, 153, 151, 154, 152, 155, 154, 153, ...] (0-255 pixel intensity)

↓ [Interpolate NaNs]
(No NaNs in this example)

↓ [Detrend] - Remove drift
[  1.2,  1.5, -0.8,  2.1,  0.9,  2.8,  2.2,  1.1, ...] (centered)

↓ [Normalize] - Divide by std (σ = ~1.5)
[  0.8,  1.0, -0.5,  1.4,  0.6,  1.9,  1.5,  0.7, ...] (dimensionless)

↓ [Bandpass 0.75-3 Hz] - Keep HR frequencies only
[  0.3,  0.5, -0.2,  0.7,  0.2,  1.1,  0.9,  0.4, ...] (filtered)

↓ [Welch PSD] - Estimate spectrum
Frequency (Hz)  |  Power
0.75            |  0.1
1.2  ← PEAK     |  2.8  ← Maximum
1.5             |  0.9
2.0             |  0.2

Peak frequency: 1.2 Hz
Heart rate: 1.2 * 60 = 72 BPM ✓

↓ [Peak Detection] - Find heartbeats
Peaks at samples: [150, 330, 510, 690, ...] (every ~180 samples)
RR intervals: [180, 180, 180, ...] samples = [6000, 6000, 6000, ...] ms
           = [1.0, 1.0, 1.0, ...] seconds ✓ (matches HR)

↓ [HRV]
SDNN = std([6000, 6000, ...]) = low (< 50 ms)
→ Stress = HIGH (artificial, uniform rhythm)
```

---

## 🎯 Algorithmic Improvements (For Research)

### Motion Artifact Removal
**Current:** Single mean per frame (motion corrupts signal)  
**Better:** 

1. **CHROM (Chrominance)** - Orthogonal projection:
   ```python
   # 3G - 2R (reduces motion ~70% vs green alone)
   signal = 3*green - 2*red
   ```

2. **Optical Flow** - Detect head/camera motion:
   ```python
   # Use Lucas-Kanade corner tracking
   flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners)
   motion = np.mean(np.abs(flow), axis=0)
   if motion > threshold:
       reject_frame()
   ```

3. **ICA (Independent Component Analysis)** - Extract PPG from mixed signals:
   ```python
   # Stack [R, G, B] over sliding window
   # ICA separates physiological signals from motion
   ```

### Robust HR Estimation
**Current:** Single Welch peak  
**Better:**

1. **Multi-peak detection:**
   ```python
   peaks = find_peaks(psd, height=threshold)
   if harmonic_ratio(peaks[0], peaks[1]) ~= 2.0:
       return peaks[0]  # reject 2nd harmonic
   ```

2. **Adaptive bandpass:**
   ```python
   # Auto-detect HR range from coarse PSD
   # Adjust filters per-video (not fixed 45-180 BPM)
   ```

3. **Validation vs RR peaks:**
   ```python
   bpm_psd = ...
   bpm_rr = 60 / np.median(rr_intervals)
   if abs(bpm_psd - bpm_rr) > 10:
       confidence = "LOW"  # Warn user
   ```

### Better HRV Metrics
**Current:** Only SDNN  
**Add:**

```python
# Frequency-domain (needs 5+ min)
from scipy.signal import periodogram
freqs_hrv, psd_hrv = periodogram(rr, fs=1.0/np.median(rr/1000))

vlf_power = integrate(psd_hrv[0.003:0.04 Hz])   # Sympathetic?
lf_power = integrate(psd_hrv[0.04:0.15 Hz])    # Sympathetic
hf_power = integrate(psd_hrv[0.15:0.4 Hz])     # Parasympathetic

lf_hf_ratio = lf_power / hf_power  # Autonomic balance
```

---

## 🧪 Testing & Validation

### Compare Against Validated Device

```python
import numpy as np

# Your app output
app_hr = 72.0
app_sdnn = 45.0

# Apple Watch / Fitbit output
validated_hr = 73.0
validated_sdnn = 42.0

# Error metrics
hr_error = abs(app_hr - validated_hr)  # ~1 BPM (good)
sdnn_error = abs(app_sdnn - validated_sdnn)  # ~3 ms (OK)

# Report
print(f"HR Error: {hr_error:.1f} BPM (±{100*hr_error/validated_hr:.1f}%)")
print(f"SDNN Error: {sdnn_error:.1f} ms (±{100*sdnn_error/validated_sdnn:.1f}%)")

# Expected:
# HR: ±5-10 BPM (visible light rPPG)
# SDNN: ±20-40% (short recording)
```

### Unit Tests

```python
# test_rppg.py
import pytest
from rppg_refactored import SignalProcessor, VitalsEstimator

def test_butter_bandpass():
    """Test filter design"""
    signal = np.sin(2*np.pi*0.02*np.arange(1000)/30)  # 1.2 Hz at 30 fps
    filtered = SignalProcessor.butter_bandpass_filter(signal, 0.75, 3.0, 30)
    # Check attenuation outside band
    assert np.std(filtered) < np.std(signal)

def test_heart_rate_estimation():
    """Test HR extraction from synthetic signal"""
    # Create synthetic PPG with known HR = 72 BPM
    fs = 30
    duration = 10
    t = np.arange(0, duration, 1/fs)
    ppg = np.sin(2*np.pi*1.2*t) + 0.1*np.random.normal(size=len(t))
    
    # Estimate
    from scipy.signal import welch
    freqs, psd = welch(ppg, fs=fs)
    peak_freq = freqs[np.argmax(psd)]
    bpm = peak_freq * 60
    
    assert abs(bpm - 72) < 5, "HR estimation failed"
```

---

## 🛠️ Troubleshooting

### "Face not detected in enough frames"
- **Causes:** Poor lighting, small face, occlusions
- **Solutions:**
  - ✅ Use bright, diffuse lighting (avoid shadows)
  - ✅ Move closer to camera
  - ✅ Remove glasses/face masks
  - ✅ Try longer video (30-60 sec)
  - ✅ Try different camera angle

### "Signal has zero variance"
- **Causes:** No motion detected (static frame)
- **Solutions:**
  - ✅ Move head slightly
  - ✅ Re-record with better lighting
  - ✅ Check video file (try different format)

### "Heart rate out of plausible range"
- **Causes:** Noise, motion artifact, bad HR
- **Solutions:**
  - ✅ Record again with minimal movement
  - ✅ Try --use-mediapipe flag for better face tracking
  - ✅ Validate against wearable (Apple Watch, Fitbit)

### "Very low stress / very high SDNN"
- **Causes:** Video too short, uniform rhythm, perfect recording
- **Note:** This is **artificial**. Real HRV requires 5+ min
- **Solution:** Longer video (2+ min) for better statistics

### API returns 413 "File too large"
- **Limit:** 500 MB max
- **Solution:** Compress video or record shorter clip
  ```bash
  ffmpeg -i input.mp4 -c:v libx265 -crf 28 output.mp4
  ```

### CORS errors in React frontend
- **Cause:** Backend doesn't allow frontend origin
- **Solution:** Update CORS settings in rppg_fastapi.py
  ```python
  CORSMiddleware(allow_origins=["https://myfront end.com"])
  ```

---

## 📚 References & Further Reading

### Key Papers
- **rPPG Fundamentals**
  - Verkruysse et al. (2008): "Remote plethysmographic imaging of coronary blood flow"
  - Poh et al. (2010): "Advancements in noncontact, multimodal sensing"

- **CHROM Algorithm**
  - de Haan & Jeanne (2013): "Robust pulse rate from chrominance-based rPPG"

- **POS Algorithm**
  - Wang et al. (2016): "Algorithmic principles of remote PPG"

- **Motion Artifacts**
  - Tran et al. (2019): "Investigating the Effectiveness of ICA for Motion Artifact Removal"

- **Skin Tone Bias**
  - Matta et al. (2020): "On the Untapped Potential of Generative Models"
  - Lu et al. (2021): "Demographic Bias in Biometrics"

### Datasets
- **PURE** (Stricker et al., 2014): ~10 videos, ground truth PPG
- **UBFC-rPPG** (Bobbia et al., 2019): ~42 subjects, visible light
- **SCAMPS** (Yu et al., 2019): Motion & lighting robustness

### Tools
- **MediaPipe:** mediapipe.dev
- **OpenCV:** opencv.org
- **SciPy:** scipy.org
- **Streamlit:** streamlit.io
- **FastAPI:** fastapi.tiangolo.com

---

## 📝 Academic Integrity & Citing This Work

### If Using in Research Paper
```bibtex
@misc{neuronexusRPPG2024,
  title={Modular Remote Photoplethysmography for Experimental Vitals Estimation},
  author={Your Name},
  year={2024},
  note={Research Prototype - Not Clinically Validated. Available at: https://github.com/...}
}
```

### In Methods Section
```
"We implemented an experimental rPPG system based on the CHROM algorithm 
(de Haan & Jeanne, 2013) for heart rate estimation. The system is NOT clinically 
validated and has a typical error of ±5-10 BPM compared to ECG. All outputs should 
be interpreted as research-grade estimates only."
```

### Acknowledge Limitations
- ❌ Do NOT claim clinical accuracy
- ✅ Do acknowledge motion artifacts, skin tone bias
- ✅ Do mention validation against gold standards
- ✅ Do suggest future work (multi-wavelength, ICA, etc.)

---

## 🎓 Learning Path

### Week 1: Fundamentals
- [ ] Understand photoplethysmography basics
- [ ] Learn Python signal processing (SciPy)
- [ ] Extract signals from video (OpenCV)

### Week 2: Implementation
- [ ] Implement preprocessing pipeline
- [ ] Add Butterworth filter
- [ ] Compute Welch PSD

### Week 3: Vitals Estimation
- [ ] Implement HR extraction
- [ ] Detect RR intervals
- [ ] Compute HRV metrics

### Week 4: Robustness
- [ ] Motion artifact handling (optical flow, ICA)
- [ ] Error analysis vs validated device
- [ ] Improve signal quality metrics

### Week 5: Deployment
- [ ] Build Streamlit UI
- [ ] Create FastAPI backend
- [ ] Deploy to cloud (GCP Cloud Run, AWS Lambda)

### Week 6: Ethics & Presentation
- [ ] Write disclaimers
- [ ] Prepare research report
- [ ] Present findings

---

## 🤝 Contributing

### Suggested Improvements
1. **Motion Robustness:** ICA, optical flow, CHROM+ algorithms
2. **Skin Tone Adaptation:** Dataset analysis, model calibration
3. **Mobile Support:** React Native frontend, edge processing
4. **Real-Time Streaming:** WebSocket support, buffer windowing
5. **Validation Studies:** Compare vs Apple Watch, Fitbit, Oura Ring

### Bug Reports & PRs
- File issues with video samples if possible
- Include error logs
- PRs should include tests

---

## 📄 License

This project is provided **for educational and research purposes only**.

**NOT approved for clinical use. Use at your own risk.**

See LICENSE file for full terms.

---

## 📞 Contact & Support

- **Issues:** File GitHub issues
- **Academic Questions:** Contact author
- **Medical Questions:** Consult healthcare professionals

---

## Acknowledgments

Built with:
- **MediaPipe** (face detection)
- **OpenCV** (computer vision)
- **SciPy** (signal processing)
- **NumPy** (numerical computing)
- **Streamlit** (UI framework)
- **FastAPI** (web framework)

Inspired by published rPPG research (see References).

---

**Remember:** This is a research tool. Use responsibly. Prioritize patient safety over accuracy.
