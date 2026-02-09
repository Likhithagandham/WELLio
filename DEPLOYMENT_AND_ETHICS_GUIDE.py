"""
DEPLOYMENT & ETHICS GUIDE
==========================

How to safely deploy the rPPG system as a research demo / hackathon project.
Includes disclaimers, liability protection, and best practices.
"""

# ============================================================================
# PART 1: MEDICAL & ETHICAL DISCLAIMERS
# ============================================================================

"""
CRITICAL DISCLAIMER (MUST appear prominently everywhere):

⚠️  RESEARCH PROTOTYPE - NOT CLINICALLY VALIDATED

This application is an EXPERIMENTAL RESEARCH TOOL for educational,
hackathon, and research purposes only.

🔴 WHAT THIS APP DOES NOT DO:
- ❌ Diagnose or treat medical conditions
- ❌ Replace medical devices or professional evaluation
- ❌ Provide accurate vital sign measurements
- ❌ Generate clinical-grade data
- ❌ Replace pulse oximeters, BP monitors, or ECG devices

⚠️  KEY LIMITATIONS:
- Heart Rate: Typical error ±5-10 BPM (vs medical gold standard ECG ±2 BPM)
- Blood Pressure: Uncalibrated, ±15 mmHg error (NOT validated on any population)
- SpO₂: Not estimated (visible light rPPG cannot reliably estimate SpO₂)
- Heart Rate Variability: Limited by video length (<2 min typical) vs 5 min gold standard
- All outputs: Experimental, subject to motion artifacts and lighting variation

⚠️  DO NOT:
- Use for medical diagnosis
- Use for treatment decisions
- Use on patients in clinical settings without validation
- Trust outputs without cross-validation against validated devices
- Claim clinical grade accuracy
- Use for remote patient monitoring without supervision

✓ APPROPRIATE USES:
- Educational projects (signal processing, computer vision)
- Hackathon demonstrations
- Research prototyping and proof-of-concept
- Personal wellness tracking (non-medical)
- Comparison studies (vs validated devices)
- Teaching biomedical signal processing

🏥 MEDICAL LIABILITY:
- Users assume all responsibility for use
- Creator provides no warranty or guarantee of accuracy
- Not approved by FDA, CE, or any regulatory body
- Not suitable for clinical use without extensive validation
"""

# ============================================================================
# PART 2: REGULATORY COMPLIANCE
# ============================================================================

"""
REGULATORY LANDSCAPE (as of 2026):

1. FDA (U.S.)
   - Any device claiming to measure vital signs is a medical device
   - Class II devices typically require 510(k) clearance
   - This app: NOT cleared, NOT intended for clinical use
   - Recommendation: Clearly label "research prototype" to avoid regulatory review
   
2. CE Marking (EU)
   - Medical Device Regulation (MDR) / In Vitro Diagnostic Regulation (IVDR)
   - Not CE marked; not intended for clinical use in EU
   
3. Canada, Australia, Japan
   - Similar regulatory frameworks
   - This app: Outside scope (research tool, not medical device)
   
COMPLIANCE STRATEGY FOR RESEARCH PROJECTS:
- Clearly mark as "experimental research prototype"
- Do NOT make clinical claims
- Include explicit disclaimers in all outputs
- Do NOT market as medical device
- Keep for internal research/education only
- If publishing, clearly state limitations and non-clinical nature
"""

# ============================================================================
# PART 3: DEPLOYMENT ARCHITECTURES
# ============================================================================

"""
Architecture 1: LOCAL STREAMLIT APP (Safest for Hackathons)
============================================================

Structure:
  - User runs locally: `streamlit run rppg_streamlit_ui.py`
  - User uploads their own video
  - No server involved
  - Results stay local
  
Advantages:
  ✅ No privacy concerns (data stays local)
  ✅ No regulatory jurisdiction issues
  ✅ No server liability
  ✅ Easy to deploy (one command)
  ✅ Works offline
  
Disadvantages:
  ❌ Poor user experience (need Python installed)
  ❌ No multi-user support
  
Use case: Hackathon judging, classroom demo, research lab


Architecture 2: CLOUD API + WEB/MOBILE FRONTEND
================================================

Stack:
  Backend: FastAPI + Docker + AWS/GCP/Azure
  Frontend: React (web) or React Native (mobile)
  Database: Optional (only for research, not medical records)
  
Components:
  
  1. Backend Deployment (FastAPI)
  ```
  Docker image built from rppg_fastapi.py
  Deployed to AWS EC2/ECS, GCP Cloud Run, or Azure Container Instances
  HTTPS/TLS encryption
  Rate limiting & API keys for access control
  Logging to CloudWatch/Stackdriver
  ```
  
  2. Frontend (React)
  ```
  Single-page app (SPA)
  Video upload → calls /analyze endpoint
  Displays results with disclaimers
  Sample: /examples/rppg_react_frontend.jsx
  ```
  
  3. Database (if needed)
  ```
  AWS RDS / Cloud SQL
  Store anonymized analysis results ONLY
  Do NOT store videos
  Do NOT use for patient records
  GDPR/CCPA compliance required
  ```
  
Deployment Steps:
  
  1. Create Docker image:
     ```bash
     docker build -f Dockerfile.fastapi -t rppg-api:latest .
     ```
  
  2. Push to registry (ECR, GCR, Docker Hub)
  
  3. Deploy to cloud:
     AWS ECS: create task definition, service, load balancer
     GCP: Cloud Run (serverless, cheapest)
     Azure: Container Instances or App Service
  
  4. Set up CORS, rate limiting, logging
  
  5. Frontend points to API endpoint


Architecture 3: HYBRID (Streamlit + Optional Cloud Processing)
==============================================================

Streamlit app with optional cloud processing:
  - Default: local processing
  - Option: "Send to cloud for faster processing"
  - Cloud can use GPU (faster) or batch analysis
  
Advantage: User choice + scalability
"""

# ============================================================================
# PART 4: SAMPLE DOCKER DEPLOYMENT
# ============================================================================

DOCKERFILE_FASTAPI = """
# Dockerfile for rPPG FastAPI
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY rppg_refactored.py .
COPY rppg_fastapi.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run
CMD ["uvicorn", "rppg_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKER_COMPOSE = """
version: '3.8'

services:
  rppg-api:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    labels:
      - "com.example.description=rPPG Vitals API"

  # Optional: React frontend (if building full-stack)
  rppg-web:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - rppg-api
"""

# ============================================================================
# PART 5: DATA PRIVACY & GDPR COMPLIANCE
# ============================================================================

"""
PRIVACY BEST PRACTICES:

1. VIDEO HANDLING:
   ✅ Do NOT store user videos
   ✅ Process and immediately delete
   ✅ Store metadata only (timestamps, file sizes)
   ❌ Never transmit videos to cloud unless encrypted end-to-end
   
2. RESULTS STORAGE:
   ✅ Store anonymized results (random IDs, not names/dates)
   ✅ Remove personally identifiable info (faces, names)
   ✅ If storing, use encrypted database
   ❌ Do NOT share with 3rd parties without consent
   
3. GDPR COMPLIANCE (for EU users):
   ✅ Clear privacy policy
   ✅ Explicit opt-in for processing
   ✅ Right to access: export user data
   ✅ Right to deletion: delete all user data
   ✅ Data retention: delete old results (e.g., after 30 days)
   
4. CCPA (U.S. California):
   ✅ Disclose what data is collected
   ✅ Allow opt-out of data sales (don't sell anyway)
   ✅ Provide download of personal information
   
5. SAMPLE PRIVACY POLICY:
"""

PRIVACY_POLICY = """
PRIVACY POLICY - rPPG Vitals Estimation Demo

1. DATA COLLECTION:
   We collect:
   - Video file (temporarily, for processing only)
   - Analysis results (heart rate, HRV, estimated vital signs)
   - Metadata (timestamp, file size, processing time)
   
   We do NOT collect:
   - Personal identifiers (name, email, phone)
   - Device identifiers
   - Location data
   - Cookies or tracking

2. DATA USAGE:
   - Processing: Run signal analysis on your video
   - Improvement: Aggregate anonymized results to improve algorithms
   - Research: May publish findings (no individual data)

3. DATA RETENTION:
   - Videos: Deleted immediately after processing
   - Results: Cached for 1 hour for quick re-download, then deleted
   - Logs: Kept for 7 days for debugging

4. DATA SECURITY:
   - HTTPS/TLS encryption for all transmissions
   - No database of personal information
   - Results not shared with 3rd parties

5. YOUR RIGHTS:
   - Request your data: contact support
   - Delete your data: automatic after cache expiration
   - Opt-out: don't use the service

6. MEDICAL DISCLAIMER:
   This app is experimental and NOT clinically validated.
   Do NOT use for medical decisions.

7. CONTACT:
   Questions? Email: [your email]
"""

# ============================================================================
# PART 6: SAMPLE TERMS OF SERVICE
# ============================================================================

TERMS_OF_SERVICE = """
TERMS OF SERVICE - rPPG Vitals Demo

1. DISCLAIMER:
   This application is an experimental research tool. All outputs are 
   provided "as-is" without warranty. We make NO claim of accuracy or fitness
   for any particular purpose.

2. NOT MEDICAL ADVICE:
   This app does not provide medical advice, diagnosis, or treatment.
   Do not use for medical decisions. Consult a healthcare professional.

3. LIABILITY LIMITATION:
   In no event shall we be liable for any indirect, incidental, special,
   consequential, or punitive damages arising from your use of this app,
   even if advised of the possibility of such damages.

4. USER RESPONSIBILITY:
   You are responsible for any consequences of using this app.
   You assume all risks.

5. VIDEO PRIVACY:
   You grant us permission to process your video. Videos are deleted
   immediately after processing and not stored.

6. ACCEPTABLE USE:
   - ✅ Educational, research, hackathon use
   - ✅ Personal wellness tracking (non-medical)
   - ❌ Clinical diagnosis
   - ❌ Medical device replacement
   - ❌ Resale or commercial use without license

7. TERMINATION:
   We may revoke access if terms are violated.

8. GOVERNING LAW:
   [Your jurisdiction]
"""

# ============================================================================
# PART 7: INSTRUCTORS GUIDE (for academic use)
# ============================================================================

"""
FOR UNIVERSITY COURSES / STUDENT PROJECTS:

Integration into Curriculum:
  - Signal Processing: Filter design, FFT, Welch PSD
  - Computer Vision: Face detection (MediaPipe, Haar)
  - Biomedical Engineering: PPG, HRV, vital signs
  - ML/AI: Feature extraction, prediction, model deployment
  - Ethics: AI responsibility, disclaimers, regulatory compliance

Assignment Ideas:
  1. Basic: Extract signals and plot them
  2. Intermediate: Implement custom bandpass filter, compare with rPPG
  3. Advanced: Motion artifact removal (ICA, optical flow)
  4. Project: Compare against validated device (Apple Watch, Fitbit)
  5. Research: Skin tone adaptation, multi-wavelength simulation

Grading Rubric:
  [✅] Code quality & documentation
  [✅] Signal processing correctness
  [✅] Experimental methodology (if comparison study)
  [✅] Ethical awareness & disclaimers
  [✅] Presentation & visualization
  [❌] Clinical accuracy (NOT expected, not graded)

Learning Outcomes:
  - Understand rPPG signal acquisition and processing
  - Implement digital filtering and spectral analysis
  - Apply face detection and ROI extraction
  - Quantify uncertainty and limitations
  - Practice responsible AI/ML development

Preventing Misuse:
  - Explicitly grade on ethical statements
  - Require disclaimers in all reports
  - Prohibit clinical claims
  - Encourage comparison with validated devices
"""

# ============================================================================
# PART 8: DEPLOYMENT CHECKLIST
# ============================================================================

DEPLOYMENT_CHECKLIST = """
BEFORE DEPLOYING TO PRODUCTION:

Code & Testing:
  [ ] All unit tests pass
  [ ] Load test: 100 concurrent requests
  [ ] Error handling for all edge cases (bad videos, NaNs, etc.)
  [ ] Logging covers all critical paths
  [ ] No hardcoded secrets/credentials

Security:
  [ ] HTTPS/TLS enabled
  [ ] CORS properly configured (not "*")
  [ ] Rate limiting enabled (100 req/min per IP)
  [ ] API key authentication (if needed)
  [ ] Input validation (file size, format, name)
  [ ] No SQL injection / path traversal vulnerabilities
  [ ] Dependency versions pinned (requirements.txt)
  [ ] Security scan (OWASP, Snyk)

Disclaimers & Legal:
  [ ] Privacy policy posted
  [ ] Terms of service posted
  [ ] Medical disclaimer on every page
  [ ] Clear "EXPERIMENTAL RESEARCH PROTOTYPE" labeling
  [ ] Legal review (if not academic)
  [ ] Compliance check (FDA, GDPR, CCPA, etc.)

Operations:
  [ ] Logging to centralized system (CloudWatch, etc.)
  [ ] Monitoring (uptime, latency, errors)
  [ ] Alerting configured (if issues)
  [ ] Backup strategy
  [ ] Disaster recovery plan
  [ ] On-call support process

Infrastructure:
  [ ] Multi-region deployment (if high traffic)
  [ ] CDN for frontend assets
  [ ] Database replication
  [ ] Load balancer configured
  [ ] Auto-scaling rules
  [ ] Database backups (automated)

Documentation:
  [ ] API docs (auto-generated by FastAPI /docs)
  [ ] Deployment guide
  [ ] Troubleshooting guide
  [ ] Architecture diagram
  [ ] Contact information

Post-Deployment:
  [ ] Monitor for errors & crashes (first week)
  [ ] Gather user feedback
  [ ] Iterate on disclaimers based on user misconceptions
  [ ] Track metrics (usage, errors, performance)
"""

# ============================================================================
# PART 9: QUICK START SCRIPTS
# ============================================================================

"""
QUICK START 1: LOCAL STREAMLIT

1. Install dependencies:
   pip install streamlit opencv-python scipy mediapipe numpy pandas matplotlib

2. Run app:
   streamlit run rppg_streamlit_ui.py

3. Open browser:
   http://localhost:8501


QUICK START 2: LOCAL FASTAPI

1. Install dependencies:
   pip install fastapi uvicorn aiofiles

2. Run server:
   python rppg_fastapi.py
   # or: uvicorn rppg_fastapi:app --reload

3. Test API:
   curl -X POST http://localhost:8000/health

4. Swagger UI:
   http://localhost:8000/docs


QUICK START 3: DOCKER

1. Build image:
   docker build -f Dockerfile.fastapi -t rppg-api:latest .

2. Run container:
   docker run -p 8000:8000 rppg-api:latest

3. Access API:
   http://localhost:8000/docs


QUICK START 4: DOCKER COMPOSE (Full Stack)

1. Create docker-compose.yml (see above)

2. Run:
   docker-compose up

3. Access:
   API: http://localhost:8000
   Frontend: http://localhost:3000 (if included)
"""

# ============================================================================
# FINAL CHECKLIST FOR STUDENT PROJECTS
# ============================================================================

"""
✅ CHECKLIST BEFORE SUBMISSION (Hackathons, Universities, Papers)

1. DISCLAIMERS:
   [ ] "EXPERIMENTAL RESEARCH PROTOTYPE" appears prominently
   [ ] "NOT clinically validated" stated clearly
   [ ] "NOT a replacement for medical devices"
   [ ] Medical liability disclaimer included
   
2. CODE:
   [ ] Clean, well-documented, follows PEP 8
   [ ] Error handling for edge cases
   [ ] Tests included (if required)
   [ ] Reproducible (include requirements.txt, instructions)
   
3. REPORT / PAPER:
   [ ] Clearly state this is proof-of-concept, not production
   [ ] Compare against validated devices (if possible)
   [ ] Acknowledge limitations and sources of error
   [ ] Cite rPPG literature
   [ ] Discuss ethical implications
   [ ] Recommend further validation work
   
4. PRESENTATION:
   [ ] Show signal processing steps
   [ ] Visualize filtering effects
   [ ] Discuss uncertainties
   [ ] Avoid overstating accuracy
   [ ] Include judges/audience in limitations
   
5. FOLLOW-UP RESEARCH:
   [ ] Suggest improvements (motion robustness, multi-wavelength, etc.)
   [ ] Recommend validation studies
   [ ] Discuss deployment challenges
   [ ] Address privacy & ethical concerns

GOOD OPENING STATEMENT:
"This is an experimental signal processing project exploring rPPG vitals 
estimation. It's a proof-of-concept for learning purposes, not a clinical 
device. Results have ±5-10% error compared to validated devices and should 
not be used for medical decisions."

AVOID:
"This app measures your heart rate accurately"
"Clinically validated"
"Better than commercial devices"
"Use for health monitoring"
"""

print(__doc__)
