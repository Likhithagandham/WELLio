/**
 * WELLio Pure Front-end Camera Module
 * 
 * This module handles:
 * 1. getUserMedia (Camera Access)
 * 2. MediaRecorder (Video Recording)
 * 3. Local browser-side playback/save
 * 4. Multipart/form-data upload to backend
 */

class HealthCamera {
    constructor() {
        // UI Elements - Video
        this.preview = document.getElementById('preview');
        this.resultVideo = document.getElementById('recording-result');
        this.overlay = document.getElementById('video-overlay');
        this.overlayMsg = document.getElementById('overlay-message');
        this.recIndicator = document.getElementById('rec-indicator');
        this.recTimer = document.getElementById('rec-timer');

        // UI Elements - Controls
        this.setupGroup = document.getElementById('setup-group');
        this.recordGroup = document.getElementById('record-group');
        this.resultGroup = document.getElementById('result-group');

        // UI Elements - Status
        this.statusBar = document.getElementById('status-bar');
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        this.progressWrapper = document.getElementById('progress-container');
        this.progressFill = document.getElementById('progress-fill');

        // Buttons
        this.btns = {
            startCamera: document.getElementById('start-camera-btn'),
            startRec: document.getElementById('start-rec-btn'),
            stopRec: document.getElementById('stop-rec-btn'),
            save: document.getElementById('save-btn'),
            upload: document.getElementById('upload-btn'),
            reset: document.getElementById('reset-btn')
        };

        // State
        this.stream = null;
        this.mediaRecorder = null;
        this.chunks = [];
        this.recordedBlob = null;
        this.timer = null;
        this.secondsElapsed = 0;
        this.MAX_DURATION = 30; // seconds

        this.initEventListeners();
    }

    initEventListeners() {
        this.btns.startCamera.addEventListener('click', () => this.startCamera());
        this.btns.startRec.addEventListener('click', () => this.startRecording());
        this.btns.stopRec.addEventListener('click', () => this.stopRecording());
        this.btns.save.addEventListener('click', () => this.saveLocally());
        this.btns.upload.addEventListener('click', () => this.uploadToBackend());
        this.btns.reset.addEventListener('click', () => this.resetApp());
    }

    // --- 1. Camera Management ---

    async startCamera() {
        this.updateStatus('blue', 'Requesting Access...');
        this.btns.startCamera.disabled = true;

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                },
                audio: false
            });

            this.preview.srcObject = this.stream;
            this.updateStatus('green', 'Camera Active');
            this.setUIState('record');
            this.overlayMsg.innerText = 'Position your face in the center';
        } catch (err) {
            console.error("Camera Error:", err);
            this.handleCameraError(err);
        }
    }

    handleCameraError(err) {
        let msg = "Camera initialization failed.";
        if (err.name === 'NotAllowedError') msg = "Permission Denied. Please allow camera access.";
        else if (err.name === 'NotFoundError') msg = "No camera found on this device.";
        else if (err.name === 'NotReadableError') msg = "Camera is already in use by another app.";

        this.updateStatus('red', 'Error');
        this.overlayMsg.innerText = msg;
        this.btns.startCamera.disabled = false;
    }

    // --- 2. Recording Management ---

    startRecording() {
        if (!this.stream) return;

        this.chunks = [];
        const mimeType = this.getSupportedMimeType();

        try {
            this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
        } catch (e) {
            alert("This browser does not support MediaRecorder.");
            return;
        }

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.chunks.push(e.data);
        };

        this.mediaRecorder.onstop = () => {
            this.onRecordingComplete();
        };

        this.mediaRecorder.start();
        this.setRecordingUI(true);
        this.startTimer();
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }

    onRecordingComplete() {
        this.setRecordingUI(false);
        this.stopTimer();

        const mimeType = this.chunks[0].type;
        this.recordedBlob = new Blob(this.chunks, { type: mimeType });

        // Show result preview
        const url = URL.createObjectURL(this.recordedBlob);
        this.resultVideo.src = url;
        this.resultVideo.classList.remove('hidden');
        this.preview.classList.add('hidden');

        this.overlayMsg.classList.add('hidden');
        this.setUIState('result');
        this.updateStatus('blue', 'Recording Saved in Memory');

        // Streamlit Integration
        if (window.Streamlit) {
            const reader = new FileReader();
            reader.readAsDataURL(this.recordedBlob);
            reader.onloadend = () => {
                Streamlit.setComponentValue({
                    status: "success",
                    data: reader.result,
                    type: mimeType
                });
            };
        }
    }

    // --- 3. UI State Management ---

    setUIState(state) {
        this.setupGroup.classList.toggle('hidden', state !== 'setup');
        this.recordGroup.classList.toggle('hidden', state !== 'record');
        this.resultGroup.classList.toggle('hidden', state !== 'result');

        // Notify Streamlit that our height might have changed
        if (window.Streamlit) {
            setTimeout(() => {
                Streamlit.setFrameHeight(document.body.scrollHeight);
            }, 50);
        }
    }

    updateStatus(color, text) {
        this.statusDot.className = `status-dot ${color}`;
        this.statusText.innerText = text;
    }

    setRecordingUI(isRecording) {
        this.btns.startRec.disabled = isRecording;
        this.btns.stopRec.disabled = !isRecording;
        this.recIndicator.classList.toggle('hidden', !isRecording);
        this.updateStatus(isRecording ? 'red' : 'green', isRecording ? 'Recording...' : 'Camera Active');

        if (isRecording) {
            this.overlayMsg.innerText = 'Scanning specialized metrics...';
        }
    }

    // --- 4. Timer Management ---

    startTimer() {
        this.secondsElapsed = 0;
        this.recTimer.innerText = '00:00';

        this.timer = setInterval(() => {
            this.secondsElapsed++;
            const mins = Math.floor(this.secondsElapsed / 60).toString().padStart(2, '0');
            const secs = (this.secondsElapsed % 60).toString().padStart(2, '0');
            this.recTimer.innerText = `${mins}:${secs}`;

            if (this.secondsElapsed >= this.MAX_DURATION) {
                this.stopRecording();
            }
        }, 1000);
    }

    stopTimer() {
        clearInterval(this.timer);
    }

    // --- 5. Export / Reset Functions ---

    saveLocally() {
        if (!this.recordedBlob) return;
        const link = document.createElement('a');
        link.href = URL.createObjectURL(this.recordedBlob);
        link.download = `WELLio_HealthScan_${Date.now()}.webm`;
        link.click();
    }

    async uploadToBackend() {
        if (!this.recordedBlob) return;

        this.progressWrapper.classList.remove('hidden');
        this.updateProgress(10);
        this.updateStatus('blue', 'Uploading to Backend...');
        this.btns.upload.disabled = true;

        const formData = new FormData();
        formData.append('video', this.recordedBlob, 'scan.webm');

        try {
            // Simulated fake progress
            let p = 20;
            const simInterval = setInterval(() => {
                p += 5;
                if (p < 90) this.updateProgress(p);
            }, 300);

            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });

            clearInterval(simInterval);

            if (response.ok) {
                this.updateProgress(100);
                this.updateStatus('green', 'Upload Success!');
                alert("Scan result uploaded successfully. The backend is now processing your vitals.");
            } else {
                throw new Error(`Server responded with ${response.status}`);
            }
        } catch (err) {
            console.error("Upload Error:", err);
            this.updateStatus('red', 'Upload Failed');
            alert(`Failed to send recording: ${err.message}`);
        } finally {
            this.btns.upload.disabled = false;
            setTimeout(() => {
                this.progressWrapper.classList.add('hidden');
                this.updateProgress(0);
            }, 3000);
        }
    }

    updateProgress(percent) {
        this.progressFill.style.width = `${percent}%`;
    }

    resetApp() {
        // Stop any current video objects
        this.preview.srcObject = null;
        this.resultVideo.src = "";

        // Hide result, show preview
        this.resultVideo.classList.add('hidden');
        this.preview.classList.remove('hidden');
        this.overlayMsg.classList.remove('hidden');

        // Reset state
        this.recordedBlob = null;
        this.chunks = [];
        this.setUIState('record'); // Back to record screen since we still have the stream? 
        // Or actually full reset?

        // If we want to kill the camera stream on reset:
        /*
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.setUIState('setup');
            this.btns.startCamera.disabled = false;
        }
        */
        this.updateStatus('green', 'Ready');
    }

    getSupportedMimeType() {
        const types = [
            'video/webm;codecs=vp8,opus',
            'video/webm;codecs=vp9,opus',
            'video/webm',
            'video/mp4'
        ];
        return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
    }
}

// Ensure the JS is stable before instantiating
document.addEventListener('DOMContentLoaded', () => {
    window.app = new HealthCamera();
    console.log("WELLio Pure Camera Module Initialized.");
});

// Security: Detect if user tries to refresh during recording
window.onbeforeunload = function () {
    if (window.app && window.app.mediaRecorder && window.app.mediaRecorder.state === 'recording') {
        return "Recording in progress. Are you sure you want to leave?";
    }
};
