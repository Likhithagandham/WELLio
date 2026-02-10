"""
Centralized Translation Module for Wellio
==========================================

Provides multilingual support for English, Hindi, and Telugu.
Includes static UI translations and dynamic AI content translation.
"""

from groq import Groq
from typing import Optional

# Language configuration
LANGUAGES = {
    "en": {"name": "English", "flag": "🇬🇧"},
    "hi": {"name": "हिंदी", "flag": "🇮🇳"},
    "te": {"name": "తెలుగు", "flag": "🇮🇳"}
}

# Comprehensive translation dictionaries
TRANSLATIONS = {
    "en": {
        # App & Navigation
        "app_title": "Wellio",
        "app_subtitle": "rPPG Vitals Estimation",
        "page_title": "Experimental rPPG Vitals Demo",
        
        # Authentication
        "login_title": "Wellio - Login",
        "login_subtitle": "Sign in to access your health dashboard",
        "signup_title": "Wellio - Sign Up",
        "signup_subtitle": "Create your account to start monitoring your health",
        "email_label": "Email",
        "email_placeholder": "your.email@example.com",
        "password_label": "Password",
        "password_confirm_label": "Confirm Password",
        "password_help": "At least 8 characters, with uppercase, lowercase, and number",
        "name_label": "Full Name",
        "name_placeholder": "John Doe",
        "login_button": "Login",
        "signup_button": "Sign Up",
        "create_account_button": "Create Account",
        "back_to_login": "Back to Login",
        "logout_button": "Logout",
        "login_success": "Login successful!",
        "signup_success": "Account created successfully! Please login.",
        "new_to_wellio": "New to Wellio? Click 'Sign Up' to create your account and start monitoring your health!",
        "data_secure": "Your data is secure: Passwords are encrypted using industry-standard bcrypt hashing.",
        
        # Password Strength
        "password_strength": "Password Strength",
        "strength_weak": "Weak",
        "strength_medium": "Medium",
        "strength_strong": "Strong",
        
        # Validation Messages
        "fill_all_fields": "Please fill in all fields",
        "passwords_no_match": "Passwords do not match",
        "enter_email_password": "Please enter both email and password",
        
        # Sidebar
        "settings_title": "Settings",
        "user_profile_title": "User Profile",
        "history_title": "History",
        "total_sessions": "Total sessions",
        "recent_analyses": "Recent analyses:",
        "no_history": "No history yet. Complete an analysis to get started!",
        "view_trends_button": "View Trend Analysis",
        "show_advanced_plots": "Show advanced signal plots",
        
        # Language Selector
        "language_label": "Language",
        
        # Profile Form
        "age_label": "Age",
        "gender_label": "Gender",
        "height_label": "Height (cm)",
        "weight_label": "Weight (kg)",
        "diet_label": "Diet",
        "exercise_label": "Exercise frequency",
        "sleep_label": "Sleep hours (per night)",
        "smoking_label": "Smoking habits",
        "drinking_label": "Drinking habits",
        "save_profile_button": "Save Profile",
        "profile_saved": "Profile saved successfully!",
        
        # Gender Options
        "gender_prefer_not": "Prefer not to say",
        "gender_female": "Female",
        "gender_male": "Male",
        "gender_other": "Other",
        
        # Diet Options
        "diet_non_veg": "Non-Vegetarian",
        "diet_veg": "Vegetarian",
        "diet_vegan": "Vegan",
        "diet_other": "Other",
        
        # Exercise Options
        "exercise_never": "Never",
        "exercise_1_2": "1–2x/week",
        "exercise_3_4": "3–4x/week",
        "exercise_daily": "Daily",
        
        # Smoking/Drinking Options
        "habit_never": "Never",
        "habit_occasional": "Occasional",
        "habit_regular": "Regular",
        "habit_former": "Former",
        
        # Advanced Settings
        "signal_processing": "Signal Processing",
        "bandpass_low": "Bandpass Low (Hz)",
        "bandpass_high": "Bandpass High (Hz)",
        "face_detection": "Face Detection",
        "detection_scale": "Detection Scale Factor",
        "min_neighbors": "Minimum Neighbors",
        "lighting_adjustments": "Lighting Adjustments",
        "enhance_contrast": "Enhance Contrast",
        "apply_denoising": "Apply Denoising",
        
        # Video Upload
        "upload_title": "Upload Video for Analysis",
        "upload_instructions": "Upload a short video (10-30 seconds) of your face in good lighting",
        "video_requirements_title": "Video Requirements",
        "requirement_duration": "Duration: 10-30 seconds",
        "requirement_lighting": "Good, even lighting on face",
        "requirement_position": "Face clearly visible and centered",
        "requirement_movement": "Minimal head movement",
        "requirement_camera": "Camera should be stable",
        "upload_button": "Choose Video File",
        "analyze_button": "Analyze Video",
        
        # Live Recording
        "recording_mode_label": "Choose Recording Method",
        "recording_mode_upload": "📤 Upload Video",
        "recording_mode_live": "📹 Record Live",
        "live_recording_title": "Live Face Recording",
        "live_recording_subtitle": "Position your face in the oval guide for best results",
        "upload_video_title": "Upload Video",
        "upload_video_help": "Supported formats: MP4, MOV, AVI, MKV",
        "upload_instructions_title": "Video Requirements",
        
        # Analysis Status
        "analyzing": "Analyzing",
        "loading_video": "Loading video...",
        "detecting_face": "Detecting face...",
        "extracting_signal": "Extracting PPG signal...",
        "computing_vitals": "Computing vitals...",
        "generating_insights": "Generating AI insights...",
        "analysis_complete": "Analysis complete!",
        
        # Results
        "vital_signs": "Vital Signs",
        "estimated_pulse": "Estimated Pulse (rPPG)",
        "stress_index": "Stress Index (0–10)",
        "estimated_bp": "Estimated BP",
        "estimated_spo2": "Estimated SpO₂",
        "confidence": "Confidence",
        "experimental_stress": "Experimental stress measurement",
        
        # Risk Assessment
        "risk_assessment": "Risk Assessment",
        "risk_score": "Risk Score",
        "low_risk": "Low Risk",
        "moderate_risk": "Moderate Risk",
        "high_risk": "High Risk",
        "risk_factors": "Risk Factors",
        "protective_factors": "Protective Factors",
        
        # Health Insights
        "health_insights_title": "Health Insights (AI-Generated)",
        "recommendations_title": "Personalized Recommendations",
        "symptoms_watch_title": "Symptoms to Watch ",
        
        # Signal Processing
        "signal_processing_title": "Signal Processing & Analysis",
        "filtered_ppg": "Filtered PPG Signal & Power Spectral Density",
        "hrv_title": "Heart Rate Variability (RR Intervals)",
        "rr_interval_analysis": "RR Interval Analysis",
        "hrv_summary": "HRV Summary",
        
        # Historical View
        "viewing_historical": "Viewing Historical Session | Click 'Back to New Analysis' to return",
        "analysis_date": "Analysis Date",
        "back_to_new_analysis": "Back to New Analysis",
        
        # Trend Analysis
        "trends_title": "Health Trends Analysis",
        "trends_subtitle": "Track your health metrics over time and identify patterns",
        "back_to_home": "Back to Home",
        "days_7": "7 Days",
        "days_14": "14 Days",
        "days_30": "30 Days",
        "days_90": "90 Days",
        "summary": "Summary",
        "key_findings": "Key Findings",
        "recommendations": "Recommendations",
        "metric_trends": "Metric Trends",
        "average": "Average",
        "min": "Min",
        "max": "Max",
        "trend": "Trend",
        "increasing": "Increasing",
        "decreasing": "Decreasing",
        "stable": "Stable",
        "status": "Status",
        "improving": "Improving",
        "worsening": "Worsening",
        "concerning": "Concerning",
        "not_enough_data": "Not enough data for trend analysis in the last {days} days. Complete at least 2 analyses to unlock trends.",
        
        # Chatbot
        "chatbot_title": "Health Assistant",
        "chatbot_subtitle": "Ask questions about your health data, trends, or general health topics",
        "chatbot_expand": "Expand",
        "chatbot_minimize": "Minimize",
        "chatbot_input_placeholder": "Ask me about your health data, trends, or general health questions...",
        "chatbot_clear": "Clear Chat",
        "chatbot_go_upload": "Go to Upload",
        "suggested_questions": "Suggested questions:",
        "question_trends": "How are my trends?",
        "question_heart_rate": "What is a normal heart rate?",
        "question_bp": "How to lower blood pressure?",
        "question_stress": "Tips to reduce stress?",
        
        # Chatbot Disclaimer
        "chatbot_disclaimer": "🤖 **AI Health Assistant Disclaimer:** This chatbot provides general health information and insights based on your data. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.",
        "escalation_message": "⚠️ **Important:** Your query suggests a potentially serious health concern. Please consult a healthcare professional immediately. If this is an emergency, call your local emergency services.",
        
        # PDF Report
        "download_report": "Download Report",
        "generate_pdf": "Generate PDF Report",
        "download_pdf": "Download PDF",
        "pdf_generated": "PDF generated successfully!",
        "pdf_error": "Error generating PDF",
        
        # Errors
        "error_session_not_found": "Session not found.",
        "error_no_video": "Please upload a video first.",
        "error_analysis_failed": "Analysis failed. Please try again.",
        "error_invalid_video": "Invalid video file. Please upload a valid video.",
        
        # Disclaimers
        "disclaimer_title": "⚠️ Important Disclaimer",
        "disclaimer_text": "This is an experimental research tool. Results are estimates and should NOT be used for medical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.",
        "mandatory_disclaimer": "This system is a research prototype. Blood pressure and SpO₂ values are experimental estimates derived from video signals and are not intended for medical diagnosis.",
        
        # PDF Report Labels
        "pdf_report_title": "Wellio Health Report",
        "pdf_report_date": "Report Date:",
        "pdf_report_time": "Report Time:",
        "pdf_analysis_type": "Analysis Type:",
        "pdf_report_id": "Report ID:",
        "pdf_user_profile": "User Profile",
        "pdf_vital_signs": "Vital Signs",
        "pdf_metric": "Metric",
        "pdf_value": "Value",
        "pdf_status": "Status",
        "pdf_heart_rate": "Heart Rate (rPPG)",
        "pdf_stress_index": "Stress Index",
        "pdf_blood_pressure": "Blood Pressure",
        "pdf_spo2": "SpO₂",
        "pdf_hrv_sdnn": "HRV (SDNN)",
        "pdf_beats_analyzed": "beats analyzed",
        "pdf_risk_assessment": "Risk Assessment",
        "pdf_risk_score": "Risk Score:",
        "pdf_risk_level": "Risk Level:",
        "pdf_risk_factors": "Risk Factors:",
        "pdf_protective_factors": "Protective Factors:",
        "pdf_ai_insights": "AI Health Insights",
        "pdf_analysis": "Analysis:",
        "pdf_recommendations": "Personalized Recommendations:",
        "pdf_symptoms_watch": "Symptoms to Watch:",
        "pdf_signal_analysis": "Signal Analysis",
        "pdf_disclaimer": "IMPORTANT DISCLAIMER: This report is generated by an AI-based non-contact remote photoplethysmography (rPPG) system and is intended for informational and wellness purposes only. This is NOT a medical diagnosis. The measurements are experimental and may have significant error margins. Do not use this report to make medical decisions. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment. If you experience concerning symptoms, seek immediate medical attention.",
        
        # Stress Labels
        "stress_very_low": "Very Low Stress",
        "stress_low": "Low Stress",
        "stress_moderate": "Moderate Stress",
        "stress_high": "High Stress",
        "stress_very_high": "Very High Stress",
        
        # BP Labels
        "bp_low": "Low",
        "bp_normal": "Normal",
        "bp_elevated": "Elevated",
        "bp_stage1": "Stage 1 (High)",
        "bp_stage2": "Stage 2 (High)",
        
        # SpO2 Labels
        "spo2_normal": "Normal",
        "spo2_slightly_low": "Slightly Low",
        "spo2_low": "Low",
        "spo2_very_low": "Very Low",
        
        # Chatbot Messages
        "chatbot_unavailable": "Sorry, the chatbot is currently unavailable. Please try again later.",
        "chatbot_error": "I apologize, but I'm having trouble generating a response right now. Please try again in a moment.",
        "chatbot_safety_note": "*Note: Please consult a healthcare professional for proper diagnosis and treatment.*",
        
        # Health Insights Sections
        "insights_detailed_analysis": "DETAILED HEALTH ANALYSIS:",
        "insights_risk_factors": "RISK FACTORS:",
        "insights_positive_indicators": "POSITIVE INDICATORS:",
        "insights_recommendations": "PERSONALIZED RECOMMENDATIONS:",
        "insights_symptoms_watch": "SYMPTOMS TO WATCH:",
        
        # Trend Analysis Labels
        "trend_up": "up",
        "trend_down": "down",
        "trend_stable": "stable",
        "trend_improving": "Improving",
        "trend_worsening": "Worsening",
        "trend_concerning": "Concerning",
        "trend_heart_rate": "Heart Rate",
        "trend_stress_level": "Stress Level",
        "trend_systolic_bp": "Systolic BP",
        "trend_diastolic_bp": "Diastolic BP",
        "trend_spo2": "SpO₂",
        "trend_summary_prefix": "Analyzed {count} sessions over {days} days.",
        "trend_all_stable": "All metrics are stable.",
        "trend_continue_monitoring": "Continue monitoring your health regularly",
        "trend_hr_improving": "Heart rate is improving (avg: {avg} BPM)",
        "trend_hr_increasing": "Heart rate is increasing (avg: {avg} BPM)",
        "trend_hr_concerning": "Heart rate shows concerning pattern (avg: {avg} BPM)",
        "trend_stress_decreasing": "Stress levels are decreasing ({change}% improvement)",
        "trend_stress_increasing": "Stress levels are increasing (avg: {avg}/10)",
        "trend_bp_improving": "Blood pressure is improving (avg: {avg} mmHg systolic)",
        "trend_bp_increasing": "Blood pressure is increasing (avg: {avg} mmHg systolic)",
        "trend_spo2_concerning": "Oxygen saturation is concerning (avg: {avg}%)",
        "trend_rec_stress": "Consider stress management techniques and regular exercise",
        "trend_rec_consult_hr": "Consult a healthcare professional about your heart rate",
        "trend_rec_continue": "Continue your current stress management practices",
        "trend_rec_meditation": "Try meditation, deep breathing, or regular exercise to manage stress",
        "trend_rec_bp": "Monitor sodium intake and maintain regular physical activity",
        "trend_rec_consult_spo2": "Consult a healthcare professional about your oxygen levels",
        
        # Pulse Labels
        "pulse_low": "Low",
        "pulse_slightly_low": "Slightly Low",
        "pulse_normal": "Normal",
        "pulse_high": "High",
        "pulse_very_high": "Very High",
        
        # Upload Section
        "upload_video_title": "Upload Video for Analysis",
        "upload_video_help": "Upload a short video (10-30 seconds) of your face in good lighting",
        "upload_instructions_title": "Video Requirements",
        "complete_profile_warning": "Please complete your profile first",
        "fill_profile_sidebar": "Fill out your profile in the sidebar to continue",
        "profile_required_info": "Profile information is required for accurate health analysis",
        "processing_frame": "Processing frame",
        "complete": "Complete",
        "video_processed_success": "Video processed successfully!",
        "start_new_analysis": "Start New Analysis",
        
        # Analysis Results
        "typical_error": "Typical error",
        "estimated_bp_experimental": "Estimated BP (Experimental)",
        "estimated_spo2_experimental": "Estimated SpO₂ (Experimental)",
        
        # Risk Assessment Labels
        "risk_assessment_experimental": "Risk Assessment (Heuristic, Experimental)",
        "risk_summary_low": "Your risk score is low. Continue maintaining your healthy lifestyle habits.",
        "risk_summary_moderate": "Your risk score is moderate. Consider improving the risk factors listed above to lower your overall risk.",
        "risk_summary_high": "Your risk score is high. We strongly recommend consulting with a healthcare professional and addressing the risk factors listed above.",
        
        # Health Insights Labels
        "generating_insights": "Generating health insights...",
        "insights_unavailable": "Health insights unavailable at the moment.",
        "insights_module_unavailable": "Health Insights module is currently unavailable. Update your dependencies to enable this feature.",
        "maintain_healthy_habits": "Maintain your current healthy habits.",
        "no_symptoms_watch": "No specific symptoms to watch for at this time.",
        
        # Signal Processing Labels
        "filtered_ppg_signal": "Filtered PPG Signal (Green Channel)",
        "frame": "Frame",
        "normalized_intensity": "Normalized Intensity",
        "power_spectral_density": "Power Spectral Density (Welch)",
        "frequency_hz": "Frequency (Hz)",
        "power_log_scale": "Power (log scale)",
        "peak_bpm": "Peak",
        "rr_interval_distribution": "RR Interval Distribution",
        "rr_interval_ms": "RR Interval (ms)",
        "frequency": "Frequency",
        "rr_intervals_over_time": "RR Intervals Over Time",
        "beat_number": "Beat #",
        "hrv_summary_label": "HRV Summary",
        "beats_detected": "# of beats detected",
        "sdnn_std_dev": "SDNN (std dev)",
        "mean_rr": "Mean RR",
        "pnn50": "pNN50",
        "not_enough_beats": "Not enough beats detected for HRV analysis. Try a longer or clearer video.",
        "advanced_signal_quality": "Advanced Signal Quality Metrics",
        "signal_to_noise_ratio": "Signal-to-Noise Ratio (SNR)",
        "snr_help": "Higher is better. >2.0 = good, 1.0–2.0 = moderate, <1.0 = poor",
        "quality_flags": "Quality Flags",
        
        # Session Save & PDF
        "save_download": "Save & Download",
        "session_saved": "Session saved to history!",
        "could_not_save_session": "Could not save session to history",
        "generate_pdf_report": "Generate PDF Report",
        "generating_pdf": "Generating PDF...",
        "download_pdf_button": "Download PDF",
        
        # Error Messages
        "error_processing_video": "Error Processing Video",
        "troubleshooting": "Troubleshooting:",
        "ensure_good_lighting": "Ensure good lighting (bright environment)",
        "keep_face_visible": "Keep face visible and relatively still",
        "try_different_video": "Try a different or shorter video",
        "ensure_video_format": "Ensure video format is MP4 or MOV",
        "error_generating_insights": "Error generating insights",
        
        # Chatbot Labels
        "expand": "Expand",
        "minimize": "Minimize",
        "chatbot_input_label": "Chat Input",
        "go_to_upload": "Go to Upload",
        "question_exercise": "What are the health benefits of regular exercise?",
        
        # Historical View Labels
        "viewing_history": "Viewing Historical Session",
        "back_to_new_analysis_instruction": "Click 'Back to New Analysis' to return",
        "session_not_found": "Session not found.",
        "back_to_new_analysis_button": "Back to New Analysis",
        
        # Trend Analysis Labels  
        "analyzing_trends": "Analyzing trends over the last {days} days...",
        "not_enough_trend_data": "Not enough data for trend analysis in the last {days} days. Complete at least 2 analyses to unlock trends.",
        "heart_rate_trend": "Heart Rate Trend - Last {period} Days",
        "stress_level_trend": "Stress Level Trend - Last {period} Days",
        "bp_trend": "Blood Pressure Trend - Last {period} Days",
        "spo2_trend": "Oxygen Saturation Trend - Last {period} Days",
        "date": "Date",
        "heart_rate_bpm": "Heart Rate (BPM)",
        "stress_level_scale": "Stress Level (0-10)",
        "systolic_bp_mmhg": "Systolic BP (mmHg)",
        "spo2_percent": "SpO₂ (%)",
        "target": "Target",
        "normal_threshold": "Normal Threshold",
        "trend_line": "Trend",
        "increasing_arrow": "↑ Increasing",
        "decreasing_arrow": "↓ Decreasing",
        "stable_arrow": "→ Stable",
        
        # Sidebar Labels
        "history_sidebar_title": "History",
        "signal_processing_sidebar": "Signal Processing",
        "face_detection_sidebar": "Face Detection",
        "lighting_adjustments_sidebar": "Lighting Adjustments",
        
        # Gender Options (using existing keys but adding for clarity)
        "prefer_not_say": "Prefer not to say",
        "female": "Female",
        "male": "Male",
        "other": "Other",
        
        # Diet Options (using existing keys)
        "non_vegetarian": "Non-Vegetarian",
        "vegetarian": "Vegetarian",
        "vegan": "Vegan",
        
        # Exercise Options (using existing keys)
        "never": "Never",
        "exercise_1_2": "1–2x/week",
        "exercise_3_4": "3–4x/week",
        "daily": "Daily",
        
        # Smoking/Drinking Options (using existing keys)
        "occasional": "Occasional",
        "regular": "Regular",
        "former": "Former",
        
        # Misc
        "na": "N/A",
        "unknown": "Unknown",
        "loading": "Loading...",
        "processing": "Processing...",
        "please_wait": "Please wait...",
        # Audio Summary
        "audio_intro": "Here is your health summary.",
        "audio_hr": "Your estimated Heart Rate is {value} beats per minute.",
        "audio_stress": "Stress Level is {value} out of 10.",
        "audio_bp": "Estimated Blood Pressure is {systolic} over {diastolic}.",
        "audio_spo2": "Oxygen Saturation is {value} percent.",
        "audio_risk": "Your Risk Assessment score is {score} out of 10. This is {level}.",
        "audio_insights_intro": "Here are some insights.",
        "audio_recs": "Recommendations: ",
        "audio_symptoms": "Symptoms to watch for: ",
    },
    
    "hi": {
        # App & Navigation
        "app_title": "वेलियो",
        "app_subtitle": "rPPG वाइटल्स अनुमान",
        "page_title": "प्रायोगिक rPPG वाइटल्स डेमो",
        
        # Authentication
        "login_title": "वेलियो - लॉगिन",
        "login_subtitle": "अपने स्वास्थ्य डैशबोर्ड तक पहुंचने के लिए साइन इन करें",
        "signup_title": "वेलियो - साइन अप",
        "signup_subtitle": "अपने स्वास्थ्य की निगरानी शुरू करने के लिए अपना खाता बनाएं",
        "email_label": "ईमेल",
        "email_placeholder": "your.email@example.com",
        "password_label": "पासवर्ड",
        "password_confirm_label": "पासवर्ड की पुष्टि करें",
        "password_help": "कम से कम 8 अक्षर, अपरकेस, लोअरकेस और नंबर के साथ",
        "name_label": "पूरा नाम",
        "name_placeholder": "जॉन डो",
        "login_button": "लॉगिन",
        "signup_button": "साइन अप",
        "create_account_button": "खाता बनाएं",
        "back_to_login": "लॉगिन पर वापस जाएं",
        "logout_button": "लॉगआउट",
        "login_success": "लॉगिन सफल!",
        "signup_success": "खाता सफलतापूर्वक बनाया गया! कृपया लॉगिन करें।",
        "new_to_wellio": "वेलियो में नए हैं? अपना खाता बनाने और अपने स्वास्थ्य की निगरानी शुरू करने के लिए 'साइन अप' पर क्लिक करें!",
        "data_secure": "आपका डेटा सुरक्षित है: पासवर्ड उद्योग-मानक bcrypt हैशिंग का उपयोग करके एन्क्रिप्ट किए गए हैं।",
        
        # Password Strength
        "password_strength": "पासवर्ड की मजबूती",
        "strength_weak": "कमजोर",
        "strength_medium": "मध्यम",
        "strength_strong": "मजबूत",
        
        # Validation Messages
        "fill_all_fields": "कृपया सभी फ़ील्ड भरें",
        "passwords_no_match": "पासवर्ड मेल नहीं खाते",
        "enter_email_password": "कृपया ईमेल और पासवर्ड दोनों दर्ज करें",
        
        # Sidebar
        "settings_title": "सेटिंग्स",
        "user_profile_title": "उपयोगकर्ता प्रोफ़ाइल",
        "history_title": "इतिहास",
        "total_sessions": "कुल सत्र",
        "recent_analyses": "हाल के विश्लेषण:",
        "no_history": "अभी तक कोई इतिहास नहीं। शुरू करने के लिए एक विश्लेषण पूरा करें!",
        "view_trends_button": "ट्रेंड विश्लेषण देखें",
        "show_advanced_plots": "उन्नत सिग्नल प्लॉट दिखाएं",
        
        # Language Selector
        "language_label": "भाषा",
        
        # Profile Form
        "age_label": "आयु",
        "gender_label": "लिंग",
        "height_label": "ऊंचाई (सेमी)",
        "weight_label": "वजन (किग्रा)",
        "diet_label": "आहार",
        "exercise_label": "व्यायाम की आवृत्ति",
        "sleep_label": "नींद के घंटे (प्रति रात)",
        "smoking_label": "धूम्रपान की आदतें",
        "drinking_label": "पीने की आदतें",
        "save_profile_button": "प्रोफ़ाइल सहेजें",
        "profile_saved": "प्रोफ़ाइल सफलतापूर्वक सहेजी गई!",
        
        # Gender Options
        "gender_prefer_not": "नहीं कहना पसंद करते",
        "gender_female": "महिला",
        "gender_male": "पुरुष",
        "gender_other": "अन्य",
        
        # Diet Options
        "diet_non_veg": "मांसाहारी",
        "diet_veg": "शाकाहारी",
        "diet_vegan": "शुद्ध शाकाहारी",
        "diet_other": "अन्य",
        
        # Exercise Options
        "exercise_never": "कभी नहीं",
        "exercise_1_2": "1-2 बार/सप्ताह",
        "exercise_3_4": "3-4 बार/सप्ताह",
        "exercise_daily": "रोज़ाना",
        
        # Smoking/Drinking Options
        "habit_never": "कभी नहीं",
        "habit_occasional": "कभी-कभी",
        "habit_regular": "नियमित",
        "habit_former": "पूर्व",
        
        # Advanced Settings
        "signal_processing": "सिग्नल प्रोसेसिंग",
        "bandpass_low": "बैंडपास लो (Hz)",
        "bandpass_high": "बैंडपास हाई (Hz)",
        "face_detection": "चेहरा पहचान",
        "detection_scale": "डिटेक्शन स्केल फैक्टर",
        "min_neighbors": "न्यूनतम पड़ोसी",
        "lighting_adjustments": "प्रकाश समायोजन",
        "enhance_contrast": "कंट्रास्ट बढ़ाएं",
        "apply_denoising": "डीनॉइज़िंग लागू करें",
        
        # Video Upload
        "upload_title": "विश्लेषण के लिए वीडियो अपलोड करें",
        "upload_instructions": "अच्छी रोशनी में अपने चेहरे का एक छोटा वीडियो (10-30 सेकंड) अपलोड करें",
        "video_requirements_title": "वीडियो आवश्यकताएं",
        "requirement_duration": "अवधि: 10-30 सेकंड",
        "requirement_lighting": "चेहरे पर अच्छी, समान रोशनी",
        "requirement_position": "चेहरा स्पष्ट रूप से दिखाई दे और केंद्रित हो",
        "requirement_movement": "सिर की न्यूनतम गति",
        "requirement_camera": "कैमरा स्थिर होना चाहिए",
        "upload_button": "वीडियो फ़ाइल चुनें",
        "analyze_button": "वीडियो का विश्लेषण करें",
        
        # Live Recording
        "recording_mode_label": "रिकॉर्डिंग विधि चुनें",
        "recording_mode_upload": "📤 वीडियो अपलोड करें",
        "recording_mode_live": "📹 लाइव रिकॉर्ड करें",
        "live_recording_title": "लाइव फेस रिकॉर्डिंग",
        "live_recording_subtitle": "सर्वोत्तम परिणामों के लिए अपने चेहरे को ओवल गाइड में रखें",
        "upload_video_title": "वीडियो अपलोड करें",
        "upload_video_help": "समर्थित प्रारूप: MP4, MOV, AVI, MKV",
        "upload_instructions_title": "वीडियो आवश्यकताएं",
        
        # Analysis Status
        "analyzing": "विश्लेषण कर रहे हैं",
        "loading_video": "वीडियो लोड हो रहा है...",
        "detecting_face": "चेहरा पहचान रहे हैं...",
        "extracting_signal": "PPG सिग्नल निकाल रहे हैं...",
        "computing_vitals": "वाइटल्स की गणना कर रहे हैं...",
        "generating_insights": "AI अंतर्दृष्टि उत्पन्न कर रहे हैं...",
        "analysis_complete": "विश्लेषण पूर्ण!",
        
        # Results
        "vital_signs": "महत्वपूर्ण संकेत",
        "estimated_pulse": "अनुमानित पल्स (rPPG)",
        "stress_index": "तनाव सूचकांक (0-10)",
        "estimated_bp": "अनुमानित BP",
        "estimated_spo2": "अनुमानित SpO₂",
        "confidence": "विश्वास",
        "experimental_stress": "प्रायोगिक तनाव माप",
        
        # Risk Assessment
        "risk_assessment": "जोखिम मूल्यांकन",
        "risk_score": "जोखिम स्कोर",
        "low_risk": "कम जोखिम",
        "moderate_risk": "मध्यम जोखिम",
        "high_risk": "उच्च जोखिम",
        "risk_factors": "जोखिम कारक",
        "protective_factors": "सुरक्षात्मक कारक",
        
        # Health Insights
        "health_insights_title": "स्वास्थ्य अंतर्दृष्टि (AI-जनित)",
        "recommendations_title": "व्यक्तिगत सिफारिशें",
        "symptoms_watch_title": "देखने योग्य लक्षण",
        
        # Signal Processing
        "signal_processing_title": "सिग्नल प्रोसेसिंग और विश्लेषण",
        "filtered_ppg": "फ़िल्टर किया गया PPG सिग्नल और पावर स्पेक्ट्रल डेंसिटी",
        "hrv_title": "हृदय गति परिवर्तनशीलता (RR अंतराल)",
        "rr_interval_analysis": "RR अंतराल विश्लेषण",
        "hrv_summary": "HRV सारांश",
        
        # Historical View
        "viewing_historical": "ऐतिहासिक सत्र देख रहे हैं | वापस जाने के लिए 'नए विश्लेषण पर वापस जाएं' पर क्लिक करें",
        "analysis_date": "विश्लेषण तिथि",
        "back_to_new_analysis": "नए विश्लेषण पर वापस जाएं",
        
        # Trend Analysis
        "trends_title": "स्वास्थ्य ट्रेंड विश्लेषण",
        "trends_subtitle": "समय के साथ अपने स्वास्थ्य मेट्रिक्स को ट्रैक करें और पैटर्न की पहचान करें",
        "back_to_home": "होम पर वापस जाएं",
        "days_7": "7 दिन",
        "days_14": "14 दिन",
        "days_30": "30 दिन",
        "days_90": "90 दिन",
        "summary": "सारांश",
        "key_findings": "मुख्य निष्कर्ष",
        "recommendations": "सिफारिशें",
        "metric_trends": "मेट्रिक ट्रेंड",
        "average": "औसत",
        "min": "न्यूनतम",
        "max": "अधिकतम",
        "trend": "ट्रेंड",
        "increasing": "बढ़ रहा है",
        "decreasing": "घट रहा है",
        "stable": "स्थिर",
        "status": "स्थिति",
        "improving": "सुधार हो रहा है",
        "worsening": "बिगड़ रहा है",
        "concerning": "चिंताजनक",
        "not_enough_data": "पिछले {days} दिनों में ट्रेंड विश्लेषण के लिए पर्याप्त डेटा नहीं। ट्रेंड अनलॉक करने के लिए कम से कम 2 विश्लेषण पूरे करें।",
        
        # Chatbot
        "chatbot_title": "स्वास्थ्य सहायक",
        "chatbot_subtitle": "अपने स्वास्थ्य डेटा, ट्रेंड या सामान्य स्वास्थ्य विषयों के बारे में प्रश्न पूछें",
        "chatbot_expand": "विस्तार करें",
        "chatbot_minimize": "छोटा करें",
        "chatbot_input_placeholder": "मुझसे अपने स्वास्थ्य डेटा, ट्रेंड या सामान्य स्वास्थ्य प्रश्नों के बारे में पूछें...",
        "chatbot_clear": "चैट साफ़ करें",
        "chatbot_go_upload": "अपलोड पर जाएं",
        "suggested_questions": "सुझाए गए प्रश्न:",
        "question_trends": "मेरे ट्रेंड कैसे हैं?",
        "question_heart_rate": "सामान्य हृदय गति क्या है?",
        "question_bp": "रक्तचाप कैसे कम करें?",
        "question_stress": "तनाव कम करने के टिप्स?",
        
        # Chatbot Disclaimer
        "chatbot_disclaimer": "🤖 **AI स्वास्थ्य सहायक अस्वीकरण:** यह चैटबॉट आपके डेटा के आधार पर सामान्य स्वास्थ्य जानकारी और अंतर्दृष्टि प्रदान करता है। यह पेशेवर चिकित्सा सलाह, निदान या उपचार का विकल्प नहीं है। चिकित्सा चिंताओं के लिए हमेशा एक योग्य स्वास्थ्य सेवा प्रदाता से परामर्श करें।",
        "escalation_message": "⚠️ **महत्वपूर्ण:** आपका प्रश्न संभावित रूप से गंभीर स्वास्थ्य चिंता का सुझाव देता है। कृपया तुरंत एक स्वास्थ्य सेवा पेशेवर से परामर्श करें। यदि यह आपातकाल है, तो अपनी स्थानीय आपातकालीन सेवाओं को कॉल करें।",
        
        # PDF Report
        "download_report": "रिपोर्ट डाउनलोड करें",
        "generate_pdf": "PDF रिपोर्ट जनरेट करें",
        "download_pdf": "PDF डाउनलोड करें",
        "pdf_generated": "PDF सफलतापूर्वक जनरेट की गई!",
        "pdf_error": "PDF जनरेट करने में त्रुटि",
        
        # Errors
        "error_session_not_found": "सत्र नहीं मिला।",
        "error_no_video": "कृपया पहले एक वीडियो अपलोड करें।",
        "error_analysis_failed": "विश्लेषण विफल। कृपया पुनः प्रयास करें।",
        "error_invalid_video": "अमान्य वीडियो फ़ाइल। कृपया एक मान्य वीडियो अपलोड करें।",
        
        # Disclaimers
        "disclaimer_title": "⚠️ महत्वपूर्ण अस्वीकरण",
        "disclaimer_text": "यह एक प्रायोगिक अनुसंधान उपकरण है। परिणाम अनुमान हैं और चिकित्सा निदान या उपचार निर्णयों के लिए उपयोग नहीं किए जाने चाहिए। चिकित्सा सलाह के लिए हमेशा स्वास्थ्य सेवा पेशेवरों से परामर्श करें।",
        
        # Pulse Labels
        "pulse_low": "कम",
        "pulse_slightly_low": "थोड़ा कम",
        "pulse_normal": "सामान्य",
        "pulse_high": "उच्च",
        "pulse_very_high": "बहुत उच्च",
        
        # Upload Section
        "upload_video_title": "विश्लेषण के लिए वीडियो अपलोड करें",
        "upload_video_help": "अच्छी रोशनी में अपने चेहरे का एक छोटा वीडियो (10-30 सेकंड) अपलोड करें",
        "upload_instructions_title": "वीडियो आवश्यकताएं",
        "complete_profile_warning": "कृपया पहले अपनी प्रोफ़ाइल पूरी करें",
        "fill_profile_sidebar": "जारी रखने के लिए साइडबार में अपनी प्रोफ़ाइल भरें",
        "profile_required_info": "सटीक स्वास्थ्य विश्लेषण के लिए प्रोफ़ाइल जानकारी आवश्यक है",
        "processing_frame": "फ्रेम प्रोसेस हो रहा है",
        "complete": "पूर्ण",
        "video_processed_success": "वीडियो सफलतापूर्वक प्रोसेस किया गया!",
        "start_new_analysis": "नया विश्लेषण शुरू करें",
        
        # Analysis Results
        "typical_error": "सामान्य त्रुटि",
        "estimated_bp_experimental": "अनुमानित BP (प्रायोगिक)",
        "estimated_spo2_experimental": "अनुमानित SpO₂ (प्रायोगिक)",
        
        # Risk Assessment Labels
        "risk_assessment_experimental": "जोखिम मूल्यांकन (अनुमानी, प्रायोगिक)",
        "risk_summary_low": "आपका जोखिम स्कोर कम है। अपनी स्वस्थ जीवनशैली की आदतों को बनाए रखना जारी रखें।",
        "risk_summary_moderate": "आपका जोखिम स्कोर मध्यम है। अपने समग्र जोखिम को कम करने के लिए ऊपर सूचीबद्ध जोखिम कारकों में सुधार करने पर विचार करें।",
        "risk_summary_high": "आपका जोखिम स्कोर उच्च है। हम दृढ़ता से अनुशंसा करते हैं कि आप एक स्वास्थ्य सेवा पेशेवर से परामर्श करें और ऊपर सूचीबद्ध जोखिम कारकों को संबोधित करें।",
        
        # Health Insights Labels
        "generating_insights": "स्वास्थ्य अंतर्दृष्टि उत्पन्न कर रहे हैं...",
        "insights_unavailable": "इस समय स्वास्थ्य अंतर्दृष्टि उपलब्ध नहीं है।",
        "insights_module_unavailable": "स्वास्थ्य अंतर्दृष्टि मॉड्यूल वर्तमान में उपलब्ध नहीं है। इस सुविधा को सक्षम करने के लिए अपनी निर्भरताओं को अपडेट करें।",
        "maintain_healthy_habits": "अपनी वर्तमान स्वस्थ आदतों को बनाए रखें।",
        "no_symptoms_watch": "इस समय देखने के लिए कोई विशिष्ट लक्षण नहीं हैं।",
        
        # Signal Processing Labels
        "filtered_ppg_signal": "फ़िल्टर किया गया PPG सिग्नल (ग्रीन चैनल)",
        "frame": "फ्रेम",
        "normalized_intensity": "सामान्यीकृत तीव्रता",
        "power_spectral_density": "पावर स्पेक्ट्रल डेंसिटी (वेल्च)",
        "frequency_hz": "आवृत्ति (Hz)",
        "power_log_scale": "पावर (लॉग स्केल)",
        "peak_bpm": "शिखर",
        "rr_interval_distribution": "RR अंतराल वितरण",
        "rr_interval_ms": "RR अंतराल (ms)",
        "frequency": "आवृत्ति",
        "rr_intervals_over_time": "समय के साथ RR अंतराल",
        "beat_number": "बीट #",
        "hrv_summary_label": "HRV सारांश",
        "beats_detected": "पता लगाई गई बीट्स की संख्या",
        "sdnn_std_dev": "SDNN (मानक विचलन)",
        "mean_rr": "औसत RR",
        "pnn50": "pNN50",
        "not_enough_beats": "HRV विश्लेषण के लिए पर्याप्त बीट्स का पता नहीं चला। एक लंबा या स्पष्ट वीडियो आज़माएं।",
        "advanced_signal_quality": "उन्नत सिग्नल गुणवत्ता मेट्रिक्स",
        "signal_to_noise_ratio": "सिग्नल-टू-नॉइज़ रेशियो (SNR)",
        "snr_help": "अधिक बेहतर है। >2.0 = अच्छा, 1.0–2.0 = मध्यम, <1.0 = खराब",
        "quality_flags": "गुणवत्ता फ्लैग",
        
        # Session Save & PDF
        "save_download": "सहेजें और डाउनलोड करें",
        "session_saved": "सत्र इतिहास में सहेजा गया!",
        "could_not_save_session": "सत्र को इतिहास में सहेज नहीं सका",
        "generate_pdf_report": "PDF रिपोर्ट जनरेट करें",
        "generating_pdf": "PDF जनरेट हो रहा है...",
        "download_pdf_button": "PDF डाउनलोड करें",
        
        # Error Messages
        "error_processing_video": "वीडियो प्रोसेस करने में त्रुटि",
        "troubleshooting": "समस्या निवारण:",
        "ensure_good_lighting": "अच्छी रोशनी सुनिश्चित करें (उज्ज्वल वातावरण)",
        "keep_face_visible": "चेहरे को दिखाई देने वाला और अपेक्षाकृत स्थिर रखें",
        "try_different_video": "एक अलग या छोटा वीडियो आज़माएं",
        "ensure_video_format": "सुनिश्चित करें कि वीडियो फ़ॉर्मेट MP4 या MOV है",
        "error_generating_insights": "अंतर्दृष्टि उत्पन्न करने में त्रुटि",
        
        # Chatbot Labels
        "expand": "विस्तार करें",
        "minimize": "छोटा करें",
        "chatbot_input_label": "चैट इनपुट",
        "go_to_upload": "अपलोड पर जाएं",
        "question_exercise": "नियमित व्यायाम के स्वास्थ्य लाभ क्या हैं?",
        
        # Historical View Labels
        "viewing_history": "ऐतिहासिक सत्र देख रहे हैं",
        "back_to_new_analysis_instruction": "वापस जाने के लिए 'नए विश्लेषण पर वापस जाएं' पर क्लिक करें",
        "session_not_found": "सत्र नहीं मिला।",
        "back_to_new_analysis_button": "नए विश्लेषण पर वापस जाएं",
        
        # Trend Analysis Labels  
        "analyzing_trends": "पिछले {days} दिनों के ट्रेंड का विश्लेषण कर रहे हैं...",
        "not_enough_trend_data": "पिछले {days} दिनों में ट्रेंड विश्लेषण के लिए पर्याप्त डेटा नहीं। ट्रेंड अनलॉक करने के लिए कम से कम 2 विश्लेषण पूरे करें।",
        "heart_rate_trend": "हृदय गति ट्रेंड - पिछले {period} दिन",
        "stress_level_trend": "तनाव स्तर ट्रेंड - पिछले {period} दिन",
        "bp_trend": "रक्तचाप ट्रेंड - पिछले {period} दिन",
        "spo2_trend": "ऑक्सीजन संतृप्ति ट्रेंड - पिछले {period} दिन",
        "date": "तिथि",
        "heart_rate_bpm": "हृदय गति (BPM)",
        "stress_level_scale": "तनाव स्तर (0-10)",
        "systolic_bp_mmhg": "सिस्टोलिक BP (mmHg)",
        "spo2_percent": "SpO₂ (%)",
        "target": "लक्ष्य",
        "normal_threshold": "सामान्य सीमा",
        "trend_line": "ट्रेंड",
        "increasing_arrow": "↑ बढ़ रहा है",
        "decreasing_arrow": "↓ घट रहा है",
        "stable_arrow": "→ स्थिर",
        
        # Sidebar Labels
        "history_sidebar_title": "इतिहास",
        "signal_processing_sidebar": "सिग्नल प्रोसेसिंग",
        "face_detection_sidebar": "चेहरा पहचान",
        "lighting_adjustments_sidebar": "प्रकाश समायोजन",
        
        # Gender Options (using existing keys but adding for clarity)
        "prefer_not_say": "नहीं कहना पसंद करते",
        "female": "महिला",
        "male": "पुरुष",
        "other": "अन्य",
        
        # Diet Options (using existing keys)
        "non_vegetarian": "मांसाहारी",
        "vegetarian": "शाकाहारी",
        "vegan": "शुद्ध शाकाहारी",
        
        # Exercise Options (using existing keys)
        "never": "कभी नहीं",
        "exercise_1_2": "1-2 बार/सप्ताह",
        "exercise_3_4": "3-4 बार/सप्ताह",
        "daily": "रोज़ाना",
        
        # Smoking/Drinking Options (using existing keys)
        "occasional": "कभी-कभी",
        "regular": "नियमित",
        "former": "पूर्व",
        
        # Misc
        "na": "लागू नहीं",
        "unknown": "अज्ञात",
        "loading": "लोड हो रहा है...",
        "processing": "प्रोसेस हो रहा है...",
        "please_wait": "कृपया प्रतीक्षा करें...",
        # Audio Summary
        "audio_intro": "यहाँ आपका स्वास्थ्य सारांश है।",
        "audio_hr": "आपकी अनुमानित हृदय गति {value} बीट्स प्रति मिनट है।",
        "audio_stress": "तनाव स्तर 10 में से {value} है।",
        "audio_bp": "अनुमानित रक्तचाप {systolic} बटा {diastolic} है।",
        "audio_spo2": "ऑक्सीजन संतृप्ति {value} प्रतिशत है।",
        "audio_risk": "आपका जोखिम मूल्यांकन स्कोर 10 में से {score} है। यह {level} है।",
        "audio_insights_intro": "यहाँ कुछ अंतर्दृष्टि दी गई हैं।",
        "audio_recs": "सिफारिशें: ",
        "audio_symptoms": "देखने योग्य लक्षण: ",
    },
    
    "te": {
        # App & Navigation
        "app_title": "వెల్లియో",
        "app_subtitle": "rPPG వైటల్స్ అంచనా",
        "page_title": "ప్రయోగాత్మక rPPG వైటల్స్ డెమో",
        
        # Authentication
        "login_title": "వెల్లియో - లాగిన్",
        "login_subtitle": "మీ ఆరోగ్య డాష్‌బోర్డ్‌ను యాక్సెస్ చేయడానికి సైన్ ఇన్ చేయండి",
        "signup_title": "వెల్లియో - సైన్ అప్",
        "signup_subtitle": "మీ ఆరోగ్యాన్ని పర్యవేక్షించడం ప్రారంభించడానికి మీ ఖాతాను సృష్టించండి",
        "email_label": "ఇమెయిల్",
        "email_placeholder": "your.email@example.com",
        "password_label": "పాస్‌వర్డ్",
        "password_confirm_label": "పాస్‌వర్డ్‌ను నిర్ధారించండి",
        "password_help": "కనీసం 8 అక్షరాలు, అప్పర్‌కేస్, లోయర్‌కేస్ మరియు సంఖ్యతో",
        "name_label": "పూర్తి పేరు",
        "name_placeholder": "జాన్ డో",
        "login_button": "లాగిన్",
        "signup_button": "సైన్ అప్",
        "create_account_button": "ఖాతాను సృష్టించండి",
        "back_to_login": "లాగిన్‌కు తిరిగి వెళ్ళండి",
        "logout_button": "లాగ్అవుట్",
        "login_success": "లాగిన్ విజయవంతం!",
        "signup_success": "ఖాతా విజయవంతంగా సృష్టించబడింది! దయచేసి లాగిన్ చేయండి.",
        "new_to_wellio": "వెల్లియోకు కొత్తవారా? మీ ఖాతాను సృష్టించడానికి మరియు మీ ఆరోగ్యాన్ని పర్యవేక్షించడం ప్రారంభించడానికి 'సైన్ అప్' క్లిక్ చేయండి!",
        "data_secure": "మీ డేటా సురక్షితం: పాస్‌వర్డ్‌లు పరిశ్రమ-ప్రమాణ bcrypt హ్యాషింగ్ ఉపయోగించి ఎన్‌క్రిప్ట్ చేయబడ్డాయి.",
        
        # Password Strength
        "password_strength": "పాస్‌వర్డ్ బలం",
        "strength_weak": "బలహీనమైన",
        "strength_medium": "మధ్యస్థ",
        "strength_strong": "బలమైన",
        
        # Validation Messages
        "fill_all_fields": "దయచేసి అన్ని ఫీల్డ్‌లను పూరించండి",
        "passwords_no_match": "పాస్‌వర్డ్‌లు సరిపోలలేదు",
        "enter_email_password": "దయచేసి ఇమెయిల్ మరియు పాస్‌వర్డ్ రెండింటినీ నమోదు చేయండి",
        
        # Sidebar
        "settings_title": "సెట్టింగ్‌లు",
        "user_profile_title": "వినియోగదారు ప్రొఫైల్",
        "history_title": "చరిత్ర",
        "total_sessions": "మొత్తం సెషన్‌లు",
        "recent_analyses": "ఇటీవలి విశ్లేషణలు:",
        "no_history": "ఇంకా చరిత్ర లేదు. ప్రారంభించడానికి విశ్లేషణను పూర్తి చేయండి!",
        "view_trends_button": "ట్రెండ్ విశ్లేషణను చూడండి",
        "show_advanced_plots": "అధునాతన సిగ్నల్ ప్లాట్‌లను చూపించు",
        
        # Language Selector
        "language_label": "భాష",
        
        # Profile Form
        "age_label": "వయస్సు",
        "gender_label": "లింగం",
        "height_label": "ఎత్తు (సెం.మీ)",
        "weight_label": "బరువు (కి.గ్రా)",
        "diet_label": "ఆహారం",
        "exercise_label": "వ్యాయామ ఫ్రీక్వెన్సీ",
        "sleep_label": "నిద్ర గంటలు (రాత్రికి)",
        "smoking_label": "ధూమపాన అలవాట్లు",
        "drinking_label": "మద్యపాన అలవాట్లు",
        "save_profile_button": "ప్రొఫైల్‌ను సేవ్ చేయండి",
        "profile_saved": "ప్రొఫైల్ విజయవంతంగా సేవ్ చేయబడింది!",
        
        # Gender Options
        "gender_prefer_not": "చెప్పకూడదనుకుంటున్నాను",
        "gender_female": "స్త్రీ",
        "gender_male": "పురుషుడు",
        "gender_other": "ఇతర",
        
        # Diet Options
        "diet_non_veg": "మాంసాహారం",
        "diet_veg": "శాకాహారం",
        "diet_vegan": "శుద్ధ శాకాహారం",
        "diet_other": "ఇతర",
        
        # Exercise Options
        "exercise_never": "ఎప్పుడూ లేదు",
        "exercise_1_2": "1-2 సార్లు/వారం",
        "exercise_3_4": "3-4 సార్లు/వారం",
        "exercise_daily": "ప్రతిరోజూ",
        
        # Smoking/Drinking Options
        "habit_never": "ఎప్పుడూ లేదు",
        "habit_occasional": "అప్పుడప్పుడు",
        "habit_regular": "క్రమం తప్పకుండా",
        "habit_former": "గతంలో",
        
        # Advanced Settings
        "signal_processing": "సిగ్నల్ ప్రాసెసింగ్",
        "bandpass_low": "బ్యాండ్‌పాస్ లో (Hz)",
        "bandpass_high": "బ్యాండ్‌పాస్ హై (Hz)",
        "face_detection": "ముఖ గుర్తింపు",
        "detection_scale": "డిటెక్షన్ స్కేల్ ఫ్యాక్టర్",
        "min_neighbors": "కనిష్ట పొరుగువారు",
        "lighting_adjustments": "లైటింగ్ సర్దుబాట్లు",
        "enhance_contrast": "కాంట్రాస్ట్‌ను పెంచండి",
        "apply_denoising": "డీనాయిజింగ్ వర్తింపజేయండి",
        
        # Video Upload
        "upload_title": "విశ్లేషణ కోసం వీడియోను అప్‌లోడ్ చేయండి",
        "upload_instructions": "మంచి లైటింగ్‌లో మీ ముఖం యొక్క చిన్న వీడియో (10-30 సెకన్లు) అప్‌లోడ్ చేయండి",
        "video_requirements_title": "వీడియో అవసరాలు",
        "requirement_duration": "వ్యవధి: 10-30 సెకన్లు",
        "requirement_lighting": "ముఖంపై మంచి, సమాన లైటింగ్",
        "requirement_position": "ముఖం స్పష్టంగా కనిపించాలి మరియు కేంద్రీకృతంగా ఉండాలి",
        "requirement_movement": "తల కదలిక కనిష్టంగా ఉండాలి",
        "requirement_camera": "కెమెరా స్థిరంగా ఉండాలి",
        "upload_button": "వీడియో ఫైల్‌ను ఎంచుకోండి",
        "analyze_button": "వీడియోను విశ్లేషించండి",
        
        # Analysis Status
        "analyzing": "విశ్లేషిస్తోంది",
        "loading_video": "వీడియో లోడ్ అవుతోంది...",
        "detecting_face": "ముఖాన్ని గుర్తిస్తోంది...",
        "extracting_signal": "PPG సిగ్నల్‌ను సంగ్రహిస్తోంది...",
        "computing_vitals": "వైటల్స్‌ను గణిస్తోంది...",
        "generating_insights": "AI అంతర్దృష్టులను రూపొందిస్తోంది...",
        "analysis_complete": "విశ్లేషణ పూర్తయింది!",
        
        # Results
        "vital_signs": "ముఖ్యమైన సంకేతాలు",
        "estimated_pulse": "అంచనా పల్స్ (rPPG)",
        "stress_index": "ఒత్తిడి సూచిక (0-10)",
        "estimated_bp": "అంచనా BP",
        "estimated_spo2": "అంచనా SpO₂",
        "confidence": "విశ్వాసం",
        "experimental_stress": "ప్రయోగాత్మక ఒత్తిడి కొలత",
        
        # Risk Assessment
        "risk_assessment": "ప్రమాద అంచనా",
        "risk_score": "ప్రమాద స్కోర్",
        "low_risk": "తక్కువ ప్రమాదం",
        "moderate_risk": "మధ్యస్థ ప్రమాదం",
        "high_risk": "అధిక ప్రమాదం",
        "risk_factors": "ప్రమాద కారకాలు",
        "protective_factors": "రక్షణ కారకాలు",
        
        # Health Insights
        "health_insights_title": "ఆరోగ్య అంతర్దృష్టులు (AI-రూపొందించబడినవి)",
        "recommendations_title": "వ్యక్తిగత సిఫార్సులు",
        "symptoms_watch_title": "చూడవలసిన లక్షణాలు",
        
        # Signal Processing
        "signal_processing_title": "సిగ్నల్ ప్రాసెసింగ్ & విశ్లేషణ",
        "filtered_ppg": "ఫిల్టర్ చేయబడిన PPG సిగ్నల్ & పవర్ స్పెక్ట్రల్ డెన్సిటీ",
        "hrv_title": "హృదయ స్పందన వైవిధ్యం (RR అంతరాలు)",
        "rr_interval_analysis": "RR అంతరాల విశ్లేషణ",
        "hrv_summary": "HRV సారాంశం",
        
        # Historical View
        "viewing_historical": "చారిత్రక సెషన్‌ను చూస్తోంది | తిరిగి వెళ్ళడానికి 'కొత్త విశ్లేషణకు తిరిగి వెళ్ళండి' క్లిక్ చేయండి",
        "analysis_date": "విశ్లేషణ తేదీ",
        "back_to_new_analysis": "కొత్త విశ్లేషణకు తిరిగి వెళ్ళండి",
        
        # Trend Analysis
        "trends_title": "ఆరోగ్య ట్రెండ్ విశ్లేషణ",
        "trends_subtitle": "కాలక్రమేణా మీ ఆరోగ్య మెట్రిక్‌లను ట్రాక్ చేయండి మరియు నమూనాలను గుర్తించండి",
        "back_to_home": "హోమ్‌కు తిరిగి వెళ్ళండి",
        "days_7": "7 రోజులు",
        "days_14": "14 రోజులు",
        "days_30": "30 రోజులు",
        "days_90": "90 రోజులు",
        "summary": "సారాంశం",
        "key_findings": "ముఖ్య ఫలితాలు",
        "recommendations": "సిఫార్సులు",
        "metric_trends": "మెట్రిక్ ట్రెండ్‌లు",
        "average": "సగటు",
        "min": "కనిష్ట",
        "max": "గరిష్ట",
        "trend": "ట్రెండ్",
        "increasing": "పెరుగుతోంది",
        "decreasing": "తగ్గుతోంది",
        "stable": "స్థిరంగా",
        "status": "స్థితి",
        "improving": "మెరుగుపడుతోంది",
        "worsening": "చెడిపోతోంది",
        "concerning": "ఆందోళనకరం",
        "not_enough_data": "గత {days} రోజులలో ట్రెండ్ విశ్లేషణ కోసం తగినంత డేటా లేదు. ట్రెండ్‌లను అన్‌లాక్ చేయడానికి కనీసం 2 విశ్లేషణలను పూర్తి చేయండి.",
        
        # Chatbot
        "chatbot_title": "ఆరోగ్య సహాయకుడు",
        "chatbot_subtitle": "మీ ఆరోగ్య డేటా, ట్రెండ్‌లు లేదా సాధారణ ఆరోగ్య విషయాల గురించి ప్రశ్నలు అడగండి",
        "chatbot_expand": "విస్తరించండి",
        "chatbot_minimize": "చిన్నదిగా చేయండి",
        "chatbot_input_placeholder": "మీ ఆరోగ్య డేటా, ట్రెండ్‌లు లేదా సాధారణ ఆరోగ్య ప్రశ్నల గురించి నన్ను అడగండి...",
        "chatbot_clear": "చాట్‌ను క్లియర్ చేయండి",
        "chatbot_go_upload": "అప్‌లోడ్‌కు వెళ్ళండి",
        "suggested_questions": "సూచించిన ప్రశ్నలు:",
        "question_trends": "నా ట్రెండ్‌లు ఎలా ఉన్నాయి?",
        "question_heart_rate": "సాధారణ హృదయ స్పందన రేటు ఏమిటి?",
        "question_bp": "రక్తపోటును ఎలా తగ్గించాలి?",
        "question_stress": "ఒత్తిడిని తగ్గించే చిట్కాలు?",
        
        # Chatbot Disclaimer
        "chatbot_disclaimer": "🤖 **AI ఆరోగ్య సహాయకుడు నిరాకరణ:** ఈ చాట్‌బాట్ మీ డేటా ఆధారంగా సాధారణ ఆరోగ్య సమాచారం మరియు అంతర్దృష్టులను అందిస్తుంది. ఇది వృత్తిపరమైన వైద్య సలహా, రోగ నిర్ధారణ లేదా చికిత్సకు ప్రత్యామ్నాయం కాదు. వైద్య ఆందోళనల కోసం ఎల్లప్పుడూ అర్హత కలిగిన ఆరోగ్య సంరక్షణ ప్రదాతను సంప్రదించండి.",
        "escalation_message": "⚠️ **ముఖ్యమైనది:** మీ ప్రశ్న సంభావ్యంగా తీవ్రమైన ఆరోగ్య ఆందోళనను సూచిస్తుంది. దయచేసి వెంటనే ఆరోగ్య సంరక్షణ నిపుణుడిని సంప్రదించండి. ఇది అత్యవసర పరిస్థితి అయితే, మీ స్థానిక అత్యవసర సేవలకు కాల్ చేయండి.",
        
        # PDF Report
        "download_report": "నివేదికను డౌన్‌లోడ్ చేయండి",
        "generate_pdf": "PDF నివేదికను రూపొందించండి",
        "download_pdf": "PDF డౌన్‌లోడ్ చేయండి",
        "pdf_generated": "PDF విజయవంతంగా రూపొందించబడింది!",
        "pdf_error": "PDF రూపొందించడంలో లోపం",
        
        # Errors
        "error_session_not_found": "సెషన్ కనుగొనబడలేదు.",
        "error_no_video": "దయచేసి ముందుగా వీడియోను అప్‌లోడ్ చేయండి.",
        "error_analysis_failed": "విశ్లేషణ విఫలమైంది. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "error_invalid_video": "చెల్లని వీడియో ఫైల్. దయచేసి చెల్లుబాటు అయ్యే వీడియోను అప్‌లోడ్ చేయండి.",
        
        # Disclaimers
        "disclaimer_title": "⚠️ ముఖ్యమైన నిరాకరణ",
        "disclaimer_text": "ఇది ప్రయోగాత్మక పరిశోధన సాధనం. ఫలితాలు అంచనాలు మరియు వైద్య రోగ నిర్ధారణ లేదా చికిత్స నిర్ణయాల కోసం ఉపయోగించకూడదు. వైద్య సలహా కోసం ఎల్లప్పుడూ ఆరోగ్య సంరక్షణ నిపుణులను సంప్రదించండి.",
        
        # Pulse Labels
        "pulse_low": "తక్కువ",
        "pulse_slightly_low": "కొంచెం తక్కువ",
        "pulse_normal": "సాధారణ",
        "pulse_high": "అధిక",
        "pulse_very_high": "చాలా అధిక",
        
        # Upload Section
        "upload_video_title": "విశ్లేషణ కోసం వీడియోను అప్‌లోడ్ చేయండి",
        "upload_video_help": "మంచి లైటింగ్‌లో మీ ముఖం యొక్క చిన్న వీడియో (10-30 సెకన్లు) అప్‌లోడ్ చేయండి",
        "upload_instructions_title": "వీడియో అవసరాలు",
        
        # Live Recording
        "recording_mode_label": "రికార్డింగ్ పద్ధతిని ఎంచుకోండి",
        "recording_mode_upload": "📤 వీడియో అప్‌లోడ్ చేయండి",
        "recording_mode_live": "📹 లైవ్ రికార్డ్ చేయండి",
        "live_recording_title": "లైవ్ ఫేస్ రికార్డింగ్",
        "live_recording_subtitle": "ఉత్తమ ఫలితాల కోసం మీ ముఖాన్ని ఓవల్ గైడ్‌లో ఉంచండి",
        "complete_profile_warning": "దయచేసి ముందుగా మీ ప్రొఫైల్‌ను పూర్తి చేయండి",
        "fill_profile_sidebar": "కొనసాగించడానికి సైడ్‌బార్‌లో మీ ప్రొఫైల్‌ను పూరించండి",
        "profile_required_info": "ఖచ్చితమైన ఆరోగ్య విశ్లేషణ కోసం ప్రొఫైల్ సమాచారం అవసరం",
        "processing_frame": "ఫ్రేమ్ ప్రాసెస్ అవుతోంది",
        "complete": "పూర్తయింది",
        "video_processed_success": "వీడియో విజయవంతంగా ప్రాసెస్ చేయబడింది!",
        "start_new_analysis": "కొత్త విశ్లేషణను ప్రారంభించండి",
        
        # Analysis Results
        "typical_error": "సాధారణ లోపం",
        "estimated_bp_experimental": "అంచనా BP (ప్రయోగాత్మక)",
        "estimated_spo2_experimental": "అంచనా SpO₂ (ప్రయోగాత్మక)",
        
        # Risk Assessment Labels
        "risk_assessment_experimental": "ప్రమాద అంచనా (అనుమానిక, ప్రయోగాత్మక)",
        "risk_summary_low": "మీ ప్రమాద స్కోర్ తక్కువగా ఉంది. మీ ఆరోగ్యకరమైన జీవనశైలి అలవాట్లను కొనసాగించండి.",
        "risk_summary_moderate": "మీ ప్రమాద స్కోర్ మధ్యస్థంగా ఉంది. మీ మొత్తం ప్రమాదాన్ని తగ్గించడానికి పైన జాబితా చేయబడిన ప్రమాద కారకాలను మెరుగుపరచడాన్ని పరిగణించండి.",
        "risk_summary_high": "మీ ప్రమాద స్కోర్ అధికంగా ఉంది. మేము ఆరోగ్య సంరక్షణ నిపుణుడిని సంప్రదించమని మరియు పైన జాబితా చేయబడిన ప్రమాద కారకాలను పరిష్కరించమని గట్టిగా సిఫార్సు చేస్తున్నాము.",
        
        # Health Insights Labels
        "generating_insights": "ఆరోగ్య అంతర్దృష్టులను రూపొందిస్తోంది...",
        "insights_unavailable": "ఈ సమయంలో ఆరోగ్య అంతర్దృష్టులు అందుబాటులో లేవు.",
        "insights_module_unavailable": "ఆరోగ్య అంతర్దృష్టుల మాడ్యూల్ ప్రస్తుతం అందుబాటులో లేదు. ఈ ఫీచర్‌ను ఎనేబుల్ చేయడానికి మీ డిపెండెన్సీలను అప్‌డేట్ చేయండి.",
        "maintain_healthy_habits": "మీ ప్రస్తుత ఆరోగ్యకరమైన అలవాట్లను కొనసాగించండి.",
        "no_symptoms_watch": "ఈ సమయంలో చూడవలసిన నిర్దిష్ట లక్షణాలు లేవు.",
        
        # Signal Processing Labels
        "filtered_ppg_signal": "ఫిల్టర్ చేయబడిన PPG సిగ్నల్ (గ్రీన్ ఛానల్)",
        "frame": "ఫ్రేమ్",
        "normalized_intensity": "సాధారణీకరించిన తీవ్రత",
        "power_spectral_density": "పవర్ స్పెక్ట్రల్ డెన్సిటీ (వెల్చ్)",
        "frequency_hz": "ఫ్రీక్వెన్సీ (Hz)",
        "power_log_scale": "పవర్ (లాగ్ స్కేల్)",
        "peak_bpm": "శిఖరం",
        "rr_interval_distribution": "RR అంతరాల పంపిణీ",
        "rr_interval_ms": "RR అంతరం (ms)",
        "frequency": "ఫ్రీక్వెన్సీ",
        "rr_intervals_over_time": "కాలక్రమేణా RR అంతరాలు",
        "beat_number": "బీట్ #",
        "hrv_summary_label": "HRV సారాంశం",
        "beats_detected": "గుర్తించిన బీట్ల సంఖ్య",
        "sdnn_std_dev": "SDNN (ప్రామాణిక విచలనం)",
        "mean_rr": "సగటు RR",
        "pnn50": "pNN50",
        "not_enough_beats": "HRV విశ్లేషణ కోసం తగినంత బీట్లు గుర్తించబడలేదు. పొడవైన లేదా స్పష్టమైన వీడియోను ప్రయత్నించండి.",
        "advanced_signal_quality": "అధునాతన సిగ్నల్ నాణ్యత మెట్రిక్‌లు",
        "signal_to_noise_ratio": "సిగ్నల్-టు-నాయిస్ రేషియో (SNR)",
        "snr_help": "ఎక్కువ మంచిది. >2.0 = మంచిది, 1.0–2.0 = మధ్యస్థం, <1.0 = పేలవం",
        "quality_flags": "నాణ్యత ఫ్లాగ్‌లు",
        
        # Session Save & PDF
        "save_download": "సేవ్ చేసి డౌన్‌లోడ్ చేయండి",
        "session_saved": "సెషన్ చరిత్రలో సేవ్ చేయబడింది!",
        "could_not_save_session": "సెషన్‌ను చరిత్రలో సేవ్ చేయడం సాధ్యం కాలేదు",
        "generate_pdf_report": "PDF నివేదికను రూపొందించండి",
        "generating_pdf": "PDF రూపొందిస్తోంది...",
        "download_pdf_button": "PDF డౌన్‌లోడ్ చేయండి",
        
        # Error Messages
        "error_processing_video": "వీడియోను ప్రాసెస్ చేయడంలో లోపం",
        "troubleshooting": "సమస్య పరిష్కారం:",
        "ensure_good_lighting": "మంచి లైటింగ్‌ను నిర్ధారించండి (ప్రకాశవంతమైన వాతావరణం)",
        "keep_face_visible": "ముఖం కనిపించేలా మరియు సాపేక్షంగా స్థిరంగా ఉంచండి",
        "try_different_video": "వేరే లేదా చిన్న వీడియోను ప్రయత్నించండి",
        "ensure_video_format": "వీడియో ఫార్మాట్ MP4 లేదా MOV అని నిర్ధారించండి",
        "error_generating_insights": "అంతర్దృష్టులను రూపొందించడంలో లోపం",
        
        # Chatbot Labels
        "expand": "విస్తరించండి",
        "minimize": "చిన్నదిగా చేయండి",
        "chatbot_input_label": "చాట్ ఇన్‌పుట్",
        "go_to_upload": "అప్‌లోడ్‌కు వెళ్ళండి",
        "question_exercise": "క్రమం తప్పకుండా వ్యాయామం చేయడం వల్ల ఆరోగ్య ప్రయోజనాలు ఏమిటి?",
        
        # Historical View Labels
        "viewing_history": "చారిత్రక సెషన్‌ను చూస్తోంది",
        "back_to_new_analysis_instruction": "తిరిగి వెళ్ళడానికి 'కొత్త విశ్లేషణకు తిరిగి వెళ్ళండి' క్లిక్ చేయండి",
        "session_not_found": "సెషన్ కనుగొనబడలేదు.",
        "back_to_new_analysis_button": "కొత్త విశ్లేషణకు తిరిగి వెళ్ళండి",
        
        # Trend Analysis Labels  
        "analyzing_trends": "గత {days} రోజుల ట్రెండ్‌లను విశ్లేషిస్తోంది...",
        "not_enough_trend_data": "గత {days} రోజులలో ట్రెండ్ విశ్లేషణ కోసం తగినంత డేటా లేదు. ట్రెండ్‌లను అన్‌లాక్ చేయడానికి కనీసం 2 విశ్లేషణలను పూర్తి చేయండి.",
        "heart_rate_trend": "హృదయ స్పందన ట్రెండ్ - గత {period} రోజులు",
        "stress_level_trend": "ఒత్తిడి స్థాయి ట్రెండ్ - గత {period} రోజులు",
        "bp_trend": "రక్తపోటు ట్రెండ్ - గత {period} రోజులు",
        "spo2_trend": "ఆక్సిజన్ సంతృప్తత ట్రెండ్ - గత {period} రోజులు",
        "date": "తేదీ",
        "heart_rate_bpm": "హృదయ స్పందన రేటు (BPM)",
        "stress_level_scale": "ఒత్తిడి స్థాయి (0-10)",
        "systolic_bp_mmhg": "సిస్టోలిక్ BP (mmHg)",
        "spo2_percent": "SpO₂ (%)",
        "target": "లక్ష్యం",
        "normal_threshold": "సాధారణ పరిమితి",
        "trend_line": "ట్రెండ్",
        "increasing_arrow": "↑ పెరుగుతోంది",
        "decreasing_arrow": "↓ తగ్గుతోంది",
        "stable_arrow": "→ స్థిరంగా",
        
        # Sidebar Labels
        "history_sidebar_title": "చరిత్ర",
        "signal_processing_sidebar": "సిగ్నల్ ప్రాసెసింగ్",
        "face_detection_sidebar": "ముఖ గుర్తింపు",
        "lighting_adjustments_sidebar": "లైటింగ్ సర్దుబాట్లు",
        
        # Gender Options (using existing keys but adding for clarity)
        "prefer_not_say": "చెప్పకూడదనుకుంటున్నాను",
        "female": "స్త్రీ",
        "male": "పురుషుడు",
        "other": "ఇతర",
        
        # Diet Options (using existing keys)
        "non_vegetarian": "మాంసాహారం",
        "vegetarian": "శాకాహారం",
        "vegan": "శుద్ధ శాకాహారం",
        
        # Exercise Options (using existing keys)
        "never": "ఎప్పుడూ లేదు",
        "exercise_1_2": "1-2 సార్లు/వారం",
        "exercise_3_4": "3-4 సార్లు/వారం",
        "daily": "ప్రతిరోజూ",
        
        # Smoking/Drinking Options (using existing keys)
        "occasional": "అప్పుడప్పుడు",
        "regular": "క్రమం తప్పకుండా",
        "former": "గతంలో",
        
        # Misc
        "na": "వర్తించదు",
        "unknown": "తెలియదు",
        "loading": "లోడ్ అవుతోంది...",
        "processing": "ప్రాసెస్ అవుతోంది...",
        "please_wait": "దయచేసి వేచి ఉండండి...",
        # Audio Summary
        "audio_intro": "ఇక్కడ మీ ఆరోగ్య సారాంశం ఉంది.",
        "audio_hr": "మీ అంచనా హృదయ స్పందన రేటు నిమిషానికి {value} బీట్స్.",
        "audio_stress": "ఒత్తిడి స్థాయి 10కి {value}.",
        "audio_bp": "అంచనా వేసిన రక్తపోటు {systolic} బై {diastolic}.",
        "audio_spo2": "ఆక్సిజన్ సంతృప్తత {value} శాతం.",
        "audio_risk": "మీ ప్రమాద అంచనా స్కోరు 10కి {score}. ఇది {level}.",
        "audio_insights_intro": "ఇక్కడ కొన్ని అంతర్దృష్టులు ఉన్నాయి.",
        "audio_recs": "సిఫార్సులు: ",
        "audio_symptoms": "చూడవలసిన లక్షణాలు: ",
    },
}


def get_text(key: str, lang: str = "en") -> str:
    """
    Get translated text for a given key and language.
    
    Args:
        key: Translation key
        lang: Language code ('en', 'hi', 'te')
    
    Returns:
        Translated text, falls back to English if key not found
    """
    # Validate language
    if lang not in LANGUAGES:
        lang = "en"
    
    # Get translation, fallback to English if not found
    try:
        return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))
    except:
        return key


def get_available_languages() -> dict:
    """
    Get list of available languages with metadata.
    
    Returns:
        Dictionary of language codes with names and flags
    """
    return LANGUAGES


def translate_dynamic(text: str, target_lang: str, api_key: str) -> str:
    """
    Translate dynamic AI-generated content using Groq API.
    
    Args:
        text: Text to translate (in English)
        target_lang: Target language code ('hi', 'te')
        api_key: Groq API key
    
    Returns:
        Translated text, falls back to original if translation fails
    """
    # If target is English, return as-is
    if target_lang == "en":
        return text
    
    # If text is empty, return as-is
    if not text or not text.strip():
        return text
    
    try:
        # Get language name
        lang_name = LANGUAGES.get(target_lang, {}).get("name", target_lang)
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Create translation prompt
        prompt = f"""Translate the following health-related text from English to {lang_name}.
Maintain the tone, formatting, and any special characters (like bullet points).
Keep medical terms accurate and culturally appropriate.

Text to translate:
{text}

Provide ONLY the translation, no explanations or additional text."""
        
        # Call Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,  # Lower temperature for more consistent translations
            max_tokens=2000,
        )
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text
        
    except Exception as e:
        print(f"Translation error: {e}")
        # Gracefully fall back to original text
        return text


def format_with_params(text: str, **kwargs) -> str:
    """
    Format translation text with parameters.
    
    Args:
        text: Text with placeholders like {days}
        **kwargs: Parameters to substitute
    
    Returns:
        Formatted text
    """
    try:
        return text.format(**kwargs)
    except:
        return text
