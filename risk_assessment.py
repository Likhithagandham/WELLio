"""
Heuristic Risk Assessment Module
================================

This module implements a rule-based, heuristic risk assessment system based on
estimated vital signs and user profile data.

⚠️ IMPORTANT:
- This system is NOT a medical device.
- It is non-diagnostic and intended for informational purposes only.
- Logic is deterministic and explainable.
- Missing inputs contribute 0 risk points (fail-safe).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class RiskInput:
    """Structured input for risk assessment."""
    # Vitals (Optional to handle missing data gracefully)
    heart_rate: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    stress_score: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    spo2: Optional[float] = None
    signal_confidence: float = 0.0
    
    # Profile (Defaults provided for robustness)
    age: int = 30
    sex: str = "Male" # "Male", "Female", "Other"
    activity_level: str = "Moderate" # "Low", "Moderate", "High"
    is_athlete: bool = False
    is_smoker: bool = False
    medical_conditions: List[str] = field(default_factory=list)

@dataclass
class RiskResult:
    """Structured output for risk assessment."""
    risk_score: float
    risk_level: str # "LOW", "MODERATE", "HIGH"
    alerts: List[str]
    profile_factors_used: List[str]
    recommendation: str
    confidence_percent: float
    disclaimer: str

def calculate_risk(input_data: RiskInput) -> RiskResult:
    """
    Calculates a heuristic risk score (0-10) based on vitals and profile.
    
    Rules:
    - Vitals contribute base risk points.
    - Profile factors modify the score (capped at +/- 2).
    - Missing vitals contribute 0 risk.
    - Final score is clamped between 0 and 10.
    - logic is deterministic.
    """
    
    alerts = []
    factors = []
    
    # --- 1. Vital Signs Scoring ---
    vital_points = 0.0
    
    # Heart Rate (BPM)
    hr_points = 0.0
    if input_data.heart_rate is not None:
        hr = input_data.heart_rate
        if hr < 35:
            hr_points = 4.0
            alerts.append(f"Critical Bradycardia Detected ({hr:.0f} BPM)")
        elif 35 <= hr <= 49:
            hr_points = 2.0
            alerts.append(f"Low Heart Rate ({hr:.0f} BPM)")
        elif 50 <= hr <= 120:
            hr_points = 0.0
        elif 121 <= hr <= 140:
            hr_points = 2.0
            alerts.append(f"Elevated Heart Rate ({hr:.0f} BPM)")
        elif 141 <= hr <= 180:
            hr_points = 3.0
            alerts.append(f"High Heart Rate ({hr:.0f} BPM)")
        else: # > 180
            hr_points = 4.0
            alerts.append(f"Critical Tachycardia Detected ({hr:.0f} BPM)")
            
        # Athlete Modifier for HR
        if input_data.is_athlete:
            if hr < 50:
                # Do not penalize low HR for athletes
                if hr_points > 0:
                     factors.append("Athlete: Low HR penalty waived")
                hr_points = 0.0
            else:
                # Reduce HR risk by 1 for athletes if not already 0
                if hr_points > 0:
                    hr_points = max(0.0, hr_points - 1.0)
                    factors.append("Athlete: HR risk reduced")

        vital_points += hr_points

    # HRV (SDNN in ms)
    if input_data.hrv_sdnn is not None:
        hrv = input_data.hrv_sdnn
        if hrv < 10:
            vital_points += 3.0
            alerts.append("Critically Low HRV")
        elif 10 <= hrv <= 19:
            vital_points += 2.0
            alerts.append("Very Low HRV")
        elif 20 <= hrv <= 29:
            vital_points += 1.0
            alerts.append("Low HRV")
        else: # >= 30
            pass # 0 points

    # Stress Score (0-10)
    if input_data.stress_score is not None:
        if input_data.stress_score > 8:
            vital_points += 1.0
            alerts.append("High Stress Detected")
    
    # Experimental BP
    if input_data.systolic_bp is not None and input_data.diastolic_bp is not None:
        sbp = input_data.systolic_bp
        dbp = input_data.diastolic_bp
        
        if sbp < 90 or dbp < 60:
            vital_points += 2.0
            alerts.append("Low Blood Pressure")
        elif sbp > 180 or dbp > 110:
            vital_points += 4.0
            alerts.append("Critical High Blood Pressure")
        elif (160 <= sbp <= 180) or (100 <= dbp <= 110):
            vital_points += 2.0
            alerts.append("High Blood Pressure")
        else:
            pass # 0 points

    # Experimental SpO2
    if input_data.spo2 is not None:
        spo2 = input_data.spo2
        if spo2 < 90:
            vital_points += 4.0
            alerts.append(f"Critical Low SpO2 ({spo2:.1f}%)")
        elif 90 <= spo2 <= 94:
            vital_points += 2.0
            alerts.append(f"Low SpO2 ({spo2:.1f}%)")
        else:
            pass # 0 points

    # --- 2. Profile Modifiers ---
    profile_score = 0.0
    
    # Age
    age = input_data.age
    if age < 30:
        pass # 0
    elif 30 <= age <= 45:
        profile_score += 0.5
        factors.append(f"Age {age} (+0.5)")
    elif 46 <= age <= 60:
        profile_score += 1.0
        factors.append(f"Age {age} (+1.0)")
    else: # > 60
        profile_score += 2.0
        factors.append(f"Age {age} (+2.0)")

    # Sex
    sex = input_data.sex.lower()
    if sex == "female":
        profile_score -= 0.5
        factors.append("Sex: Female (-0.5)")
    
    # Activity Level
    activity = input_data.activity_level.lower()
    if activity == "high":
        profile_score -= 1.0
        factors.append("Activity: High (-1.0)")
    elif activity == "low":
        profile_score += 1.0
        factors.append("Activity: Low (+1.0)")
    
    # Smoking
    if input_data.is_smoker:
        profile_score += 1.0
        factors.append("Smoker (+1.0)")
        
    # Known Conditions
    condition_points = 0.0
    recognized_conditions = ["hypertension", "diabetes", "cardiac_history", "respiratory_condition"]
    for condition in input_data.medical_conditions:
        norm_cond = condition.lower().replace(" ", "_").strip()
        if norm_cond in recognized_conditions:
            condition_points += 1.0
            factors.append(f"Condition: {condition} (+1.0)")
    
    # Cap conditions at +2
    if condition_points > 2.0:
        condition_points = 2.0
        factors.append("Conditions capped at +2.0")
    profile_score += condition_points

    # Clamp Total Profile Modifiers to +/- 2 points
    if profile_score > 2.0:
        profile_score = 2.0
        factors.append("Profile impact capped at +2.0")
    elif profile_score < -2.0:
        profile_score = -2.0
        factors.append("Profile impact capped at -2.0")

    # --- 3. Final Calculation ---
    final_score = vital_points + profile_score
    
    # Clamp Final Score [0, 10]
    final_score = max(0.0, min(10.0, final_score))
    
    # Risk Level Determination
    level = "LOW"
    if final_score >= 7.0:
        level = "HIGH"
    elif final_score >= 3.0:
        level = "MODERATE"
    else:
        level = "LOW"
        
    # --- 4. Signal Confidence Gating ---
    if input_data.signal_confidence < 55.0:
        alerts.append("Low signal confidence - Risk assessment reliability reduced.")
        if level == "HIGH":
            level = "MODERATE"
            alerts.append("Risk downgraded due to low signal confidence.")
            
    # Recommendations
    rec = ""
    if level == "LOW":
        rec = "Your metrics appear within a healthy range. Maintain your current healthy habits."
    elif level == "MODERATE":
        rec = "Some metrics are outside the optimal range. Consider monitoring your vitals and improving lifestyle habits."
    else: # HIGH
        rec = "Multiple metrics are significantly outside normal ranges. Please consult a healthcare professional for verification."

    return RiskResult(
        risk_score=round(final_score, 1),
        risk_level=level,
        alerts=alerts,
        profile_factors_used=factors,
        recommendation=rec,
        confidence_percent=input_data.signal_confidence,
        disclaimer="This assessment is experimental, non-diagnostic, and intended for informational purposes only. It should not be used for medical decision-making."
    )
