"""
Risk Assessment Module
======================

Implements a heuristic, benchmark-based risk scoring system for health vitals.
The system is non-diagnostic and serves informational purposes only.
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class RiskInput:
    # Vitals
    heart_rate: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    stress_score: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    spo2: Optional[float] = None
    
    # User Profile
    age: Optional[int] = None
    sex: Optional[str] = None # "Female", "Male", "Other"
    activity_level: Optional[str] = None # "High", "Moderate", "Low"
    resting_athlete: bool = False
    is_smoker: bool = False
    conditions: List[str] = field(default_factory=list) # e.g., ["hypertension", "diabetes"]
    
    # Metadata
    signal_confidence: float = 100.0 # 0-100%

@dataclass
class RiskResult:
    risk_score: float
    risk_level: str # LOW, MODERATE, HIGH
    alerts: List[str]
    profile_factors_used: List[str]
    recommendation: str
    confidence_percent: float
    disclaimer: str

def calculate_risk(inputs: RiskInput) -> RiskResult:
    """
    Calculates a heuristic risk score (0-10) based on vitals and profile data.
    """
    vital_points = 0.0
    profile_points = 0.0
    alerts = []
    factors_used = []
    
    # ========================================================================
    # 1. VITAL BENCHMARKS
    # ========================================================================
    
    # Heart Rate (BPM)
    if inputs.heart_rate is not None:
        hr = inputs.heart_rate
        if inputs.resting_athlete and hr < 50:
            # Athlete Flag: Do NOT penalize HR < 50, and reduce HR risk by -1
            # (Though if HR is < 35, the +4 would become +3)
            if hr < 35:
                vital_points += 3 # 4 - 1
                alerts.append("Very low heart rate detected (Athlete adjusted)")
            else:
                # No penalty for 35-49 if athlete
                pass
        else:
            if hr < 35:
                vital_points += 4
                alerts.append("Critically low heart rate detected")
            elif hr < 50:
                vital_points += 2
                alerts.append("Low heart rate detected")
            elif 120 < hr <= 140:
                vital_points += 2
                alerts.append("Elevated heart rate detected")
            elif 140 < hr <= 180:
                vital_points += 3
                alerts.append("High heart rate detected")
            elif hr > 180:
                vital_points += 4
                alerts.append("Critically high heart rate detected")

    # HRV (SDNN in ms)
    if inputs.hrv_sdnn is not None:
        sdnn = inputs.hrv_sdnn
        if sdnn < 10:
            vital_points += 3
            alerts.append("Very low HRV detected")
        elif sdnn < 20:
            vital_points += 2
            alerts.append("Low HRV detected")
        elif sdnn < 30:
            vital_points += 1
            alerts.append("Slightly low HRV detected")

    # Stress Score (0-10)
    if inputs.stress_score is not None:
        if inputs.stress_score > 8:
            vital_points += 1
            alerts.append("High stress level detected")

    # Experimental Blood Pressure
    s = inputs.systolic_bp
    d = inputs.diastolic_bp
    if s is not None or d is not None:
        # SBP < 90 or DBP < 60 -> +2
        is_low = (s is not None and s < 90) or (d is not None and d < 60)
        # SBP 160-180 or DBP 100-110 -> +2
        is_high_s1 = (s is not None and 160 <= s <= 180) or (d is not None and 100 <= d <= 110)
        # SBP > 180 or DBP > 110 -> +4
        is_high_s2 = (s is not None and s > 180) or (d is not None and d > 110)
        
        if is_high_s2:
            vital_points += 4
            alerts.append("Critically high blood pressure detected")
        elif is_high_s1:
            vital_points += 2
            alerts.append("High blood pressure detected")
        elif is_low:
            vital_points += 2
            alerts.append("Low blood pressure detected")

    # Experimental SpO2 (%)
    if inputs.spo2 is not None:
        if inputs.spo2 < 90:
            vital_points += 4
            alerts.append("Critically low SpO2 detected")
        elif inputs.spo2 < 95:
            vital_points += 2
            alerts.append("Low SpO2 detected")

    # ========================================================================
    # 2. USER PROFILE MODIFIERS
    # ========================================================================
    
    # Age
    if inputs.age is not None:
        factors_used.append(f"Age: {inputs.age}")
        if 30 <= inputs.age <= 45: profile_points += 0.5
        elif 46 <= inputs.age <= 60: profile_points += 1.0
        elif inputs.age > 60: profile_points += 2.0
        
    # Sex
    if inputs.sex == "Female":
        profile_points -= 0.5
        factors_used.append("Sex: Female")
        
    # Activity Level
    if inputs.activity_level == "High":
        profile_points -= 1.0
        factors_used.append("Activity: High")
    elif inputs.activity_level == "Low":
        profile_points += 1.0
        factors_used.append("Activity: Low")
        
    # Smoking
    if inputs.is_smoker:
        profile_points += 1.0
        factors_used.append("Smoker")
        
    # Known Conditions
    condition_points = 0.0
    recognized = ["hypertension", "diabetes", "cardiac_history", "respiratory_condition"]
    for c in inputs.conditions:
        if c.lower() in recognized:
            condition_points += 1.0
            factors_used.append(f"Condition: {c}")
    
    # Cap conditions to +2
    profile_points += min(2.0, condition_points)
    
    # Total impact from all profile modifiers combined must not exceed ±2 points
    profile_points = max(-2.0, min(2.0, profile_points))
    
    # ========================================================================
    # 3. FINAL SYNTHESIS
    # ========================================================================
    
    raw_score = vital_points + profile_points
    # Clamp strictly between 0 and 10
    final_score = max(0.0, min(10.0, raw_score))
    
    # Risk Levels: 0-2 LOW, 3-6 MODERATE, 7+ HIGH
    if final_score >= 7:
        risk_level = "HIGH"
    elif final_score >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
        
    # SIGNAL CONFIDENCE GATING
    if inputs.signal_confidence < 55:
        alerts.append("Low signal confidence")
        if risk_level == "HIGH":
            risk_level = "MODERATE"
            
    # Recommendations
    if risk_level == "LOW":
        recommendation = "Maintain your current healthy habits. Continue regular monitoring."
    elif risk_level == "MODERATE":
        recommendation = "Consider reviewing your lifestyle factors. Ensure you are getting adequate rest and managing stress."
    else:
        recommendation = "Multiple indicators are outside optimal ranges. Please rest and consider re-taking the assessment in a calm state."

    disclaimer = ("This assessment is experimental, non-diagnostic, and intended for informational purposes only. "
                  "It should not be used for medical decision-making.")

    return RiskResult(
        risk_score=round(final_score, 1),
        risk_level=risk_level,
        alerts=alerts,
        profile_factors_used=factors_used,
        recommendation=recommendation,
        confidence_percent=inputs.signal_confidence,
        disclaimer=disclaimer
    )

if __name__ == "__main__":
    # Example usage
    example_input = RiskInput(
        heart_rate=125,
        hrv_sdnn=15,
        systolic_bp=165,
        is_smoker=True,
        activity_level="Low",
        age=50,
        signal_confidence=90.0
    )
    
    result = calculate_risk(example_input)
    print(f"Risk Score: {result.risk_score}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Alerts: {result.alerts}")
    print(f"Profile Factors: {result.profile_factors_used}")
    print(f"Recommendation: {result.recommendation}")
