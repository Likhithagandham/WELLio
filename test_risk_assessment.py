from risk_assessment import calculate_risk, RiskInput

def test_risk_cases():
    print("Testing Risk Assessment Module...\n")

    # Case 1: Normal Vitals, Healthy Profile
    input1 = RiskInput(
        heart_rate=70,
        hrv_sdnn=45,
        stress_score=2,
        systolic_bp=115,
        diastolic_bp=75,
        spo2=98,
        age=25,
        activity_level="High"
    )
    res1 = calculate_risk(input1)
    print(f"Case 1 (Normal): Score={res1.risk_score}, Level={res1.risk_level}")
    assert res1.risk_level == "LOW"

    # Case 2: High Risk Vitals (Critically High HR, Low SpO2, Smoker)
    input2 = RiskInput(
        heart_rate=185, # +4
        spo2=88,        # +4
        is_smoker=True,  # +1 (Profile)
        age=65          # +2 (Profile)
    )
    # Total = 8 + (capped 2) = 10
    res2 = calculate_risk(input2)
    print(f"Case 2 (High Risk): Score={res2.risk_score}, Level={res2.risk_level}")
    assert res2.risk_level == "HIGH"

    # Case 3: High Risk but Low Confidence Gate
    input3 = RiskInput(
        heart_rate=185, # +4
        spo2=88,        # +4
        signal_confidence=50.0 # Should downgrade HIGH to MODERATE
    )
    res3 = calculate_risk(input3)
    print(f"Case 3 (Confidence Gating): Score={res3.risk_score}, Level={res3.risk_level}")
    assert res3.risk_level == "MODERATE"
    assert "Low signal confidence" in res3.alerts

    # Case 4: Athlete Modifier
    input4 = RiskInput(
        heart_rate=45,
        resting_athlete=True # Should not penalize HR 45
    )
    res4 = calculate_risk(input4)
    print(f"Case 4 (Athlete): Score={res4.risk_score}, Level={res4.risk_level}")
    assert res4.risk_score == 0.0 # No penalty

    # Case 5: Capped Profile Modifiers
    input5 = RiskInput(
        age=70,           # +2
        is_smoker=True,   # +1
        activity_level="Low", # +1
        conditions=["hypertension", "diabetes"] # +2
    )
    # Profile points sum to 6, but must be capped at 2.0
    res5 = calculate_risk(input5)
    print(f"Case 5 (Profile Cap): Score={res5.risk_score}, Level={res5.risk_level}")
    assert res5.risk_score == 2.0

    print("\n✅ All Risk Tests Passed!")

if __name__ == "__main__":
    test_risk_cases()
