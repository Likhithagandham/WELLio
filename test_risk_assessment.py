"""
Test Suite for Risk Assessment Module
"""
import unittest
from risk_assessment import RiskInput, calculate_risk

class TestRiskAssessment(unittest.TestCase):
    
    def test_baseline_normal(self):
        """Test a perfectly healthy individual."""
        data = RiskInput(
            heart_rate=70, hrv_sdnn=50, stress_score=2,
            systolic_bp=115, diastolic_bp=75, spo2=98,
            signal_confidence=95,
            age=25, sex="Male", activity_level="Moderate", is_smoker=False
        )
        res = calculate_risk(data)
        self.assertEqual(res.risk_score, 0.0)
        self.assertEqual(res.risk_level, "LOW")
        
    def test_missing_data_safety(self):
        """Test that missing data does not crash and contributes 0 risk."""
        data = RiskInput(signal_confidence=80, age=25) 
        # All vitals None by default
        res = calculate_risk(data)
        self.assertEqual(res.risk_score, 0.0) # 0 vitals + 0 profile (default age 30)
        
    def test_high_risk_vitals_capping(self):
        """Test that risk score is clamped at 10."""
        data = RiskInput(
            heart_rate=190, # +4
            hrv_sdnn=5,     # +3
            spo2=85,        # +4
            stress_score=9, # +1
            # Total raw = 12
            signal_confidence=90
        )
        res = calculate_risk(data)
        self.assertEqual(res.risk_score, 10.0)
        self.assertEqual(res.risk_level, "HIGH")

    def test_athlete_logic(self):
        """Test athlete modifier logic."""
        # non-athlete low HR
        data1 = RiskInput(heart_rate=45, is_athlete=False, age=25)
        res1 = calculate_risk(data1)
        self.assertEqual(res1.risk_score, 2.0) # +2 for 35-49 BPM
        
        # athlete low HR
        data2 = RiskInput(heart_rate=45, is_athlete=True, age=25)
        res2 = calculate_risk(data2)
        self.assertEqual(res2.risk_score, 0.0) # waived
        
        # athlete high HR (reduced risk)
        data3 = RiskInput(heart_rate=130, is_athlete=False, age=25) # 121-140 is +2
        res3 = calculate_risk(data3)
        self.assertEqual(res3.risk_score, 2.0)
        
        data4 = RiskInput(heart_rate=130, is_athlete=True, age=25) # +2 reduced by 1
        res4 = calculate_risk(data4)
        self.assertEqual(res4.risk_score, 1.0) 

    def test_profile_capping(self):
        """Test that profile modifiers are capped at +/- 2."""
        # Age 70 (+2), Smoker (+1), Conditions (+2) -> Raw +5
        # Expected max +2
        data = RiskInput(
            age=70, sex="Male", is_smoker=True, 
            medical_conditions=["hypertension", "diabetes"],
            heart_rate=70 # vital 0
        )
        res = calculate_risk(data)
        # Profile score logic: 
        # Age >60: +2
        # Smoker: +1
        # Conditions: +2
        # Raw Profile: 5
        # Capped Profile: 2
        
        # NOTE: calculate_risk implementation has a subtle bug potential if I imply 
        # that individual sections are capped vs total.
        # Let's check my implementation:
        # Age 70 -> +2 (line 123 in risk_assessment.py, wait I can't see lines but logic was: >60 -> +2)
        # Smoker -> +1
        # Conditions -> +2
        # Sum = 5. 
        # Logic: if profile_score > 2.0: profile_score = 2.0
        # So final should be 2.0
        self.assertEqual(res.risk_score, 2.0)
        
    def test_confidence_gating(self):
        """Test that High Risk is downgraded if confidence is low."""
        data = RiskInput(
            heart_rate=190, # +4
            spo2=85,        # +4
            # Total 8 -> HIGH
            signal_confidence=40 # < 55
        )
        res = calculate_risk(data)
        self.assertEqual(res.risk_level, "MODERATE")
        self.assertIn("Low signal confidence", str(res.alerts))

if __name__ == "__main__":
    unittest.main()
