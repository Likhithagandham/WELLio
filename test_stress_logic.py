
from rppg_refactored import calculate_stress_score, StressResult
import sys

def test_stress():
    print("Testing Stress Logic...")
    
    # CASE 1: Normal
    res = calculate_stress_score(hrv_sdnn=40, rr_interval_ms=800)
    print(f"Normal (40ms, 800ms): {res.label} Score={res.score} Expect=2")
    assert res.score == 2
    assert res.label == "Normal"
    
    # CASE 2: Mild (Low HRV)
    res = calculate_stress_score(hrv_sdnn=25, rr_interval_ms=800)
    print(f"Mild (Low HRV 25ms, Normal RR): {res.label} Score={res.score} Expect=5")
    assert res.score == 5
    assert res.label == "Mild Stress"
    
    # CASE 3: Mild (Elevated RR)
    res = calculate_stress_score(hrv_sdnn=40, rr_interval_ms=550)
    print(f"Mild (Normal HRV, Elevated RR 550ms): {res.label} Score={res.score} Expect=5")
    assert res.score == 5
    assert res.label == "Mild Stress"

    # CASE 4: High (Both Bad)
    res = calculate_stress_score(hrv_sdnn=15, rr_interval_ms=450)
    print(f"High (Very Low HRV 15ms, High RR 450ms): {res.label} Score={res.score} Expect=8")
    assert res.score == 8
    assert res.label == "High Stress"
    
    # CASE 5: Missing Data
    res = calculate_stress_score(None, None)
    print(f"Missing: {res.label}")
    assert res.label == "Unavailable"
    
    # CASE 6: Partial (HRV Only - Low)
    res = calculate_stress_score(25, None)
    print(f"Partial HRV(Low): {res.label} Score={res.score} Expect=5")
    assert res.score == 5
    
    print("✅ All Tests Passed!")

if __name__ == "__main__":
    try:
        test_stress()
    except AssertionError as e:
        print(f"❌ Test Failed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
