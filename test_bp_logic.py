
from rppg_refactored import calculate_bp, BPResult
import sys

def test_bp():
    print("Testing BP Logic...")
    
    # CASE 1: Normal Base (HR=70, HRV=40, Stress=None, RR=800)
    # Base: 115/75. Updates: None. Output: 115/75 (Normal)
    res = calculate_bp(70, 40, None, 800, 100)
    print(f"Normal: {res.systolic}/{res.diastolic} Label={res.label} Expect=115/75 Normal")
    assert res.systolic == 115
    assert res.diastolic == 75
    assert "Normal" in res.label

    # CASE 2: High HR + Low HRV + High Stress (Max modifiers)
    # Base: 115/75
    # HR > 100 (110) -> +10/+5 -> 125/80
    # HRV < 20 (15) -> +10 SBP -> 135/80
    # Stress HIGH -> +15/+8 -> 150/88
    # RR < 500 (450) -> +10 SBP -> 160/88
    # Result: 160/88. High (>140 SBP).
    res = calculate_bp(110, 15, "HIGH", 450, 100)
    print(f"Max Modifiers: {res.systolic}/{res.diastolic} Label={res.label} Expect=160/88 High")
    assert res.systolic == 160
    assert res.diastolic == 88
    assert "High" in res.label

    # CASE 3: Low BP (Low HR)
    # Base: 115/75
    # HR < 50 (45) -> -10/-5 -> 105/70
    # Result: 105/70. Normal range? No, Normal is 90-120 AND 60-80. 105 is in 90-120. 70 is in 60-80. So Normal.
    # Wait, let's force Low. SBP < 90.
    # Need -25 drop.
    # HR < 50 -> -10. SBP=105.
    # Let's say we had a modifier for "Relaxed"? No, only Stress adds.
    # Base is 115. Can we hit <90?
    # Only modifier decreasing is Low HR (-10). 
    # So max low is 105/70.
    # UNLESS logic allows negative modifiers elsewhere? No.
    # Ah, clamping handles 80-200.
    # Actually, is there any way to get "Low" with current rules?
    # Rules: HR < 50 -> -10/-5.
    # That's it. So strict Low (<90) might differ from standard 115.
    # Let's test what we have. 105/70 should be Normal.
    res = calculate_bp(45, 40, "NORMAL", 800, 100)
    print(f"Low HR: {res.systolic}/{res.diastolic} Label={res.label} Expect=105/70 Normal")
    assert res.systolic == 105
    assert res.diastolic == 70
    assert "Normal" in res.label

    # CASE 4: High-Normal
    # SBP 120-140.
    # Base 115. Need +5.
    # Mild Stress (+5/+3) -> 120/78.
    # Label: High-Normal (120-140).
    res = calculate_bp(70, 40, "MILD", 800, 100)
    print(f"High-Normal: {res.systolic}/{res.diastolic} Label={res.label} Expect=120/78 High-Normal")
    assert res.systolic == 120
    assert res.diastolic == 78
    assert "High-Normal" in res.label

    # CASE 5: Low Confidence
    res = calculate_bp(70, 40, None, 800, 40)
    print(f"Low Conf: {res.reason}")
    assert "unreliable" in res.reason.lower()

    print("✅ All Tests Passed!")

if __name__ == "__main__":
    try:
        test_bp()
    except AssertionError as e:
        print(f"❌ Test Failed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
