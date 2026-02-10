
from rppg_refactored import get_bp_category
import sys

def test_bp_labeling():
    print("Testing BP Labeling Logic (Severity Based)...")
    
    # 1. Normal SBP + High DBP
    # SBP 118 (Normal=1), DBP 86 (High-Normal=2) -> Max(1,2) = 2 (High-Normal)
    lbl = get_bp_category(118, 86)
    print(f"118/86 -> {lbl} (Expect: ⬆️ High-Normal)")
    assert "High-Normal" in lbl

    # 2. High SBP + Normal DBP
    # SBP 142 (High=3), DBP 78 (Normal=1) -> Max(3,1) = 3 (High)
    lbl = get_bp_category(142, 78)
    print(f"142/78 -> {lbl} (Expect: ⬆️ High)")
    assert "High" in lbl and "Normal" not in lbl

    # 3. Low SBP + High DBP
    # SBP 88 (Low=0), DBP 92 (High=3) -> Max(0,3) = 3 (High)
    lbl = get_bp_category(88, 92)
    print(f"88/92 -> {lbl} (Expect: ⬆️ High)")
    assert "High" in lbl

    # 4. Normal SBP + Low DBP
    # SBP 92 (Normal=1), DBP 58 (Low=0) -> Max(1,0) = 1 (Normal)
    lbl = get_bp_category(92, 58)
    print(f"92/58 -> {lbl} (Expect: ✅ Normal)")
    assert "Normal" in lbl

    # 5. Missing SBP
    # DBP 95 (High=3) -> High
    lbl = get_bp_category(None, 95)
    print(f"None/95 -> {lbl} (Expect: ⬆️ High)")
    assert "High" in lbl

    # 6. Missing DBP
    # SBP 130 (High-Normal=2) -> High-Normal
    lbl = get_bp_category(130, None)
    print(f"130/None -> {lbl} (Expect: ⬆️ High-Normal)")
    assert "High-Normal" in lbl

    # 7. Missing Both
    lbl = get_bp_category(None, None)
    print(f"None/None -> {lbl} (Expect: Unavailable)")
    assert lbl == "Unavailable"
    
    # 8. Edge Case: 119/79 (Top of Normal) -> Normal
    lbl = get_bp_category(119, 79)
    print(f"119/79 -> {lbl} (Expect: ✅ Normal)")
    assert "Normal" in lbl

    # 9. Edge Case: 120/80 (Bottom of High-Normal) -> High-Normal
    lbl = get_bp_category(120, 80)
    print(f"120/80 -> {lbl} (Expect: ⬆️ High-Normal)")
    assert "High-Normal" in lbl

    print("✅ All Tests Passed!")

if __name__ == "__main__":
    try:
        test_bp_labeling()
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
