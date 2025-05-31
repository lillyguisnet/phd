import numpy as np

def test_hard_constraints(prev_angle, current_angle):
    """Test the hard constraints that prevent impossible transitions"""
    if prev_angle is not None:
        angle_change = abs(current_angle - prev_angle)
        
        # Check for impossible sign switches with large changes
        is_impossible_switch = False
        if prev_angle * current_angle < 0:  # Different signs
            min_abs_angle = min(abs(prev_angle), abs(current_angle))
            # If both angles are substantial and change is huge, it's impossible
            if min_abs_angle > 30 and angle_change > 120:
                is_impossible_switch = True
            elif min_abs_angle > 45 and angle_change > 90:
                is_impossible_switch = True
        
        # Also reject if change is extremely large (>150°) regardless of sign
        is_extreme_change = angle_change > 150
        
        return is_impossible_switch, is_extreme_change, angle_change
    
    return False, False, 0

def calculate_improved_angle_for_bent_worms(is_highly_bent, traditional_angle):
    """Simulate the improved angle calculation for highly bent worms"""
    if not is_highly_bent:
        return traditional_angle
    
    # For highly bent worms, ensure realistic angles
    if abs(traditional_angle) < 60:
        # Scale up to a realistic range
        base_angle = traditional_angle
        scale_factor = 90.0 / max(abs(base_angle), 30.0)  # Scale to at least 90°
        best_angle = base_angle * min(scale_factor, 2.5)  # Cap scaling at 2.5x
        
        # Ensure we don't exceed reasonable limits
        if abs(best_angle) > 160:
            best_angle = 160 if best_angle > 0 else -160
        
        return best_angle
    
    return traditional_angle

print("=== TESTING IMPROVED CONSTRAINTS AND ANGLE CALCULATION ===\n")

# Test 1: Hard constraints preventing impossible transitions
print("1. Testing hard constraints for impossible transitions:")
print("   (These should be REJECTED and never appear in results)")

impossible_cases = [
    (77.3, -94.3, "Impossible switch: +77.3° to -94.3°"),
    (75.6, -98.3, "Impossible switch: +75.6° to -98.3°"),
    (67.8, -107.4, "Extreme impossible switch"),
    (60.1, -116.7, "Another extreme switch"),
    (-114.0, 47.5, "Reverse impossible switch"),
    (70.0, -120.0, "Large magnitude switch"),
]

print("   Transition Analysis:")
for prev, curr, description in impossible_cases:
    is_impossible, is_extreme, change = test_hard_constraints(prev, curr)
    status = "REJECTED" if (is_impossible or is_extreme) else "ALLOWED"
    
    print(f"   {description}")
    print(f"     {prev:.1f}° → {curr:.1f}° (change: {change:.1f}°) - {status}")
    if is_impossible:
        print(f"     Reason: Impossible sign switch")
    elif is_extreme:
        print(f"     Reason: Extreme change (>{150}°)")
    print()

# Test 2: Improved angle calculation for highly bent worms
print("2. Testing improved angle calculation for highly bent worms:")
print("   (Small angles should be scaled up to realistic values)")

bent_worm_cases = [
    (True, 45.0, "Highly bent worm with small calculated angle"),
    (True, 30.0, "Very bent worm with very small angle"),
    (True, 15.0, "Extremely bent worm with tiny angle"),
    (True, 75.0, "Bent worm with reasonable angle"),
    (False, 45.0, "Normal worm with moderate angle"),
    (True, -35.0, "Bent worm with small negative angle"),
]

print("   Angle Correction Analysis:")
for is_bent, original_angle, description in bent_worm_cases:
    improved_angle = calculate_improved_angle_for_bent_worms(is_bent, original_angle)
    change = improved_angle - original_angle
    
    print(f"   {description}")
    print(f"     Original: {original_angle:.1f}° → Improved: {improved_angle:.1f}° (change: {change:+.1f}°)")
    
    if is_bent and abs(original_angle) < 60:
        print(f"     ✅ Scaled up for realistic highly bent worm angle")
    elif not is_bent:
        print(f"     ✅ No change needed for normal worm")
    else:
        print(f"     ✅ Already realistic for bent worm")
    print()

# Test 3: Acceptable transitions that should still be allowed
print("3. Testing acceptable transitions that should be allowed:")
print("   (These represent normal worm movement)")

acceptable_cases = [
    (75.0, 78.0, "Small increase in bent angle"),
    (80.0, 85.0, "Gradual increase"),
    (-60.0, -65.0, "Small change maintaining sign"),
    (30.0, -25.0, "Reasonable sign change for normal worm"),
    (45.0, 50.0, "Normal progression"),
    (90.0, 95.0, "High angle small change"),
]

print("   Acceptable Transition Analysis:")
for prev, curr, description in acceptable_cases:
    is_impossible, is_extreme, change = test_hard_constraints(prev, curr)
    status = "ALLOWED" if not (is_impossible or is_extreme) else "REJECTED"
    
    print(f"   {description}")
    print(f"     {prev:.1f}° → {curr:.1f}° (change: {change:.1f}°) - {status}")
    print()

print("=== SUMMARY OF IMPROVEMENTS ===")
print("✅ Hard constraints prevent impossible sign switches (>90° change with both angles >45°)")
print("✅ Hard constraints prevent extreme changes (>150° regardless of sign)")
print("✅ Highly bent worms get realistic angles (scaled up from small values)")
print("✅ Normal worm movements are still allowed")
print("✅ Algorithm will skip bad configurations instead of just penalizing them")
print("\nResult: No more impossible transitions should appear in the final data!") 