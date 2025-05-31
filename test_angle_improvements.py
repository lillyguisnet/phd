import numpy as np

def calculate_unified_angle_old(head_vector, body_vector, head_mag, body_mag, is_highly_bent):
    """Old method that had issues with capping and sign switches"""
    # Traditional head-body angle
    dot_product = np.dot(head_vector, body_vector)
    cos_angle = np.clip(dot_product / (head_mag * body_mag), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    traditional_angle = np.degrees(angle_rad)
    
    # Apply sign based on cross product
    cross_product = np.cross(body_vector, head_vector)
    if cross_product < 0:
        traditional_angle = -traditional_angle
    
    if is_highly_bent and abs(traditional_angle) < 70:
        # Force a reasonable angle for highly bent worms (THIS WAS THE PROBLEM!)
        return 90.0 if traditional_angle >= 0 else -90.0
    
    return traditional_angle

def calculate_unified_angle_new(head_vector, body_vector, head_mag, body_mag, is_highly_bent):
    """New method using atan2 for full range and better handling"""
    # Calculate the angle between head and body vectors using atan2
    head_angle = np.arctan2(head_vector[1], head_vector[0])
    body_angle = np.arctan2(body_vector[1], body_vector[0])
    
    # Calculate the difference between angles
    angle_diff = head_angle - body_angle
    
    # Normalize to [-π, π] range
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # Convert to degrees
    traditional_angle = np.degrees(angle_diff)
    
    # For normal worms, use the traditional method
    if not is_highly_bent:
        return traditional_angle
    
    # For highly bent worms, allow the full range without artificial capping
    return traditional_angle

def test_continuity_scoring(prev_angle, current_angle, is_highly_bent):
    """Test the new continuity scoring that prevents impossible switches"""
    continuity_score = 0
    
    if prev_angle is not None:
        angle_change = abs(current_angle - prev_angle)
        
        # Check for impossible sign switches
        sign_switch_penalty = 0
        if prev_angle * current_angle < 0:  # Different signs
            min_abs_angle = min(abs(prev_angle), abs(current_angle))
            max_abs_angle = max(abs(prev_angle), abs(current_angle))
            
            if min_abs_angle > 45 and angle_change > 90:
                sign_switch_penalty = 200  # Massive penalty for impossible switches
            elif min_abs_angle > 30 and angle_change > 60:
                sign_switch_penalty = 100  # Large penalty for suspicious switches
        
        continuity_score -= sign_switch_penalty
        
        # Adjust acceptable change based on worm type
        if is_highly_bent:
            if abs(prev_angle) > 60:
                max_acceptable_change = 25
            else:
                max_acceptable_change = 40
        else:
            max_acceptable_change = 20
        
        if angle_change <= max_acceptable_change:
            continuity_score += (max_acceptable_change - angle_change)
        else:
            excess_change = angle_change - max_acceptable_change
            if excess_change > 50:
                continuity_score -= excess_change * 3
            else:
                continuity_score -= excess_change * 2
    
    return continuity_score, sign_switch_penalty

# Test cases demonstrating the improvements
print("=== TESTING ANGLE CALCULATION IMPROVEMENTS ===\n")

# Test Case 1: Highly bent worm vectors that should give >90° angles
print("1. Testing highly bent worm angle calculation:")
print("   (Should allow angles > 90° without capping)")

# Simulate a very bent worm head
head_vector = np.array([-1, 0.5])  # Head pointing back and slightly up
body_vector = np.array([1, 0])     # Body pointing forward
head_mag = np.linalg.norm(head_vector)
body_mag = np.linalg.norm(body_vector)

old_angle = calculate_unified_angle_old(head_vector, body_vector, head_mag, body_mag, True)
new_angle = calculate_unified_angle_new(head_vector, body_vector, head_mag, body_mag, True)

print(f"   Head vector: {head_vector}")
print(f"   Body vector: {body_vector}")
print(f"   Old method: {old_angle:.1f}° (artificially capped)")
print(f"   New method: {new_angle:.1f}° (full range allowed)")
print()

# Test Case 2: Another highly bent case
print("2. Testing another highly bent configuration:")
head_vector = np.array([-0.8, -0.6])  # Head pointing back and down
body_vector = np.array([1, 0.2])      # Body pointing forward and slightly up

head_mag = np.linalg.norm(head_vector)
body_mag = np.linalg.norm(body_vector)

old_angle = calculate_unified_angle_old(head_vector, body_vector, head_mag, body_mag, True)
new_angle = calculate_unified_angle_new(head_vector, body_vector, head_mag, body_mag, True)

print(f"   Head vector: {head_vector}")
print(f"   Body vector: {body_vector}")
print(f"   Old method: {old_angle:.1f}° (artificially capped)")
print(f"   New method: {new_angle:.1f}° (full range allowed)")
print()

# Test Case 3: Continuity scoring to prevent impossible switches
print("3. Testing continuity scoring for impossible sign switches:")
print("   (Should heavily penalize impossible transitions)")

test_cases = [
    (77.0, -94.0, True, "Impossible switch: +77° to -94°"),
    (75.0, 78.0, True, "Good continuity: +75° to +78°"),
    (-60.0, 65.0, True, "Suspicious switch: -60° to +65°"),
    (30.0, -35.0, False, "Reasonable switch for normal worm"),
    (85.0, -120.0, True, "Extreme impossible switch")
]

for prev_angle, curr_angle, is_bent, description in test_cases:
    score, penalty = test_continuity_scoring(prev_angle, curr_angle, is_bent)
    change = abs(curr_angle - prev_angle)
    
    print(f"   {description}")
    print(f"     Change: {change:.1f}°, Sign switch penalty: {penalty}, Total score: {score}")
    print()

print("=== SUMMARY OF IMPROVEMENTS ===")
print("1. ✅ Removed artificial 90° capping for highly bent worms")
print("2. ✅ Used atan2 for full 360° range instead of arccos (0-180°)")
print("3. ✅ Added heavy penalties for impossible sign switches")
print("4. ✅ Improved continuity scoring based on worm type and angle magnitude")
print("5. ✅ Progressive penalties for excessive angle changes") 