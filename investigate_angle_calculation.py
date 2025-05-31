#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/lilly/phd')

import numpy as np
import matplotlib.pyplot as plt

print("=== INVESTIGATING ANGLE CALCULATION ISSUES ===\n")

def analyze_angle_calculation(head_vector, body_vector, frame_info=""):
    """
    Analyze different methods of calculating angles to understand the issue
    """
    print(f"Analyzing angle calculation {frame_info}:")
    print(f"  Head vector: [{head_vector[0]:.2f}, {head_vector[1]:.2f}]")
    print(f"  Body vector: [{body_vector[0]:.2f}, {body_vector[1]:.2f}]")
    
    # Method 1: Traditional dot product (gives acute angle 0-180°)
    head_mag = np.linalg.norm(head_vector)
    body_mag = np.linalg.norm(body_vector)
    
    if head_mag > 0 and body_mag > 0:
        dot_product = np.dot(head_vector, body_vector)
        cos_angle = np.clip(dot_product / (head_mag * body_mag), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        dot_angle = np.degrees(angle_rad)
        
        # Apply sign based on cross product
        cross_product = np.cross(body_vector, head_vector)
        if cross_product < 0:
            dot_angle = -dot_angle
        
        print(f"  Method 1 (dot product): {dot_angle:.1f}°")
    else:
        dot_angle = 0
        print(f"  Method 1 (dot product): Invalid (zero vector)")
    
    # Method 2: atan2 method (current implementation)
    head_angle = np.arctan2(head_vector[1], head_vector[0])
    body_angle = np.arctan2(body_vector[1], body_vector[0])
    
    angle_diff = head_angle - body_angle
    
    # Normalize to [-π, π] range
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    atan2_angle = np.degrees(angle_diff)
    print(f"  Method 2 (atan2): {atan2_angle:.1f}°")
    
    # Method 3: Alternative atan2 (checking if we should use supplementary)
    alt_angle_diff = body_angle - head_angle  # Reversed order
    while alt_angle_diff > np.pi:
        alt_angle_diff -= 2 * np.pi
    while alt_angle_diff < -np.pi:
        alt_angle_diff += 2 * np.pi
    
    alt_atan2_angle = np.degrees(alt_angle_diff)
    print(f"  Method 3 (atan2 reversed): {alt_atan2_angle:.1f}°")
    
    # Method 4: Check supplementary angles
    supplementary_dot = 180 - abs(dot_angle) if dot_angle != 0 else 0
    if dot_angle < 0:
        supplementary_dot = -supplementary_dot
    print(f"  Method 4 (supplementary): {supplementary_dot:.1f}°")
    
    # Analysis
    print(f"  Analysis:")
    print(f"    - Dot product gives: {abs(dot_angle):.1f}° (always acute/obtuse)")
    print(f"    - atan2 gives: {abs(atan2_angle):.1f}°")
    print(f"    - Difference: {abs(abs(atan2_angle) - abs(dot_angle)):.1f}°")
    
    if abs(abs(atan2_angle) - abs(dot_angle)) > 10:
        print(f"    ⚠️  SIGNIFICANT DIFFERENCE - possible acute/obtuse confusion!")
    
    # Check if we might need the supplementary angle
    if abs(dot_angle) < 90 and abs(supplementary_dot) > 90:
        print(f"    💡 For highly bent worm, supplementary angle {abs(supplementary_dot):.1f}° might be more appropriate")
    
    print()
    
    return {
        'dot_angle': dot_angle,
        'atan2_angle': atan2_angle,
        'supplementary': supplementary_dot,
        'head_angle_rad': head_angle,
        'body_angle_rad': body_angle
    }

# Test with some example vectors that might represent highly bent worms
print("1. Testing with example vectors for highly bent worms:\n")

test_cases = [
    {
        'name': 'Case 1: Head bent strongly left',
        'head_vector': np.array([-1.0, 0.5]),  # Head pointing left and slightly up
        'body_vector': np.array([1.0, 0.0])    # Body pointing right
    },
    {
        'name': 'Case 2: Head bent strongly down',
        'head_vector': np.array([0.2, -1.0]),  # Head pointing down
        'body_vector': np.array([0.0, 1.0])    # Body pointing up
    },
    {
        'name': 'Case 3: Head bent back (U-shape)',
        'head_vector': np.array([-0.8, -0.6]), # Head pointing back and down
        'body_vector': np.array([1.0, 0.2])    # Body pointing forward and slightly up
    },
    {
        'name': 'Case 4: Extreme bend (almost 180°)',
        'head_vector': np.array([-0.9, 0.1]),  # Head pointing almost opposite
        'body_vector': np.array([1.0, 0.0])    # Body pointing forward
    }
]

results = []
for case in test_cases:
    result = analyze_angle_calculation(
        case['head_vector'], 
        case['body_vector'], 
        f"({case['name']})"
    )
    results.append({**case, **result})

print("2. Summary of findings:\n")

for i, result in enumerate(results):
    print(f"   {result['name']}:")
    dot_abs = abs(result['dot_angle'])
    atan2_abs = abs(result['atan2_angle'])
    supp_abs = abs(result['supplementary'])
    
    print(f"     Dot product: {dot_abs:.1f}°")
    print(f"     atan2: {atan2_abs:.1f}°")
    print(f"     Supplementary: {supp_abs:.1f}°")
    
    # Determine which is most appropriate for a highly bent worm
    if supp_abs > 90 and dot_abs < 90:
        print(f"     🎯 For highly bent worm: supplementary ({supp_abs:.1f}°) is more realistic")
    elif atan2_abs > 90:
        print(f"     ✅ atan2 method gives realistic angle for bent worm")
    elif dot_abs > 90:
        print(f"     ✅ Dot product gives realistic angle for bent worm")
    else:
        print(f"     ❌ All methods give unrealistically small angles")
    print()

print("3. Recommendations:\n")
print("   Based on this analysis, the issue might be:")
print("   a) We're consistently choosing acute angles when obtuse would be more appropriate")
print("   b) The vector directions might be inconsistent")
print("   c) We need to check if the angle should be supplementary for highly bent worms")
print()
print("   Potential fixes:")
print("   1. For highly bent worms, if calculated angle < 90°, consider using 180° - angle")
print("   2. Add logic to detect when vectors suggest extreme bending")
print("   3. Use curvature information to validate angle magnitude")
print("   4. Check vector orientations more carefully")

# Create a visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, result in enumerate(results):
    ax = axes[i]
    
    # Plot vectors
    head_vec = result['head_vector']
    body_vec = result['body_vector']
    
    # Draw vectors from origin
    ax.arrow(0, 0, body_vec[0], body_vec[1], head_width=0.05, head_length=0.1, 
             fc='blue', ec='blue', label='Body Vector')
    ax.arrow(0, 0, head_vec[0], head_vec[1], head_width=0.05, head_length=0.1, 
             fc='red', ec='red', label='Head Vector')
    
    # Draw angle arc
    angles = np.linspace(result['body_angle_rad'], result['head_angle_rad'], 50)
    if result['head_angle_rad'] < result['body_angle_rad']:
        angles = np.linspace(result['head_angle_rad'], result['body_angle_rad'], 50)
    
    arc_radius = 0.3
    arc_x = arc_radius * np.cos(angles)
    arc_y = arc_radius * np.sin(angles)
    ax.plot(arc_x, arc_y, 'green', linewidth=2)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{result['name']}\natan2: {abs(result['atan2_angle']):.1f}°, "
                f"dot: {abs(result['dot_angle']):.1f}°")
    
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('angle_calculation_analysis.png', dpi=150)
plt.close()

print(f"\n4. Saved vector analysis plot as 'angle_calculation_analysis.png'")
print("   This shows the vector orientations and calculated angles for different cases.") 