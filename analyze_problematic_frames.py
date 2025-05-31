#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/lilly/phd')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== ANALYZING PROBLEMATIC FRAMES IN HIGHLY BENT SECTION ===\n")

# Load the results from the recent run
# We'll simulate loading the results_df that was just created
exec(open('riverchip/3_extract_head_angle.py').read())

print("1. Identifying frames in the highly bent section (around frames 400-450):")
highly_bent_section = results_df[(results_df['frame'] >= 400) & (results_df['frame'] <= 450)]

print(f"Found {len(highly_bent_section)} frames in the highly bent section")
print("\nFrame-by-frame analysis:")

problematic_frames = []
good_frames = []

for _, row in highly_bent_section.iterrows():
    frame = row['frame']
    angle = abs(row['angle_degrees'])
    
    # Based on visual evidence, angles in this section should be >90°
    if angle < 80:  # Suspiciously low for highly bent section
        problematic_frames.append((frame, angle))
        status = "PROBLEMATIC"
    elif angle > 90:
        good_frames.append((frame, angle))
        status = "GOOD"
    else:
        status = "BORDERLINE"
    
    print(f"Frame {frame}: {angle:.1f}° - {status}")

print(f"\n2. Summary of problematic frames:")
print(f"   - Problematic frames (angle < 80°): {len(problematic_frames)}")
print(f"   - Good frames (angle > 90°): {len(good_frames)}")

if problematic_frames:
    print(f"\n   Problematic frames details:")
    for frame, angle in problematic_frames:
        print(f"     Frame {frame}: {angle:.1f}°")

print(f"\n3. Testing if it's always the same frames:")
print("   Let's check if these problematic frames are consistent across different runs...")

# Check the angle distribution in the highly bent section
angles_in_section = highly_bent_section['angle_degrees'].abs()
print(f"\n4. Angle statistics in highly bent section:")
print(f"   Mean: {angles_in_section.mean():.1f}°")
print(f"   Median: {angles_in_section.median():.1f}°")
print(f"   Min: {angles_in_section.min():.1f}°")
print(f"   Max: {angles_in_section.max():.1f}°")
print(f"   Std: {angles_in_section.std():.1f}°")

# Check if there's a pattern in the angle calculation method
print(f"\n5. Investigating potential causes:")

# Look at the raw debug output to see what's happening
print("   Checking if the issue is:")
print("   a) Skeleton quality problems in specific frames")
print("   b) Angle calculation giving acute instead of obtuse angles")
print("   c) Head section selection issues")
print("   d) Scaling problems in the angle calculation")

# Let's examine a few specific problematic frames in detail
if problematic_frames:
    print(f"\n6. Detailed analysis of most problematic frame:")
    worst_frame, worst_angle = min(problematic_frames, key=lambda x: x[1])
    print(f"   Frame {worst_frame} with angle {worst_angle:.1f}°")
    
    # We would need to examine the skeleton and vectors for this frame
    print("   This frame should be examined for:")
    print("   - Skeleton quality and shape")
    print("   - Head and body vector directions")
    print("   - Whether the angle is acute when it should be obtuse")
    print("   - Head section used for calculation")

print(f"\n7. Hypothesis testing:")
print("   Based on your observation that similar head positions give different angles,")
print("   the issue might be:")
print("   - Angle calculation giving supplementary angle (180° - actual)")
print("   - Vector direction issues causing acute vs obtuse confusion")
print("   - Skeleton artifacts in specific frames")
print("   - Inconsistent head section selection despite improvements")

# Create a simple plot to visualize the problem
plt.figure(figsize=(12, 6))
frames = highly_bent_section['frame'].values
angles = highly_bent_section['angle_degrees'].abs().values

plt.plot(frames, angles, 'bo-', alpha=0.7, label='Calculated Angles')
plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90° Reference')
plt.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80° Threshold')

# Highlight problematic frames
if problematic_frames:
    prob_frames, prob_angles = zip(*problematic_frames)
    plt.scatter(prob_frames, prob_angles, color='red', s=100, alpha=0.8, 
                label='Problematic Frames', zorder=5)

plt.xlabel('Frame Number')
plt.ylabel('Absolute Angle (degrees)')
plt.title('Angles in Highly Bent Section (Frames 400-450)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('problematic_frames_analysis.png', dpi=150)
plt.close()

print(f"\n8. Saved analysis plot as 'problematic_frames_analysis.png'")
print("   This shows which specific frames are problematic in the highly bent section.") 