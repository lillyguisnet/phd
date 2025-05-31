#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/lilly/phd')

# Import necessary modules
import numpy as np
import pandas as pd

print("Testing highly bent worm detection fix...")

# Re-run the processing with the updated logic
exec(open('riverchip/3_extract_head_angle.py').read())

print("\nLooking for frames that were identified as highly bent:")
print("These should now have larger, more appropriate angles...")

# Check the problematic region around frame 425
print("\nChecking frames 420-430:")
test_frames = range(420, 431)
for frame in test_frames:
    frame_data = results_df[results_df['frame'] == frame]
    if not frame_data.empty:
        angle = frame_data.iloc[0]['angle_degrees']
        error = frame_data.iloc[0]['error']
        print(f"Frame {frame}: {angle:.1f}° - Error: {error}")

print("\nLooking for any remaining unrealistic small angles in highly bent regions...")
# Look for frames with suspiciously small angles that might indicate the worm is actually highly bent
small_angle_frames = []
for frame in sorted(results_df['frame'].unique()):
    frame_data = results_df[results_df['frame'] == frame]
    if not frame_data.empty:
        angle = abs(frame_data.iloc[0]['angle_degrees'])
        if 30 < angle < 80:  # Suspicious range - might be underestimated
            small_angle_frames.append((frame, angle))

if small_angle_frames:
    print(f"Found {len(small_angle_frames)} frames with potentially underestimated angles:")
    for frame, angle in small_angle_frames[:10]:  # Show first 10
        print(f"  Frame {frame}: {angle:.1f}°")
else:
    print("No suspicious small angles found - fix appears to be working!")

print(f"\nAngle distribution summary:")
angles = results_df['angle_degrees'].abs()
print(f"Mean angle: {angles.mean():.1f}°")
print(f"Median angle: {angles.median():.1f}°")
print(f"Angles > 100°: {(angles > 100).sum()} frames")
print(f"Angles 50-100°: {((angles >= 50) & (angles <= 100)).sum()} frames")
print(f"Angles < 50°: {(angles < 50).sum()} frames") 