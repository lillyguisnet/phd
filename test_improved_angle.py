#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/lilly/phd')

# Import necessary modules
import numpy as np
import pandas as pd

print("Testing improved angle calculation for highly bent worms...")

# Re-run the processing with the updated logic
exec(open('riverchip/3_extract_head_angle.py').read())

print("\nLooking for frames with alternative angle calculations:")
print("These should show 'Highly bent: Original angle X°, Deviation angle Y°, Using Z°'")

# Check the problematic region around frame 425
print("\nChecking frames 420-430:")
test_frames = range(420, 431)
for frame in test_frames:
    frame_data = results_df[results_df['frame'] == frame]
    if not frame_data.empty:
        angle = frame_data.iloc[0]['angle_degrees']
        error = frame_data.iloc[0]['error']
        print(f"Frame {frame}: {angle:.1f}° - Error: {error}")

print("\nAngle distribution comparison:")
angles = results_df['angle_degrees'].abs()
print(f"Angles > 120°: {(angles > 120).sum()} frames")
print(f"Angles 100-120°: {((angles >= 100) & (angles <= 120)).sum()} frames")
print(f"Angles 80-100°: {((angles >= 80) & (angles < 100)).sum()} frames")
print(f"Angles 60-80°: {((angles >= 60) & (angles < 80)).sum()} frames")
print(f"Angles < 60°: {(angles < 60).sum()} frames")

print(f"\nMean angle: {angles.mean():.1f}°")
print(f"Max angle: {angles.max():.1f}°")

# Look for the specific problematic frames from the video
print("\nLooking for frames that should have been ~120-140° but were calculated as ~60-70°:")
suspicious_frames = []
for frame in sorted(results_df['frame'].unique()):
    frame_data = results_df[results_df['frame'] == frame]
    if not frame_data.empty:
        angle = abs(frame_data.iloc[0]['angle_degrees'])
        # Look for frames that might have been underestimated
        if 50 < angle < 90:  # Suspicious range
            suspicious_frames.append((frame, angle))

if suspicious_frames:
    print(f"Found {len(suspicious_frames)} potentially underestimated frames:")
    for frame, angle in suspicious_frames[:10]:
        print(f"  Frame {frame}: {angle:.1f}°")
else:
    print("No suspicious underestimated angles found!") 