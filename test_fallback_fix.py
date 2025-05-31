#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/lilly/phd')

# Import necessary modules
import numpy as np
import pandas as pd
from pathlib import Path

# Load the existing data to test the fix
print("Testing fallback preservation fix...")

# Re-run just the processing part with the updated logic
exec(open('riverchip/3_extract_head_angle.py').read())

print("\nChecking frames 420-430 for fallback preservation:")
test_frames = range(420, 431)
for frame in test_frames:
    frame_data = results_df[results_df['frame'] == frame]
    if not frame_data.empty:
        angle = frame_data.iloc[0]['angle_degrees']
        error = frame_data.iloc[0]['error']
        print(f"Frame {frame}: {angle:.1f}° - Error: {error}")

print("\nLooking for unrealistic drops...")
# Check for large frame-to-frame changes
prev_angle = None
for frame in sorted(results_df['frame'].unique()):
    frame_data = results_df[results_df['frame'] == frame]
    if not frame_data.empty:
        angle = frame_data.iloc[0]['angle_degrees']
        if prev_angle is not None:
            change = abs(angle - prev_angle)
            if change > 30:  # Large change
                print(f"Frame {frame}: {angle:.1f}° (change: {change:.1f}° from {prev_angle:.1f}°)")
        prev_angle = angle 