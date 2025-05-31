#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/lilly/phd')

# Import necessary modules
import numpy as np
import pandas as pd
from pathlib import Path

# Load the existing data to debug the merge issue
print("Debugging merge issue...")

# Re-run just the processing part with the updated logic
exec(open('riverchip/3_extract_head_angle.py').read())

print("\nChecking frames 420-430 in results_df:")
test_frames = range(420, 431)
for frame in test_frames:
    frame_data = results_df[results_df['frame'] == frame]
    if not frame_data.empty:
        angle = frame_data.iloc[0]['angle_degrees']
        error = frame_data.iloc[0]['error']
        print(f"results_df Frame {frame}: {angle:.1f}° - Error: {error}")

print("\nChecking frames 420-430 in merged_df:")
for frame in test_frames:
    frame_data = merged_df[merged_df['frame'] == frame]
    if not frame_data.empty:
        angle = frame_data.iloc[0]['angle_degrees']
        corrected_angle = frame_data.iloc[0]['angle_degrees_corrected']
        side = frame_data.iloc[0]['side_position']
        print(f"merged_df Frame {frame}: {angle:.1f}° -> {corrected_angle:.1f}° (side: {side})")

print(f"\nFrame count comparison:")
print(f"results_df frames: {len(results_df['frame'].unique())}")
print(f"merged_df frames: {len(merged_df['frame'].unique())}")

print(f"\nFrame range comparison:")
print(f"results_df: {results_df['frame'].min()} to {results_df['frame'].max()}")
print(f"merged_df: {merged_df['frame'].min()} to {merged_df['frame'].max()}")

# Check if there are any NaN values in merged_df angle_degrees
nan_count = merged_df['angle_degrees'].isna().sum()
print(f"\nNaN values in merged_df angle_degrees: {nan_count}")

if nan_count > 0:
    print("Frames with NaN angles:")
    nan_frames = merged_df[merged_df['angle_degrees'].isna()]['frame'].tolist()
    print(f"NaN frames: {nan_frames[:10]}...")  # Show first 10 