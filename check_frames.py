#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/lilly/phd')

# Import necessary modules
import numpy as np
import pandas as pd

print("Checking specific problematic frames...")

# Load the data from the saved CSV if it exists
try:
    # Try to load the most recent results
    import pickle
    import h5py
    from pathlib import Path
    
    # Re-run the processing to get fresh results
    exec(open('riverchip/3_extract_head_angle.py').read())
    
    print("\nChecking frames 420-430 for fallback preservation:")
    test_frames = range(420, 431)
    for frame in test_frames:
        frame_data = results_df[results_df['frame'] == frame]
        if not frame_data.empty:
            angle = frame_data.iloc[0]['angle_degrees']
            error = frame_data.iloc[0]['error']
            print(f"Frame {frame}: {angle:.1f}° - Error: {error}")
    
    print("\nLooking for large frame-to-frame changes...")
    # Check for large frame-to-frame changes
    prev_angle = None
    large_changes = []
    for frame in sorted(results_df['frame'].unique()):
        frame_data = results_df[results_df['frame'] == frame]
        if not frame_data.empty:
            angle = frame_data.iloc[0]['angle_degrees']
            if prev_angle is not None:
                change = abs(angle - prev_angle)
                if change > 30:  # Large change
                    large_changes.append((frame, prev_angle, angle, change))
            prev_angle = angle
    
    if large_changes:
        print("Found large changes:")
        for frame, prev_ang, curr_ang, change in large_changes:
            print(f"Frame {frame}: {prev_ang:.1f}° -> {curr_ang:.1f}° (change: {change:.1f}°)")
    else:
        print("No large changes found! Fix appears to be working.")
        
except Exception as e:
    print(f"Error: {e}")
    print("Could not load or process data.") 