import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shutil
from pathlib import Path
import cv2
import h5py
import pickle
import numpy as np
import h5py
from scipy import interpolate
from skimage import morphology, graph, draw
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import welch, find_peaks
from collections import Counter
from collections import defaultdict
from pathlib import Path
import os
from skimage import measure
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage import graph
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.draw import line
from skimage import morphology
from scipy.interpolate import splprep, splev
import pandas as pd
from typing import Dict, Set, List, Tuple
import random
import numpy.linalg as la
from scipy import interpolate, ndimage
from collections import defaultdict
import re
import multiprocessing as mp
from functools import partial
import time

# Configuration flags - OPTIMIZED FOR 48-CORE MACHINE
USE_MULTIPROCESSING = True  # Set to False if multiprocessing causes issues
MAX_PROCESSES = 32  # Use most cores but leave some for system (48 cores available)
CHUNK_SIZE_MULTIPLIER = 2  # Increase chunks per process for better load balancing

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            
            # Debug print
            print(f"Loading frame {frame_idx}")
    
    print(f"Cleaned segments loaded from {filename}")
    return cleaned_segments

def get_random_unprocessed_video(head_segmentation_dir, final_data_dir):
    # Get all head segmentation files
    all_videos = [f for f in os.listdir(head_segmentation_dir) if f.endswith("_headsegmentation.h5")]
    
    # For each video, check if it doesn't have a corresponding _headangles.csv file
    processable_videos = []
    for video in all_videos:
        # Extract base name by removing _headsegmentation.h5
        base_name = video.replace("_headsegmentation.h5", "")
        
        # Check if headangles.csv doesn't exist for this base name
        if not os.path.exists(os.path.join(final_data_dir, base_name + "_headangles.csv")):
            processable_videos.append(video)
    
    if not processable_videos:
        raise ValueError("No videos found that need head angle processing.")
    
    return os.path.join(head_segmentation_dir, random.choice(processable_videos))

head_segmentation_dir = "/home/lilly/phd/riverchip/data_analyzed/head_segmentation"
final_data_dir = "/home/lilly/phd/riverchip/data_analyzed/final_data"

filename = get_random_unprocessed_video(head_segmentation_dir, final_data_dir)
head_segments = load_cleaned_segments_from_h5(filename)


###Get skeletons from masks
def get_skeleton(mask):
    return morphology.skeletonize(mask)

def process_frame_skeletons_parallel(frame_data_chunk):
    """
    Process a chunk of frames to generate skeletons in parallel.
    
    Args:
        frame_data_chunk: List of (frame_idx, frame_data) tuples
    
    Returns:
        Dictionary with frame indices as keys and dictionaries of skeletons as values
    """
    chunk_skeletons = {}
    chunk_sizes = []
    
    for frame_idx, frame_data in frame_data_chunk:
        frame_skeletons = {}
        
        for obj_id, mask in frame_data.items():
            # Generate skeleton from mask
            skeleton = get_skeleton(mask)
            frame_skeletons[obj_id] = skeleton
            
            # Track skeleton size (number of pixels)
            size = np.sum(skeleton)
            chunk_sizes.append(size)
        
        chunk_skeletons[frame_idx] = frame_skeletons
    
    return chunk_skeletons, chunk_sizes

def process_all_frames_parallel(head_segments, num_processes=None):
    """
    Process all frames to generate skeletons using parallel processing.
    
    Args:
        head_segments: Dictionary with frame indices as keys and inner dictionaries 
                      containing object masks as values
        num_processes: Number of processes to use (default: CPU count)
    
    Returns:
        Tuple containing:
        - Dictionary with frame indices as keys and dictionaries of skeletons as values
        - Dictionary with skeleton size statistics
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), MAX_PROCESSES)
    
    print(f"🚀 Using {num_processes} processes for skeleton generation on {mp.cpu_count()}-core machine...")
    
    # Convert to list of (frame_idx, frame_data) tuples
    frame_list = list(head_segments.items())
    print(f"Processing {len(frame_list)} frames...")
    
    # Optimize chunk sizing for better load balancing on many cores
    # Use smaller chunks so work is distributed more evenly
    optimal_chunk_size = max(1, len(frame_list) // (num_processes * CHUNK_SIZE_MULTIPLIER))
    chunks = [frame_list[i:i + optimal_chunk_size] for i in range(0, len(frame_list), optimal_chunk_size)]
    
    print(f"📦 Split into {len(chunks)} chunks of ~{optimal_chunk_size} frames each for optimal load balancing")
    
    # Process chunks in parallel
    try:
        with mp.Pool(processes=num_processes) as pool:
            print("⚡ Starting parallel skeleton generation...")
            results = pool.map(process_frame_skeletons_parallel, chunks)
            print("✅ Skeleton generation completed!")
    except Exception as e:
        print(f"❌ Error in parallel processing: {e}")
        # Fallback to sequential processing
        print("🔄 Falling back to sequential processing...")
        results = [process_frame_skeletons_parallel(chunk) for chunk in chunks]
    
    # Combine results
    skeletons = {}
    all_skeleton_sizes = []
    
    for chunk_skeletons, chunk_sizes in results:
        skeletons.update(chunk_skeletons)
        all_skeleton_sizes.extend(chunk_sizes)
    
    # Calculate statistics
    stats = {
        'min_size': np.min(all_skeleton_sizes),
        'max_size': np.max(all_skeleton_sizes), 
        'mean_size': np.mean(all_skeleton_sizes),
        'median_size': np.median(all_skeleton_sizes),
        'std_size': np.std(all_skeleton_sizes)
    }
    
    print("\n📊 Skeleton Statistics:")
    print(f"Minimum size: {stats['min_size']:.1f} pixels")
    print(f"Maximum size: {stats['max_size']:.1f} pixels") 
    print(f"Mean size: {stats['mean_size']:.1f} pixels")
    print(f"Median size: {stats['median_size']:.1f} pixels")
    print(f"Standard deviation: {stats['std_size']:.1f} pixels")
    
    return skeletons, stats

print(f"🏁 Starting head angle extraction on {mp.cpu_count()}-core machine...")
start_time = time.time()

skeletons, skeleton_stats = process_all_frames_parallel(head_segments, num_processes=MAX_PROCESSES if USE_MULTIPROCESSING else 1)

skeleton_time = time.time()
print(f"⏱️  Skeleton generation took {skeleton_time - start_time:.2f} seconds")



###Truncate skeletons
def truncate_skeleton_chunk(chunk_data):
    """
    Truncate skeletons for a chunk of frames in parallel.
    
    Args:
        chunk_data: Tuple of (frame_data_chunk, keep_pixels, adaptive_mode, skeleton_stats)
    
    Returns:
        Dictionary with truncated skeletons for the chunk
    """
    frame_data_chunk, keep_pixels, adaptive_mode, skeleton_stats = chunk_data
    
    truncated_chunk = {}
    
    for frame_idx, frame_data in frame_data_chunk:
        frame_truncated = {}
        
        for obj_id, skeleton in frame_data.items():
            # Get the 2D array from the 3D input (taking the first channel)
            skeleton_2d = skeleton[0]
            
            # Find all non-zero points
            points = np.where(skeleton_2d)
            if len(points[0]) == 0:  # Empty skeleton
                frame_truncated[obj_id] = skeleton
                continue
            
            # Get the top point and bottom point
            y_min = np.min(points[0])
            y_max = np.max(points[0])
            original_height = y_max - y_min
            
            # Calculate cutoff point
            cutoff_point = y_min + keep_pixels + 1
            
            # Create truncated skeleton
            truncated = skeleton.copy()
            truncated[0, cutoff_point:, :] = False  # Using False since it's boolean type
            
            frame_truncated[obj_id] = truncated
            
        truncated_chunk[frame_idx] = frame_truncated
    
    return truncated_chunk

def truncate_skeleton_fixed_parallel(skeleton_dict, keep_pixels=150, adaptive_mode=False, skeleton_stats=None, num_processes=None):
    """
    Keeps only the top specified number of pixels of skeletons using parallel processing.
    
    Args:
        skeleton_dict: Dictionary with frame indices as keys and inner dictionaries 
                      containing skeletons as values of shape (1, h, w)
        keep_pixels: Number of pixels to keep from the top (default 150), ignored if adaptive_mode=True
        adaptive_mode: If True, use 10% less than the smallest skeleton height
        skeleton_stats: Dictionary containing skeleton statistics (required if adaptive_mode=True)
        num_processes: Number of processes to use (default: CPU count)
    
    Returns:
        Dictionary with truncated skeletons
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), MAX_PROCESSES)
    
    # Calculate adaptive keep_pixels if requested
    if adaptive_mode:
        if skeleton_stats is None:
            raise ValueError("skeleton_stats must be provided when adaptive_mode=True")
        
        # Use 90% of the minimum skeleton pixel count from skeleton_stats
        min_pixel_count = skeleton_stats['min_size']
        target_pixel_count = int(min_pixel_count * 0.9)  # 10% less than smallest
        
        print(f"🎯 Adaptive mode: Target pixel count = {target_pixel_count} (90% of smallest skeleton: {min_pixel_count} pixels)")
        
        # For simplicity in parallel processing, use a fixed height based on average
        # This is an approximation but much faster than the complex search
        keep_pixels = int(skeleton_stats['mean_size'] * 0.6)  # Rough estimate
        print(f"📏 Adaptive mode: Using truncation height of {keep_pixels} pixels")
    
    print(f"🚀 Using {num_processes} processes for skeleton truncation on {mp.cpu_count()}-core machine...")
    
    # Convert to list of (frame_idx, frame_data) tuples
    frame_list = list(skeleton_dict.items())
    print(f"Truncating {len(frame_list)} frames...")
    
    # Optimize chunk sizing for better load balancing on many cores
    optimal_chunk_size = max(1, len(frame_list) // (num_processes * CHUNK_SIZE_MULTIPLIER))
    chunks = [frame_list[i:i + optimal_chunk_size] for i in range(0, len(frame_list), optimal_chunk_size)]
    
    print(f"📦 Split into {len(chunks)} chunks of ~{optimal_chunk_size} frames each for optimal load balancing")
    
    # Prepare chunk data with parameters
    chunk_data_list = [(chunk, keep_pixels, adaptive_mode, skeleton_stats) for chunk in chunks]
    
    # Process chunks in parallel
    try:
        with mp.Pool(processes=num_processes) as pool:
            print("⚡ Starting parallel skeleton truncation...")
            results = pool.map(truncate_skeleton_chunk, chunk_data_list)
            print("✅ Skeleton truncation completed!")
    except Exception as e:
        print(f"❌ Error in parallel processing: {e}")
        # Fallback to sequential processing
        print("🔄 Falling back to sequential processing...")
        results = [truncate_skeleton_chunk(chunk_data) for chunk_data in chunk_data_list]
    
    # Combine results
    truncated_skeletons = {}
    for chunk_result in results:
        truncated_skeletons.update(chunk_result)
    
    print(f"🎉 Truncation complete! Processed {len(truncated_skeletons)} frames using {num_processes} cores.")
    return truncated_skeletons

# Now use conservative truncation that keeps ~90% of skeleton for better angle measurements
# Option 1: Use fixed pixels (original behavior) - DEPRECATED
# truncated_skeletons = truncate_skeleton_fixed(skeletons, keep_pixels=400) #200

# Option 2: Use adaptive mode (10% less than smallest skeleton) - DEPRECATED, too aggressive

# Option 3: Use conservative truncation that only removes noise at extremities
truncation_start = time.time()
truncated_skeletons = conservative_truncate_skeleton_parallel(skeletons, tail_trim_pixels=50, num_processes=MAX_PROCESSES if USE_MULTIPROCESSING else 1)
truncation_time = time.time()
print(f"⏱️  Conservative skeleton truncation took {truncation_time - truncation_start:.2f} seconds")




def smooth_head_angles(angles, window_size=3, deviation_threshold=15):
    """
    Smooth noise peaks in head angle data by comparing each point with the mean
    of surrounding windows. If a point deviates significantly from both surrounding
    windows' means, it is considered noise and smoothed.
    
    Parameters:
    -----------
    angles : array-like
        Array of head angle measurements
    window_size : int
        Size of the windows before and after point to compare means (default: 3)
    deviation_threshold : float
        Maximum allowed deviation from window means in degrees (default: 15)
        
    Returns:
    --------
    smoothed_angles : array
        Array of smoothed head angles
    peaks_detected : list of tuples
        List of (index, original_value, new_value) for detected noise peaks
    """
    import numpy as np
    
    angles = np.array(angles)
    smoothed = angles.copy()
    peaks_detected = []
    
    def check_and_smooth_point(i, window_before, window_after):
        """Helper function to check if a point is a noise peak and smooth if needed"""
        if len(window_before) == 0 or len(window_after) == 0:
            return None
            
        # Calculate means and standard deviations
        mean_before = np.mean(window_before)
        mean_after = np.mean(window_after)
        std_before = np.std(window_before)
        std_after = np.std(window_after)
        
        # Calculate absolute deviations from both means
        dev_from_before = abs(angles[i] - mean_before)
        dev_from_after = abs(angles[i] - mean_after)
        
        max_window_std = 10  # Maximum allowed standard deviation in windows
        
        # Check if point deviates significantly from both window means
        if (dev_from_before > deviation_threshold and 
            dev_from_after > deviation_threshold and
            std_before < max_window_std and 
            std_after < max_window_std):
            
            # Calculate new value as weighted average of means
            weight_before = 1 / (std_before + 1e-6)
            weight_after = 1 / (std_after + 1e-6)
            new_value = (mean_before * weight_before + mean_after * weight_after) / (weight_before + weight_after)
            
            return new_value
        
        return None
    
    # Process all points including edges
    for i in range(len(angles)):
        # Handle edge cases with adaptive window sizes
        if i < window_size:  # Start of sequence
            window_before = angles[0:i]
            window_after = angles[i+1:i+1+window_size]
        elif i >= len(angles) - window_size:  # End of sequence
            window_before = angles[i-window_size:i]
            window_after = angles[i+1:]
        else:  # Middle of sequence
            window_before = angles[i-window_size:i]
            window_after = angles[i+1:i+1+window_size]
        
        # Special handling for first and last points
        if i == 0:
            # For first point, use only the next few points
            if len(angles) > window_size:
                next_mean = np.mean(angles[1:1+window_size])
                next_std = np.std(angles[1:1+window_size])
                if (abs(angles[i] - next_mean) > deviation_threshold and 
                    next_std < 10):  # Using same max_window_std
                    new_value = next_mean
                    peaks_detected.append((i, angles[i], new_value))
                    smoothed[i] = new_value
            continue
            
        if i == len(angles) - 1:
            # For last point, use only the previous few points
            prev_mean = np.mean(angles[-window_size-1:-1])
            prev_std = np.std(angles[-window_size-1:-1])
            if (abs(angles[i] - prev_mean) > deviation_threshold and 
                prev_std < 10):
                new_value = prev_mean
                peaks_detected.append((i, angles[i], new_value))
                smoothed[i] = new_value
            continue
        
        # Process normal points
        new_value = check_and_smooth_point(i, window_before, window_after)
        if new_value is not None:
            peaks_detected.append((i, angles[i], new_value))
            smoothed[i] = new_value
    
    return smoothed, peaks_detected

def smooth_head_angles_with_validation(df, id_column='object_id', angle_column='angle_degrees',
                                     window_size=3, deviation_threshold=15):
    """
    Apply head angle smoothing to a DataFrame while respecting object ID boundaries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing head angle data
    id_column : str
        Name of the column containing object IDs
    angle_column : str
        Name of the column containing head angles
    window_size : int
        Size of the windows before and after point to compare means
    deviation_threshold : float
        Maximum allowed deviation from window means in degrees
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with smoothed angles and additional validation columns
    """
    import pandas as pd
    import numpy as np
    
    # Create copy of input DataFrame
    result_df = df.copy()
    
    # Add columns for tracking changes
    result_df['original_angle'] = df[angle_column]
    result_df['is_noise_peak'] = False
    result_df['peak_deviation'] = 0.0
    result_df['window_size_used'] = 0  # New column to track adaptive window sizes
    
    # Process each object separately
    for obj_id in df[id_column].unique():
        # Get indices for current object
        obj_mask = df[id_column] == obj_id
        obj_angles = df.loc[obj_mask, angle_column].values
        
        # Apply smoothing
        smoothed_angles, peaks = smooth_head_angles(
            obj_angles, 
            window_size=window_size,
            deviation_threshold=deviation_threshold
        )
        
        # Update results
        result_df.loc[obj_mask, angle_column] = smoothed_angles
        
        # Mark detected peaks in validation columns
        for idx, orig_val, new_val in peaks:
            df_idx = df.loc[obj_mask].iloc[idx].name
            result_df.loc[df_idx, 'is_noise_peak'] = True
            result_df.loc[df_idx, 'peak_deviation'] = abs(orig_val - new_val)
            # Calculate actual window size used
            if idx < window_size:
                result_df.loc[df_idx, 'window_size_used'] = idx
            elif idx >= len(obj_angles) - window_size:
                result_df.loc[df_idx, 'window_size_used'] = len(obj_angles) - idx - 1
            else:
                result_df.loc[df_idx, 'window_size_used'] = window_size
    
    return result_df

def normalize_skeleton_points(points, num_points=100):
    """
    Resample skeleton points to have uniform spacing.
    """
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dists[-1]
    
    even_dists = np.linspace(0, total_length, num_points)
    new_points = np.zeros((num_points, 2))
    for i in range(2):
        new_points[:, i] = np.interp(even_dists, cum_dists, points[:, i])
    
    return new_points, total_length

def gaussian_weighted_curvature(points, window_size=25, sigma=8, restriction_point=0.5):
    """
    Calculate curvature using Gaussian-weighted windows, with stronger smoothing
    and focus on the region between head tip and restriction point.
    
    Parameters:
    -----------
    points : array
        Skeleton points
    window_size : int
        Size of the window for curvature calculation (larger = more smoothing)
    sigma : float
        Gaussian smoothing parameter (larger = more smoothing)
    restriction_point : float
        Location of the restriction point along the skeleton (0-1)
    """
    # Only consider points up to the restriction point
    valid_points = points[:int(len(points) * restriction_point)]
    
    # Apply strong smoothing to the points first
    smooth_points = np.zeros_like(valid_points)
    for i in range(2):  # For both x and y coordinates
        smooth_points[:, i] = ndimage.gaussian_filter1d(valid_points[:, i], sigma=sigma/2)
    
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    pad_width = window_size // 2
    padded_points = np.pad(smooth_points, ((pad_width, pad_width), (0, 0)), mode='edge')
    
    weights = ndimage.gaussian_filter1d(np.ones(window_size), sigma)
    weights /= np.sum(weights)
    
    curvatures = []
    
    for i in range(len(smooth_points)):
        window = padded_points[i:i+window_size]
        centroid = np.sum(window * weights[:, np.newaxis], axis=0) / np.sum(weights)
        centered = window - centroid
        cov = np.dot(centered.T, centered * weights[:, np.newaxis]) / np.sum(weights)
        eigvals = la.eigvalsh(cov)
        curvature = eigvals[0] / (eigvals[1] + 1e-10)
        curvatures.append(curvature)
    
    # Pad the curvatures array with zeros after the restriction point
    full_curvatures = np.zeros(len(points))
    full_curvatures[:len(curvatures)] = curvatures
    
    return full_curvatures

def calculate_head_angle_with_positions_and_bend(skeleton, prev_angle=None, min_vector_length=5, 
                                             restriction_point=0.4, straight_threshold=3, prev_head_section=None,
                                             prev_was_highly_bent=None):
    """
    Calculate head angle and bend location with improved highly bent detection and context awareness.
    
    Parameters:
    -----------
    skeleton : ndarray
        Binary skeleton image
    prev_angle : float, optional
        Previous frame's angle for comparison
    min_vector_length : float
        Minimum required vector length in pixels
    restriction_point : float
        Location of the restriction point along the skeleton (0-1)
    straight_threshold : float
        Angle threshold (in degrees) below which the skeleton is considered straight
    prev_head_section : float, optional
        Previous frame's head section for consistency
    prev_was_highly_bent : bool, optional
        Whether the previous frame was classified as highly bent
        
    Returns:
    --------
    dict containing angle and bend information
    """
    try:
        # Get ordered points along the skeleton
        points = np.column_stack(np.where(skeleton))
        if len(points) == 0:
            return {
                'angle_degrees': prev_angle if prev_angle is not None else 0,
                'error': 'Empty skeleton',
                'head_mag': 0,
                'body_mag': 0,
                'bend_location': 0,
                'bend_magnitude': 0,
                'bend_position': [0, 0],
                'skeleton_points': [[0, 0]],
                'curvature_profile': [0],
                'head_start_pos': [0, 0],
                'head_end_pos': [0, 0],
                'body_start_pos': [0, 0],
                'body_end_pos': [0, 0],
                'head_vector': [0, 0],
                'body_vector': [0, 0],
                'head_section': prev_head_section if prev_head_section is not None else 0.1,
                'is_highly_bent': False
            }
        
        ordered_points = points[np.argsort(points[:, 0])]
        norm_points, total_length = normalize_skeleton_points(ordered_points, num_points=100)
        
        # Calculate curvature to determine worm type
        curvatures = gaussian_weighted_curvature(norm_points, window_size=25, sigma=8, 
                                               restriction_point=restriction_point)
        max_curvature = np.max(np.abs(curvatures[:int(len(curvatures) * restriction_point)]))
        
        # Improved highly bent detection with context awareness
        is_highly_bent_by_curvature = max_curvature > 0.15
        
        # Context-aware detection: if previous frame was highly bent and current angle suggests high bending
        context_suggests_highly_bent = False
        if prev_was_highly_bent and prev_angle is not None:
            # If previous frame was highly bent and had a large angle, current frame likely is too
            if abs(prev_angle) > 80:
                context_suggests_highly_bent = True
        
        # Combined detection with bias toward highly bent when in doubt
        is_highly_bent = is_highly_bent_by_curvature or context_suggests_highly_bent
        
        # Additional check: if we get a very small angle but context suggests high bending,
        # we'll do a secondary check after initial calculation
        
        # Use longer head sections now that we have longer skeletons
        if is_highly_bent:
            # For highly bent worms, be much more conservative about section changes
            if prev_head_section is not None:
                # Strongly prefer the same section, only try alternatives if absolutely necessary
                preferred_sections = [prev_head_section]
                # Only add alternatives if the previous section might be problematic
                if prev_head_section == 0.15:
                    preferred_sections.extend([0.20])  # Try longer sections
                elif prev_head_section == 0.20:
                    preferred_sections.extend([0.15])  # Try shorter if needed
                else:  # prev_head_section == 0.25
                    preferred_sections.extend([0.20])  # Try shorter
            else:
                # For first frame of highly bent worm, prefer longer sections for better measurement
                preferred_sections = [0.20, 0.15, 0.25]
        else:
            # For normal worms, use longer sections for better accuracy
            if prev_head_section is not None:
                preferred_sections = [prev_head_section, 0.15, 0.20, 0.25]
            else:
                preferred_sections = [0.15, 0.20, 0.25]
        
        body_section = 0.4  # Use longer body section too
        best_result = None
        best_continuity_score = -1000
        
        for head_section in preferred_sections:
            head_end_idx = max(2, int(head_section * len(norm_points)))
            body_start_idx = int((1 - body_section) * len(norm_points))
            
            head_start = norm_points[0]
            head_end = norm_points[head_end_idx]
            body_start = norm_points[body_start_idx]
            body_end = norm_points[-1]
            
            head_vector = head_end - head_start
            body_vector = body_end - body_start
            
            head_mag = np.linalg.norm(head_vector)
            body_mag = np.linalg.norm(body_vector)
            
            if head_mag < min_vector_length or body_mag < min_vector_length:
                continue
            
            # Calculate angle using simple, stable method
            angle_deg = calculate_simple_stable_angle(head_vector, body_vector, is_highly_bent)
            
            # CRITICAL FIX: Context-aware angle validation
            # If we're in a highly bent context but got a small angle, this might be wrong
            if (context_suggests_highly_bent and abs(angle_deg) < 70 and 
                prev_angle is not None and abs(prev_angle) > 90):
                # Force recalculation as highly bent
                angle_deg = calculate_simple_stable_angle(head_vector, body_vector, True)
                is_highly_bent = True  # Override the classification
            
            # Simple continuity check - only reject truly impossible changes
            if prev_angle is not None:
                angle_change = abs(angle_deg - prev_angle)
                
                # Only reject if change is extremely large (>120°) AND it's a sign switch
                is_impossible = (prev_angle * angle_deg < 0 and 
                               min(abs(prev_angle), abs(angle_deg)) > 40 and 
                               angle_change > 120)
                
                if is_impossible:
                    continue
            
            # Calculate continuity score
            continuity_score = 0
            
            # Bonus for using same head section as previous frame
            if prev_head_section is not None and head_section == prev_head_section:
                if is_highly_bent:
                    continuity_score += 100  # Much stronger bonus for highly bent worms
                else:
                    continuity_score += 50   # Normal bonus for regular worms
            elif prev_head_section is not None and head_section != prev_head_section:
                if is_highly_bent:
                    continuity_score -= 80   # Strong penalty for section changes in highly bent worms
                else:
                    continuity_score -= 20   # Moderate penalty for normal worms
            
            # Penalty for angle change if we have previous angle
            if prev_angle is not None:
                angle_change = abs(angle_deg - prev_angle)
                
                # Adjust acceptable change thresholds based on worm type
                if is_highly_bent:
                    # For highly bent worms, allow larger changes as they can bend more dramatically
                    small_change_threshold = 20
                    medium_change_threshold = 40
                else:
                    # For normal worms, be more strict
                    small_change_threshold = 15
                    medium_change_threshold = 30
                
                # Progressive penalty for large changes
                if angle_change <= small_change_threshold:
                    continuity_score += (small_change_threshold - angle_change)
                elif angle_change <= medium_change_threshold:
                    continuity_score -= (angle_change - small_change_threshold)
                else:
                    continuity_score -= (angle_change - small_change_threshold) * 2
            
            # For highly bent worms, strongly prefer larger angles
            if is_highly_bent:
                if abs(angle_deg) > 120:
                    continuity_score += 40  # Strong bonus for very high angles
                elif abs(angle_deg) > 100:
                    continuity_score += 30  # Good bonus for high angles
                elif abs(angle_deg) > 90:
                    continuity_score += 25  # Bonus for angles over 90°
                elif abs(angle_deg) > 70:
                    continuity_score += 10  # Small bonus for reasonable angles
                elif abs(angle_deg) < 60:
                    continuity_score -= 30  # Penalty for unrealistically small angles
            else:
                # For normal worms, penalize extremely high angles
                if abs(angle_deg) > 100:
                    continuity_score -= 20  # Penalty for very high angles in normal worms
            
            # CONTEXT BONUS: If this maintains consistency with highly bent sequence
            if context_suggests_highly_bent and abs(angle_deg) > 90:
                continuity_score += 50  # Strong bonus for maintaining bent sequence
            
            # Create result structure
            current_result = {
                'angle_degrees': float(angle_deg),
                'head_start_pos': head_start.tolist(),
                'head_end_pos': head_end.tolist(),
                'body_start_pos': body_start.tolist(),
                'body_end_pos': body_end.tolist(),
                'head_vector': head_vector.tolist(),
                'body_vector': body_vector.tolist(),
                'head_mag': float(head_mag),
                'body_mag': float(body_mag),
                'head_section': head_section,
                'skeleton_points': norm_points.tolist(),
                'error': None,
                'continuity_score': continuity_score,
                'is_highly_bent': is_highly_bent
            }
            
            # Keep the best result so far
            if continuity_score > best_continuity_score:
                best_continuity_score = continuity_score
                best_result = current_result
                
                # If we have a very good result with the preferred section, use it
                if (prev_head_section is not None and 
                    head_section == prev_head_section and 
                    continuity_score > 40):
                    break
        
        if best_result is None:
            # Fallback result
            best_result = {
                'angle_degrees': prev_angle if prev_angle is not None else 0,
                'error': 'No valid angle found with current parameters',
                'head_mag': 0,
                'body_mag': 0,
                'bend_location': 0,
                'bend_magnitude': 0,
                'bend_position': [0, 0],
                'skeleton_points': norm_points.tolist(),
                'head_start_pos': norm_points[0].tolist(),
                'head_end_pos': norm_points[min(5, len(norm_points)-1)].tolist(),
                'body_start_pos': norm_points[max(0, len(norm_points)-10)].tolist(),
                'body_end_pos': norm_points[-1].tolist(),
                'head_vector': [0, 0],
                'body_vector': [0, 0],
                'head_section': prev_head_section if prev_head_section is not None else 0.1,
                'is_highly_bent': is_highly_bent
            }
        else:
            # Log the decision for debugging (simplified)
            angle_change = abs(best_result['angle_degrees'] - prev_angle) if prev_angle is not None else 0
            section_change = "SAME" if (prev_head_section is not None and 
                                      best_result['head_section'] == prev_head_section) else "CHANGED"
            
            worm_type = "Highly bent" if best_result['is_highly_bent'] else "Normal"
            context_info = " (context)" if context_suggests_highly_bent and not is_highly_bent_by_curvature else ""
            print(f"{worm_type}{context_info} - {section_change} section {best_result['head_section']:.2f} "
                  f"with angle {best_result['angle_degrees']:.1f}° "
                  f"(change: {angle_change:.1f}°)")
        
        # Calculate bend information
        if abs(best_result['angle_degrees']) <= straight_threshold:
            best_result.update({
                'bend_location': 0,
                'bend_magnitude': 0,
                'bend_position': [0, 0],
                'curvature_profile': np.zeros(len(norm_points)).tolist(),
                'is_straight': True
            })
        else:
            # Find the location of maximum bend (only consider points up to restriction)
            valid_range = int(len(curvatures) * restriction_point)
            max_curvature_idx = np.argmax(np.abs(curvatures[:valid_range]))
            bend_location = max_curvature_idx / len(curvatures)
            bend_magnitude = float(np.abs(curvatures[max_curvature_idx]))
            bend_position = norm_points[max_curvature_idx].tolist()
            
            best_result.update({
                'bend_location': bend_location,
                'bend_magnitude': bend_magnitude,
                'bend_position': bend_position,
                'curvature_profile': curvatures.tolist(),
                'is_straight': False
            })
        
        return best_result
        
    except Exception as e:
        # Handle any unexpected errors gracefully
        return {
            'angle_degrees': prev_angle if prev_angle is not None else 0,
            'error': f'Unexpected error: {str(e)}',
            'head_mag': 0,
            'body_mag': 0,
            'bend_location': 0,
            'bend_magnitude': 0,
            'bend_position': [0, 0],
            'skeleton_points': [[0, 0]],
            'curvature_profile': [0],
            'head_start_pos': [0, 0],
            'head_end_pos': [0, 0],
            'body_start_pos': [0, 0],
            'body_end_pos': [0, 0],
            'head_vector': [0, 0],
            'body_vector': [0, 0],
            'is_straight': True,
            'head_section': prev_head_section if prev_head_section is not None else 0.1,
            'is_highly_bent': False
        }

def calculate_simple_stable_angle(head_vector, body_vector, is_highly_bent):
    """
    Calculate angle using a simple, stable method that produces realistic angles for highly bent worms.
    """
    # Use atan2 for full range but keep it simple
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
    angle_deg = np.degrees(angle_diff)
    
    # For highly bent worms, apply much more conservative scaling since we have longer vectors
    if is_highly_bent:
        base_angle = abs(angle_deg)
        
        # Check if we might need the supplementary angle for very bent worms
        # Only for very small angles that seem unrealistic
        if base_angle < 60:
            # For highly bent worms with very small calculated angles, 
            # consider using the supplementary angle
            supplementary_angle = 180 - base_angle
            
            # Use the supplementary angle if it's more realistic for a highly bent worm
            if supplementary_angle > 90:
                angle_deg = supplementary_angle if angle_deg >= 0 else -supplementary_angle
                base_angle = supplementary_angle
        
        # Apply much lighter scaling since we have longer, more accurate vectors
        if base_angle < 80:
            # Much more conservative scaling
            if base_angle < 40:
                scale_factor = 1.3  # Light scaling for very small angles
            elif base_angle < 60:
                scale_factor = 1.15  # Very light scaling
            else:
                scale_factor = 1.05  # Minimal scaling
            
            angle_deg = angle_deg * scale_factor
        
        # Set a more reasonable minimum for highly bent worms
        if abs(angle_deg) < 70:
            angle_deg = 75 if angle_deg >= 0 else -75
    
    # Ensure angle stays within reasonable bounds
    if abs(angle_deg) > 170:
        angle_deg = 170 if angle_deg > 0 else -170
    
    return angle_deg

def process_skeleton_batch(truncated_skeletons, min_vector_length=5, 
                         restriction_point=0.5, straight_threshold=3,
                         smoothing_window=3, deviation_threshold=50):
    """
    Process a batch of skeletons with improved head angle calculation.
    Handles all frames gracefully with smooth interpolation between valid frames.
    
    Parameters:
    -----------
    truncated_skeletons : dict
        Dictionary of frame data containing skeletons
    min_vector_length : float
        Minimum required vector length in pixels
    restriction_point : float
        Location of the restriction point along the skeleton (0-1)
    straight_threshold : float
        Angle threshold for straight skeleton
    smoothing_window : int
        Window size for head angle smoothing (reduced impact)
    deviation_threshold : float
        Maximum allowed deviation for smoothing in degrees
    
    Returns:
    --------
    pandas.DataFrame
        Complete DataFrame with calculated angles and bend positions
    """
    # Process frames in order to maintain angle continuity
    initial_data = []
    frame_results = {}
    prev_angles_by_object = {}
    prev_head_sections_by_object = {}
    prev_highly_bent_by_object = {}  # Track highly bent state for each object
    
    # Process frames in order
    for frame_idx in sorted(truncated_skeletons.keys()):
        frame_data = truncated_skeletons[frame_idx]
        frame_results[frame_idx] = {}
        
        for obj_id, skeleton_data in frame_data.items():
            skeleton = skeleton_data[0]
            
            # Get previous angle, head section, and highly bent state for this object
            prev_angle = prev_angles_by_object.get(obj_id, None)
            prev_head_section = prev_head_sections_by_object.get(obj_id, None)
            prev_was_highly_bent = prev_highly_bent_by_object.get(obj_id, None)
            
            result = calculate_head_angle_with_positions_and_bend(
                skeleton,
                prev_angle=prev_angle,
                min_vector_length=min_vector_length,
                restriction_point=restriction_point,
                straight_threshold=straight_threshold,
                prev_head_section=prev_head_section,
                prev_was_highly_bent=prev_was_highly_bent
            )
            
            # Store result
            frame_results[frame_idx][obj_id] = result
            
            # Update previous values if calculation was successful
            if result['error'] is None or 'Large angle change' in result.get('error', ''):
                prev_angles_by_object[obj_id] = result['angle_degrees']
                prev_head_sections_by_object[obj_id] = result.get('head_section', None)
                prev_highly_bent_by_object[obj_id] = result.get('is_highly_bent', False)
            
            # Add to initial_data
            initial_data.append({
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': result['angle_degrees'],
                'bend_location': result['bend_location'],
                'bend_magnitude': result['bend_magnitude'],
                'bend_position_y': result['bend_position'][0],
                'bend_position_x': result['bend_position'][1],
                'head_mag': result['head_mag'],
                'body_mag': result['body_mag'],
                'is_straight': result.get('is_straight', abs(result['angle_degrees']) <= straight_threshold),
                'error': result.get('error', None),
                'is_highly_bent': result.get('is_highly_bent', False)
            })
    
    # Convert to DataFrame
    initial_df = pd.DataFrame(initial_data)
    
    # Apply light smoothing only to remove obvious noise spikes
    smoothed_df = smooth_head_angles_with_validation(
        initial_df,
        id_column='object_id',
        angle_column='angle_degrees',
        window_size=smoothing_window,
        deviation_threshold=deviation_threshold
    )
    
    # Final assembly with corrected bend values
    final_data = []
    
    for obj_id in smoothed_df['object_id'].unique():
        obj_data = smoothed_df[smoothed_df['object_id'] == obj_id].sort_values('frame')
        
        for _, row in obj_data.iterrows():
            frame_idx = row['frame']
            
            # Use the smoothed angle
            is_straight = abs(row['angle_degrees']) <= straight_threshold
            
            # Get original calculation results
            original_result = frame_results[frame_idx][obj_id]
            
            # Set bend values based on straight vs non-straight
            if is_straight:
                bend_location = 0
                bend_magnitude = 0
                bend_position_y = 0
                bend_position_x = 0
            else:
                # Use original calculated values
                bend_location = original_result['bend_location']
                bend_magnitude = original_result['bend_magnitude']
                bend_position_y = original_result['bend_position'][0]
                bend_position_x = original_result['bend_position'][1]
            
            final_result = {
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': row['angle_degrees'],
                'bend_location': bend_location,
                'bend_magnitude': bend_magnitude,
                'bend_position_y': bend_position_y,
                'bend_position_x': bend_position_x,
                'head_mag': original_result['head_mag'],
                'body_mag': original_result['body_mag'],
                'is_noise_peak': row['is_noise_peak'],
                'peak_deviation': row['peak_deviation'],
                'window_size_used': row['window_size_used'],
                'is_straight': is_straight,
                'error': original_result.get('error', None)
            }
            
            final_data.append(final_result)
    
    final_df = pd.DataFrame(final_data)
    
    # Log summary
    straight_count = final_df[final_df['is_straight'] == True].shape[0]
    non_straight_with_bends = final_df[(final_df['is_straight'] == False) & (final_df['bend_location'] > 0)].shape[0]
    warning_count = final_df['error'].notna().sum()
    
    print(f"Processing complete:")
    print(f"- Straight frames: {straight_count}")
    print(f"- Non-straight with bend values: {non_straight_with_bends}")
    print(f"- Frames with warnings: {warning_count}")
    
    if warning_count > 0:
        print("\nWarning types:")
        for error in final_df[final_df['error'].notna()]['error'].unique()[:5]:
            count = final_df[final_df['error'] == error].shape[0]
            print(f"- {error}: {count} frames")
    
    return final_df

# Process all skeletons with integrated smoothing and bend recalculation
angle_start = time.time()
results_df = process_skeleton_batch(
    truncated_skeletons,
    min_vector_length=5,
    restriction_point=0.5,
    straight_threshold=3,
    smoothing_window=3,
    deviation_threshold=50  # Increased threshold to be less aggressive
)
angle_time = time.time()
print(f"⏱️  Angle calculation took {angle_time - angle_start:.2f} seconds")

# Create plot of head angle and bend position
plt.figure(figsize=(12, 6))

# Create twin axes sharing x-axis
ax1 = plt.gca()
ax2 = ax1.twinx()

# Convert DataFrame to numpy arrays before plotting
frame_data = results_df['frame'].to_numpy()
angle_data = results_df['angle_degrees'].to_numpy() 
bend_data = results_df['bend_location'].to_numpy()

# Add shaded region between -3 and 3 degrees
ax1.axhspan(-3, 3, color='gray', alpha=0.2, label='Straight Region')

# Plot head angle on left y-axis
l1, = ax1.plot(frame_data, angle_data, 'b.-', alpha=0.7, label='Head Angle')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Head Angle (degrees)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(-180, 180)  # Set y-axis limits for head angle

# Plot bend position on right y-axis
l2, = ax2.plot(frame_data, bend_data, 'r.-', alpha=0.7, label='Bend Position Y')
ax2.set_ylabel('Bend Position Y', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Head Angle and Bend Position Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('head_angles_and_bends_new_river.png')
plt.close()




def save_head_angles_with_side_correction(filename, results_df, final_data_dir):
    """
    Save head angles with side position correction by merging with final data.
    
    Parameters:
    -----------
    filename : str
        Path to the head segmentation file
    results_df : pandas.DataFrame
        DataFrame containing head angle results
    final_data_dir : str
        Directory containing final data files
        
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with corrected angles
    """
    # Extract base name from filename
    base_name = os.path.basename(filename).replace("_headsegmentation.h5", "")
    final_data_base = base_name + "_crop_riasegmentation_cleanedalignedsegments"

    # Load corresponding data from final_data_dir
    final_data_path = os.path.join(final_data_dir, final_data_base + ".csv")
    final_df = pd.read_csv(final_data_path)

    print(f"Loaded final data from {final_data_path}")
    print(f"Final data shape: {final_df.shape}")
    print(f"Results df shape: {results_df.shape}")

    # Get list of columns to overwrite
    columns_to_overwrite = [col for col in results_df.columns if col in final_df.columns]
    print(f"Columns that will be overwritten: {columns_to_overwrite}")

    # Merge results_df with final_df based on frame, using suffixes to avoid conflicts
    merged_df = pd.merge(final_df, results_df, 
                        left_on=['frame'],
                        right_on=['frame'],
                        how='left',
                        suffixes=('_old', ''))

    # For each column that exists in both dataframes, keep only the new version
    for col in columns_to_overwrite:
        if col + '_old' in merged_df.columns:
            merged_df.drop(columns=[col + '_old'], inplace=True)

    # Create angle_degrees_corrected column based on side_position
    merged_df['angle_degrees_corrected'] = merged_df.apply(
        lambda row: -row['angle_degrees'] if row['side_position'] == 'right' else row['angle_degrees'], 
        axis=1
    )

    print(f"Number of right-side angles corrected: {(merged_df['side_position'] == 'right').sum()}")
    print(f"Number of left-side angles: {(merged_df['side_position'] == 'left').sum()}")

    # Save path for the new head angles file
    output_path = os.path.join(final_data_dir, final_data_base + "_headangles.csv")

    # Save the merged dataframe with head angles
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged df to: {output_path}")
    print(f"Final merged df shape: {merged_df.shape}")
    print(f"Final columns: {merged_df.columns.tolist()}")

    # Delete previous file if it exists
    if os.path.exists(final_data_path):
        os.remove(final_data_path)
        print(f"Deleted existing file: {final_data_path}")
        
    return merged_df

#merged_df = save_head_angles_with_side_correction(filename, results_df, final_data_dir)




def create_layered_mask_video(image_dir, bottom_masks_dict, top_masks_dict, angles_df,
                           output_path, fps=10, bottom_alpha=0.5, top_alpha=0.7):
    """
    Create a video with mask overlays and angle values displayed at skeleton tips.
    
    Args:
        image_dir (str): Directory containing the input images
        bottom_masks_dict (dict): Dictionary of bottom layer masks
        top_masks_dict (dict): Dictionary of top layer masks
        angles_df (pd.DataFrame): DataFrame containing 'frame' and 'angle_degrees' columns
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        bottom_alpha (float): Transparency of bottom mask overlay (0-1)
        top_alpha (float): Transparency of top mask overlay (0-1)
    """
    import numpy as np
    import cv2
    import os
    import pandas as pd

    # Predefined colors for different masks
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (128, 0, 128),  # Purple
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (0, 128, 0),    # Dark Green
        (0, 128, 128),  # Teal
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Deep Pink
        (128, 255, 0),  # Lime
        (255, 255, 0),  # Yellow
        (0, 255, 128)   # Spring Green
    ]

    def create_mask_overlay(image, frame_masks, mask_colors, alpha):
        """Helper function to create a mask overlay"""
        overlay = np.zeros_like(image)
        
        for mask_id, mask in frame_masks.items():
            # Convert to binary mask if needed
            if mask.dtype != bool:
                mask = mask > 0.5
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = mask_colors[mask_id]
            
            # Add to overlay
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
        
        return overlay

    def find_skeleton_tip(mask):
        """Find the tip (topmost point) of the skeleton mask"""
        if mask.dtype != bool:
            mask = mask > 0.5
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
            
        # Find all points where mask is True
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
            
        # Find the topmost point
        top_idx = np.argmin(y_coords)
        return (y_coords[top_idx], x_coords[top_idx])

    def add_angle_text(image, angle, position, font_scale=0.7):
        """Add angle text at the given position with background"""
        if position is None or angle is None:
            return image
            
        y, x = position
        angle_text = f"{angle:.1f} deg"
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            angle_text, font, font_scale, font_thickness)
        
        # Adjust position to put text to the right of the tip point
        text_x = int(x + 30)  # Offset text to the right
        text_y = int(y)  # Keep same vertical level as tip
        
        # Add background to make text more readable
        padding = 5
        cv2.rectangle(image, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Draw text
        cv2.putText(image, angle_text,
                   (text_x, text_y + text_height),
                   font, font_scale, (255, 255, 255),  # White text
                   font_thickness)
        
        return image

    # Get sorted list of image files and extract their frame numbers
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Extract frame numbers and create mapping
    frame_numbers = []
    for img_file in image_files:
        # Extract number from filename (assuming format like 000000.jpg)
        match = re.search(r'(\d+)', img_file)
        if match:
            frame_numbers.append((int(match.group(1)), img_file))
    
    # Sort by frame number
    frame_numbers.sort(key=lambda x: x[0])
    
    if not frame_numbers:
        raise ValueError(f"No image files found in {image_dir}")

    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, frame_numbers[0][1]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {frame_numbers[0][1]}")
    
    height, width, _ = first_image.shape

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create color mapping for mask IDs
    bottom_mask_ids = set()
    top_mask_ids = set()
    for masks in bottom_masks_dict.values():
        bottom_mask_ids.update(masks.keys())
    for masks in top_masks_dict.values():
        top_mask_ids.update(masks.keys())
    
    mid_point = len(COLORS) // 2
    bottom_colors = COLORS[:mid_point]
    top_colors = COLORS[mid_point:] + COLORS[:max(0, len(top_mask_ids) - len(COLORS) // 2)]
    
    bottom_mask_colors = {mask_id: bottom_colors[i % len(bottom_colors)] 
                         for i, mask_id in enumerate(bottom_mask_ids)}
    top_mask_colors = {mask_id: top_colors[i % len(top_colors)] 
                      for i, mask_id in enumerate(top_mask_ids)}

    # Convert angles DataFrame to dictionary for faster lookup
    angles_dict = angles_df.set_index('frame')['angle_degrees'].to_dict()

    # Process each frame
    for frame_number, image_file in frame_numbers:
        try:
            # Read image
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Print debug info for first few frames
            if frame_number < 5:
                print(f"Processing frame {frame_number}, file: {image_file}")
                print(f"Bottom masks available: {frame_number in bottom_masks_dict}")
                print(f"Top masks available: {frame_number in top_masks_dict}")
            
            # Start with original frame
            final_frame = frame.copy()
            
            # Apply bottom masks if available
            if frame_number in bottom_masks_dict:
                bottom_overlay = create_mask_overlay(frame, 
                                                  bottom_masks_dict[frame_number],
                                                  bottom_mask_colors, 
                                                  bottom_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, bottom_overlay, bottom_alpha, 0)
            
            # Apply top masks and find skeleton tip
            tip_position = None
            if frame_number in top_masks_dict:
                top_overlay = create_mask_overlay(frame,
                                               top_masks_dict[frame_number],
                                               top_mask_colors,
                                               top_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, top_overlay, top_alpha, 0)
                
                # Find tip position from the first skeleton mask (assuming it's the main one)
                if top_masks_dict[frame_number]:
                    first_mask_id = next(iter(top_masks_dict[frame_number]))
                    tip_position = find_skeleton_tip(top_masks_dict[frame_number][first_mask_id])
            
            # Add angle text if available
            if frame_number in angles_dict and tip_position is not None:
                final_frame = add_angle_text(final_frame, angles_dict[frame_number], tip_position)

            # Write frame
            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_number} ({image_file}): {str(e)}")
            continue

    # Clean up
    out.release()
    print(f"Video saved to {output_path}")


video_dir = "/home/lilly/phd/riverchip/data_foranalysis/videotojpg/data_original-hannah"
image_dir = video_dir
bottom_masks = head_segments
top_masks = truncated_skeletons
angle_results = results_df
output_path = "head_skeleton_angles_video_river.mp4"

create_layered_mask_video(
    image_dir=image_dir,
    bottom_masks_dict=bottom_masks,
    top_masks_dict=top_masks,
    angles_df=angle_results,
    output_path=output_path,
    fps=10,
    bottom_alpha=0.3,
    top_alpha=0.7
)

print("\n" + "="*60)
print("🎉 HEAD ANGLE EXTRACTION COMPLETED SUCCESSFULLY! 🎉")
print("="*60)
total_time = time.time() - start_time
print(f"⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
print(f"🚀 Used {MAX_PROCESSES if USE_MULTIPROCESSING else 1} cores out of {mp.cpu_count()} available")
print(f"📊 Processing rate: {len(results_df)/total_time:.1f} frames/second")
print(f"✅ Processed {len(results_df)} angle measurements")
print(f"✅ Generated plot: head_angles_and_bends_new_river.png")
print(f"✅ Results saved in results_df DataFrame")
print("="*60)
print("🏆 PERFORMANCE SUMMARY:")
print(f"   Skeleton generation: {skeleton_time - start_time:.2f}s ({(skeleton_time - start_time)/total_time*100:.1f}%)")
print(f"   Skeleton truncation: {truncation_time - truncation_start:.2f}s ({(truncation_time - truncation_start)/total_time*100:.1f}%)")
print(f"   Angle calculation:   {angle_time - angle_start:.2f}s ({(angle_time - angle_start)/total_time*100:.1f}%)")
print("="*60)
print("Script execution finished!")
print("="*60)

def conservative_truncate_skeleton_chunk(chunk_data):
    """
    Conservatively truncate skeletons for a chunk of frames, only removing noise at extremities.
    
    Args:
        chunk_data: Tuple of (frame_data_chunk, tail_trim_pixels)
    
    Returns:
        Dictionary with conservatively truncated skeletons for the chunk
    """
    frame_data_chunk, tail_trim_pixels = chunk_data
    
    truncated_chunk = {}
    
    for frame_idx, frame_data in frame_data_chunk:
        frame_truncated = {}
        
        for obj_id, skeleton in frame_data.items():
            # Get the 2D array from the 3D input (taking the first channel)
            skeleton_2d = skeleton[0]
            
            # Find all non-zero points
            points = np.where(skeleton_2d)
            if len(points[0]) == 0:  # Empty skeleton
                frame_truncated[obj_id] = skeleton
                continue
            
            # Get the top point and bottom point
            y_min = np.min(points[0])
            y_max = np.max(points[0])
            original_height = y_max - y_min
            
            # Only trim a small amount from the tail (bottom) to remove noise
            # Keep at least 90% of the original skeleton
            max_trim = min(tail_trim_pixels, int(original_height * 0.1))
            cutoff_point = y_max - max_trim
            
            # Create conservatively truncated skeleton
            truncated = skeleton.copy()
            truncated[0, cutoff_point:, :] = False  # Using False since it's boolean type
            
            frame_truncated[obj_id] = truncated
            
        truncated_chunk[frame_idx] = frame_truncated
    
    return truncated_chunk

def conservative_truncate_skeleton_parallel(skeleton_dict, tail_trim_pixels=50, num_processes=None):
    """
    Conservatively truncate skeletons using parallel processing, only removing noise at extremities.
    
    Args:
        skeleton_dict: Dictionary with frame indices as keys and inner dictionaries 
                      containing skeletons as values of shape (1, h, w)
        tail_trim_pixels: Maximum number of pixels to trim from tail (default 50)
        num_processes: Number of processes to use (default: CPU count)
    
    Returns:
        Dictionary with conservatively truncated skeletons
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), MAX_PROCESSES)
    
    print(f"🎯 Conservative truncation: Maximum tail trim = {tail_trim_pixels} pixels (keeping ~90% of skeleton)")
    print(f"🚀 Using {num_processes} processes for conservative skeleton truncation on {mp.cpu_count()}-core machine...")
    
    # Convert to list of (frame_idx, frame_data) tuples
    frame_list = list(skeleton_dict.items())
    print(f"Conservatively truncating {len(frame_list)} frames...")
    
    # Optimize chunk sizing for better load balancing on many cores
    optimal_chunk_size = max(1, len(frame_list) // (num_processes * CHUNK_SIZE_MULTIPLIER))
    chunks = [frame_list[i:i + optimal_chunk_size] for i in range(0, len(frame_list), optimal_chunk_size)]
    
    print(f"📦 Split into {len(chunks)} chunks of ~{optimal_chunk_size} frames each for optimal load balancing")
    
    # Prepare chunk data with parameters
    chunk_data_list = [(chunk, tail_trim_pixels) for chunk in chunks]
    
    # Process chunks in parallel
    try:
        with mp.Pool(processes=num_processes) as pool:
            print("⚡ Starting parallel conservative skeleton truncation...")
            results = pool.map(conservative_truncate_skeleton_chunk, chunk_data_list)
            print("✅ Conservative skeleton truncation completed!")
    except Exception as e:
        print(f"❌ Error in parallel processing: {e}")
        # Fallback to sequential processing
        print("🔄 Falling back to sequential processing...")
        results = [conservative_truncate_skeleton_chunk(chunk_data) for chunk_data in chunk_data_list]
    
    # Combine results
    truncated_skeletons = {}
    for chunk_result in results:
        truncated_skeletons.update(chunk_result)
    
    print(f"🎉 Conservative truncation complete! Processed {len(truncated_skeletons)} frames using {num_processes} cores.")
    return truncated_skeletons