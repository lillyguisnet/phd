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
    
    # For each video, check if it has a corresponding file in final_data_dir but no _headangles.csv
    processable_videos = []
    for video in all_videos:
        # Extract base name by removing _headsegmentation.h5
        base_name = video.replace("_headsegmentation.h5", "")
        
        # Add _crop_riasegmentation to match final data naming
        final_data_base = base_name + "_crop_riasegmentation"
        
        # Check if corresponding cleanedalignedsegments exists but headangles.csv doesn't
        if os.path.exists(os.path.join(final_data_dir, final_data_base + "_cleanedalignedsegments.csv")) and \
           not os.path.exists(os.path.join(final_data_dir, final_data_base + "_headangles.csv")):
            processable_videos.append(video)
    
    if not processable_videos:
        raise ValueError("No videos found that need head angle processing.")
    
    return os.path.join(head_segmentation_dir, random.choice(processable_videos))

head_segmentation_dir = "/home/lilly/phd/ria/data_analyzed/AG_WT/head_segmentation/"
final_data_dir = "/home/lilly/phd/ria/data_analyzed/AG_WT/final_data/"

filename = get_random_unprocessed_video(head_segmentation_dir, final_data_dir)
head_segments = load_cleaned_segments_from_h5(filename)


###Get skeletons from masks
def get_skeleton(mask):
    return morphology.skeletonize(mask)

def process_all_frames(head_segments):
    """
    Process all frames to generate skeletons from masks and calculate skeleton statistics.

    Args:
        head_segments: Dictionary with frame indices as keys and inner dictionaries 
                      containing object masks as values

    Returns:
        Tuple containing:
        - Dictionary with frame indices as keys and dictionaries of skeletons as values
        - Dictionary with skeleton size statistics
    """
    skeletons = {}
    skeleton_sizes = []

    for frame_idx, frame_data in head_segments.items():
        frame_skeletons = {}

        for obj_id, mask in frame_data.items():
            # Generate skeleton from mask
            skeleton = get_skeleton(mask)
            frame_skeletons[obj_id] = skeleton
            
            # Track skeleton size (number of pixels)
            size = np.sum(skeleton)
            skeleton_sizes.append(size)

        skeletons[frame_idx] = frame_skeletons

        if frame_idx % 100 == 0:  # Progress update every 100 frames
            print(f"Processed frame {frame_idx}")

    # Calculate statistics
    stats = {
        'min_size': np.min(skeleton_sizes),
        'max_size': np.max(skeleton_sizes), 
        'mean_size': np.mean(skeleton_sizes),
        'median_size': np.median(skeleton_sizes),
        'std_size': np.std(skeleton_sizes)
    }

    print("\nSkeleton Statistics:")
    print(f"Minimum size: {stats['min_size']:.1f} pixels")
    print(f"Maximum size: {stats['max_size']:.1f} pixels") 
    print(f"Mean size: {stats['mean_size']:.1f} pixels")
    print(f"Median size: {stats['median_size']:.1f} pixels")
    print(f"Standard deviation: {stats['std_size']:.1f} pixels")

    return skeletons, stats

skeletons, skeleton_stats = process_all_frames(head_segments)



###Truncate skeletons
def truncate_skeleton_fixed(skeleton_dict, keep_pixels=150):
    """
    Keeps only the top specified number of pixels of skeletons in a dictionary.
    
    Args:
        skeleton_dict: Dictionary with frame indices as keys and inner dictionaries 
                      containing skeletons as values of shape (1, h, w)
        keep_pixels: Number of pixels to keep from the top (default 150)
    
    Returns:
        Dictionary with truncated skeletons
    """
    truncated_skeletons = {}
    
    for frame_idx, frame_data in skeleton_dict.items():
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
            
            # Verify the truncation
            new_points = np.where(truncated[0])
            new_height = np.max(new_points[0]) - np.min(new_points[0])
            
            print(f"Frame {frame_idx}, Object {obj_id}:")
            print(f"Original height: {original_height}")
            print(f"New height: {new_height}")
            print(f"Top point: {y_min}")
            print(f"Cutoff point: {cutoff_point}")
            print("------------------------")
            
            frame_truncated[obj_id] = truncated
            
        truncated_skeletons[frame_idx] = frame_truncated
    
    return truncated_skeletons

# Now use it on your skeletons dictionary
truncated_skeletons = truncate_skeleton_fixed(skeletons, keep_pixels=200)




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
                                             restriction_point=0.4, straight_threshold=3):
    """
    Calculate head angle and bend location along the skeleton.
    
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
        
    Returns:
    --------
    dict containing:
        - angle_degrees: calculated angle
        - head positions and vectors (as before)
        - bend_location: relative position along skeleton (0-1)
        - bend_magnitude: strength of the bend
        - bend_position: (y,x) coordinates of maximum bend point
    """
    # Get ordered points along the skeleton
    points = np.column_stack(np.where(skeleton))
    ordered_points = points[np.argsort(points[:, 0])]
    
    # Normalize points spacing
    norm_points, total_length = normalize_skeleton_points(ordered_points, num_points=100)
    
    # First calculate the head angle before modifying anything
    head_sections = [0.05, 0.08, 0.1, 0.15]
    body_section = 0.3
    
    best_angle_result = None
    max_angle_magnitude = 0
    debug_info = []
    
    for head_section in head_sections:
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
            debug_info.append(f"Section {head_section}: Vector too short - head_mag={head_mag:.2f}, body_mag={body_mag:.2f}")
            continue
        
        dot_product = np.dot(head_vector, body_vector)
        cos_angle = np.clip(dot_product / (head_mag * body_mag), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        cross_product = np.cross(body_vector, head_vector)
        if cross_product < 0:
            angle_deg = -angle_deg
        
        if prev_angle is not None:
            angle_change = abs(angle_deg - prev_angle)
            if angle_change > 25:
                debug_info.append(f"Section {head_section}: Angle change too large - {angle_change:.2f} degrees")
                continue
        
        if abs(angle_deg) > max_angle_magnitude:
            max_angle_magnitude = abs(angle_deg)
            best_angle_result = {
                'angle_degrees': float(angle_deg),
                'head_start_pos': head_start.tolist(),
                'head_end_pos': head_end.tolist(),
                'body_start_pos': body_start.tolist(),
                'body_end_pos': body_end.tolist(),
                'head_vector': head_vector.tolist(),
                'body_vector': body_vector.tolist(),
                'head_mag': float(head_mag),
                'body_mag': float(body_mag),
                'head_section': head_section
            }
    
    if best_angle_result is None:
        return {
            'angle_degrees': prev_angle if prev_angle is not None else 0,
            'error': f"No valid angle found - Debug info: {'; '.join(debug_info)}",
            'head_mag': 0,
            'body_mag': 0,
            'bend_location': 0,
            'bend_magnitude': 0,
            'bend_position': norm_points[0].tolist(),
            'curvature_profile': [],
            'skeleton_points': norm_points.tolist()
        }
    
    # Calculate curvature for all cases
    curvatures = gaussian_weighted_curvature(norm_points, window_size=25, sigma=8, 
                                           restriction_point=restriction_point)
    
    # Now handle bend detection based on the angle result
    if abs(best_angle_result['angle_degrees']) <= straight_threshold:
        # If skeleton is straight, set bend_location to 0
        bend_location = 0
        bend_magnitude = 0
        bend_position = norm_points[0].tolist()
    else:
        # Find the location of maximum bend (only consider points up to restriction)
        valid_range = int(len(curvatures) * restriction_point)
        max_curvature_idx = np.argmax(np.abs(curvatures[:valid_range]))
        bend_location = max_curvature_idx / len(curvatures)  # Normalized position (0-1)
        bend_magnitude = float(np.abs(curvatures[max_curvature_idx]))
        bend_position = norm_points[max_curvature_idx].tolist()
    
    # Original head angle calculation logic
    head_sections = [0.05, 0.08, 0.1, 0.15]
    body_section = 0.3
    
    best_result = None
    max_angle_magnitude = 0
    
    for head_section in head_sections:
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
        
        dot_product = np.dot(head_vector, body_vector)
        cos_angle = np.clip(dot_product / (head_mag * body_mag), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        cross_product = np.cross(body_vector, head_vector)
        if cross_product < 0:
            angle_deg = -angle_deg
        
        if prev_angle is not None:
            angle_change = abs(angle_deg - prev_angle)
            if angle_change > 25:
                continue
        
        if abs(angle_deg) > max_angle_magnitude:
            max_angle_magnitude = abs(angle_deg)
            best_result = {
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
                'bend_location': bend_location,  # Relative position along skeleton (0-1)
                'bend_magnitude': bend_magnitude,  # Strength of the bend
                'bend_position': bend_position,  # (y,x) coordinates of maximum bend
                'curvature_profile': curvatures.tolist(),  # Full curvature profile
                'error': None
            }
    
    if best_result is None:
        return {
            'angle_degrees': prev_angle if prev_angle is not None else 0,
            'error': 'No valid angle found',
            'head_mag': 0,
            'body_mag': 0,
            'bend_location': None,
            'bend_magnitude': None,
            'bend_position': None
        }
    
    return best_result

def interpolate_straight_frames(df, straight_threshold=3):
    """
    Interpolate bend locations ONLY for straight frames, keeping all other values unchanged.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing angle and bend data
    straight_threshold : float
        Maximum absolute angle to consider as straight (default: 3 degrees)
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Work on each object separately
    for obj_id in df['object_id'].unique():
        # Get data for this object
        mask = df['object_id'] == obj_id
        obj_data = df[mask].copy()
        
        # Identify straight frames using the same threshold as in angle calculation
        straight_mask = (obj_data['angle_degrees'].abs() <= straight_threshold)
        
        if straight_mask.any():
            # Find runs of straight frames
            runs = []
            start = None
            for i, is_straight in enumerate(straight_mask):
                if is_straight and start is None:
                    start = i
                elif not is_straight and start is not None:
                    runs.append((start, i-1))
                    start = None
            if start is not None:  # Handle case where last run goes to end
                runs.append((start, len(straight_mask)-1))
            
            # Process each run of straight frames
            for start_idx, end_idx in runs:
                # Find nearest actual bends before and after
                prev_idx = start_idx - 1
                next_idx = end_idx + 1
                
                if prev_idx >= 0 and next_idx < len(obj_data):
                    # Both bounds exist - interpolate between them
                    prev_loc = obj_data.iloc[prev_idx]['bend_location']
                    next_loc = obj_data.iloc[next_idx]['bend_location']
                    
                    # Only interpolate the straight frames
                    for i in range(start_idx, end_idx + 1):
                        weight = (i - start_idx + 1) / (end_idx - start_idx + 2)
                        interp_val = prev_loc * (1 - weight) + next_loc * weight
                        obj_data.iloc[i, obj_data.columns.get_loc('bend_location')] = interp_val
                
                elif prev_idx >= 0:
                    # Only previous value exists
                    prev_loc = obj_data.iloc[prev_idx]['bend_location']
                    obj_data.iloc[start_idx:end_idx+1, obj_data.columns.get_loc('bend_location')] = prev_loc
                
                elif next_idx < len(obj_data):
                    # Only next value exists
                    next_loc = obj_data.iloc[next_idx]['bend_location']
                    obj_data.iloc[start_idx:end_idx+1, obj_data.columns.get_loc('bend_location')] = next_loc
        
        # Update only the straight frames in the main dataframe
        df.loc[mask] = obj_data
    
    return df

def plot_skeleton_with_bend(frame_idx, obj_id, truncated_skeletons, head_angle_results, 
                          ax=None, show_vectors=True, show_title=True):
    """
    Plot a single skeleton frame with bend location and head/body vectors.
    
    Parameters:
    -----------
    frame_idx : int
        Frame number to plot
    obj_id : int/str
        Object ID to plot
    truncated_skeletons : dict
        Dictionary containing skeleton data
    head_angle_results : dict
        Dictionary containing analysis results
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    show_vectors : bool
        Whether to show head and body direction vectors
    show_title : bool
        Whether to show plot title
    """
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get skeleton and results
    skeleton = truncated_skeletons[frame_idx][obj_id][0]
    result = head_angle_results[frame_idx][obj_id]
    
    # Get skeleton points and convert lists back to numpy arrays
    skeleton_points = np.array(result['skeleton_points'])
    
    # Plot the skeleton points
    ax.plot(skeleton_points[:, 1], skeleton_points[:, 0], 'b-', alpha=0.5, linewidth=1)
    
    # Plot bend location
    bend_pos = result['bend_position']
    ax.scatter(bend_pos[1], bend_pos[0], c='r', s=100, marker='o', label='Max Bend')
    
    if show_vectors:
        # Plot head vector
        head_start = np.array(result['head_start_pos'])
        head_vector = np.array(result['head_vector']) * 0.5  # Scale vector for visualization
        ax.arrow(head_start[1], head_start[0], 
                head_vector[1], head_vector[0],
                head_width=2, head_length=2, fc='g', ec='g', label='Head Vector')
        
        # Plot body vector
        body_start = np.array(result['body_start_pos'])
        body_vector = np.array(result['body_vector']) * 0.5  # Scale vector for visualization
        ax.arrow(body_start[1], body_start[0], 
                body_vector[1], body_vector[0],
                head_width=2, head_length=2, fc='purple', ec='purple', label='Body Vector')
    
    # Add markers for head and tail
    ax.scatter(skeleton_points[0, 1], skeleton_points[0, 0], c='g', s=100, marker='^', label='Head')
    ax.scatter(skeleton_points[-1, 1], skeleton_points[-1, 0], c='orange', s=100, marker='v', label='Tail')
    
    if show_title:
        ax.set_title(f'Frame {frame_idx}, Object {obj_id}\n'
                    f'Angle: {result["angle_degrees"]:.1f}°, '
                    f'Bend Location: {result["bend_location"]:.2f}, '
                    f'Magnitude: {result["bend_magnitude"]:.2f}')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    return ax



def process_skeleton_batch(truncated_skeletons, min_vector_length=5, 
                         restriction_point=0.5, straight_threshold=3,
                         smoothing_window=3, deviation_threshold=15):
    """
    Process a batch of skeletons with head angle smoothing and bend recalculation.
    
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
        Window size for head angle smoothing
    deviation_threshold : float
        Maximum allowed deviation for smoothing in degrees
    
    Returns:
    --------
    pandas.DataFrame
        Complete DataFrame with smoothed angles and recalculated bend positions
    """
    # First pass: Calculate initial angles
    initial_data = []
    dropped_frames = []
    prev_angles = {}
    
    for frame_idx in sorted(truncated_skeletons.keys()):
        frame_data = truncated_skeletons[frame_idx]
        
        for obj_id, skeleton_data in frame_data.items():
            skeleton = skeleton_data[0]
            prev_angle = prev_angles.get(obj_id)
            
            result = calculate_head_angle_with_positions_and_bend(
                skeleton,
                prev_angle=prev_angle,
                min_vector_length=min_vector_length,
                restriction_point=restriction_point,
                straight_threshold=straight_threshold
            )
            
            if result['angle_degrees'] is None or result['error'] is not None:
                dropped_frames.append({
                    'frame': frame_idx,
                    'object_id': obj_id,
                    'reason': result['error']
                })
                continue
                
            prev_angles[obj_id] = result['angle_degrees']
                
            initial_data.append({
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': result['angle_degrees'],
                'head_mag': result['head_mag'],
                'body_mag': result['body_mag']
            })
    
    print(f"Dropped {len(dropped_frames)} frames")
    for drop in dropped_frames:  # Show first 10 dropped frames
        print(f"Frame {drop['frame']}: {drop['reason']}")
    
    # Convert to DataFrame and apply head angle smoothing
    initial_df = pd.DataFrame(initial_data)
    smoothed_df = smooth_head_angles_with_validation(
        initial_df,
        id_column='object_id',
        angle_column='angle_degrees',
        window_size=smoothing_window,
        deviation_threshold=deviation_threshold
    )
    
    # Second pass: Recalculate bend positions using smoothed angles
    final_data = []
    
    # Group by object_id to maintain continuity
    for obj_id in smoothed_df['object_id'].unique():
        obj_data = smoothed_df[smoothed_df['object_id'] == obj_id]
        prev_angle = None
        
        for _, row in obj_data.iterrows():
            frame_idx = row['frame']
            skeleton = truncated_skeletons[frame_idx][obj_id][0]
            
            # Calculate bends using the smoothed angle as reference
            result = calculate_head_angle_with_positions_and_bend(
                skeleton,
                prev_angle=prev_angle,
                min_vector_length=min_vector_length,
                restriction_point=restriction_point,
                straight_threshold=straight_threshold
            )
            
            prev_angle = row['angle_degrees']  # Use smoothed angle for continuity
            
            final_data.append({
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': row['angle_degrees'],  # Keep smoothed angle
                'bend_location': result['bend_location'],
                'bend_magnitude': result['bend_magnitude'],
                'bend_position_y': result['bend_position'][0] if result['bend_position'] else None,
                'bend_position_x': result['bend_position'][1] if result['bend_position'] else None,
                'head_mag': result['head_mag'],
                'body_mag': result['body_mag'],
                'is_noise_peak': row['is_noise_peak'],
                'peak_deviation': row['peak_deviation'],
                'window_size_used': row['window_size_used']
            })
    
    final_df = pd.DataFrame(final_data)
    
    # Apply bend interpolation for straight frames
    final_df = interpolate_straight_frames(final_df, straight_threshold=straight_threshold)
    
    return final_df


# Process all skeletons with integrated smoothing and bend recalculation
results_df = process_skeleton_batch(
    truncated_skeletons,
    min_vector_length=5,
    restriction_point=0.5,
    straight_threshold=3,
    smoothing_window=3,
    deviation_threshold=12
)


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
ax1.set_ylim(-90, 90)  # Set y-axis limits for head angle

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
plt.savefig('head_angles_and_bends_new.png')
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
    str
        Path to the saved head angles file
    """
    # Extract base name from filename
    base_name = os.path.basename(filename).replace("_headsegmentation.h5", "")
    final_data_base = base_name + "_crop_riasegmentation_cleanedalignedsegments"

    # Load corresponding data from final_data_dir
    final_data_path = os.path.join(final_data_dir, final_data_base + ".csv")
    final_df = pd.read_csv(final_data_path)

    print(f"Loaded final data from {final_data_path}")
    print(f"Final data shape: {final_df.shape}")

    # Merge results_df with final_df based on frame and object_id
    merged_df = pd.merge(final_df, results_df, 
                        left_on=['frame'],
                        right_on=['frame'],
                        how='left')

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

    # Delete previous file if it exists
    if os.path.exists(final_data_path):
        os.remove(final_data_path)
        print(f"Deleted existing file: {final_data_path}")
        
    return merged_df

merged_df = save_head_angles_with_side_correction(filename, results_df, final_data_dir)




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

    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
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
    angles_dict = angles_df.set_index('frame')['angle_degrees_corrected'].to_dict()

    # Process each frame
    for frame_idx, image_file in enumerate(image_files):
        try:
            # Read image
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Start with original frame
            final_frame = frame.copy()
            
            # Apply bottom masks if available
            if frame_idx in bottom_masks_dict:
                bottom_overlay = create_mask_overlay(frame, 
                                                  bottom_masks_dict[frame_idx],
                                                  bottom_mask_colors, 
                                                  bottom_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, bottom_overlay, bottom_alpha, 0)
            
            # Apply top masks and find skeleton tip
            tip_position = None
            if frame_idx in top_masks_dict:
                top_overlay = create_mask_overlay(frame,
                                               top_masks_dict[frame_idx],
                                               top_mask_colors,
                                               top_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, top_overlay, top_alpha, 0)
                
                # Find tip position from the first skeleton mask (assuming it's the main one)
                if top_masks_dict[frame_idx]:
                    first_mask_id = next(iter(top_masks_dict[frame_idx]))
                    tip_position = find_skeleton_tip(top_masks_dict[frame_idx][first_mask_id])
            
            # Add angle text if available
            if frame_idx in angles_dict and tip_position is not None:
                final_frame = add_angle_text(final_frame, angles_dict[frame_idx], tip_position)

            # Write frame
            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue

    # Clean up
    out.release()
    print(f"Video saved to {output_path}")


"""
video_dir = "/home/lilly/phd/ria/data_foranalysis/AG_WT/videotojpg/AG_WT-MMH99_10s_20190220_04"
image_dir = video_dir
bottom_masks = head_segments
top_masks = truncated_skeletons
angle_results = merged_df
output_path = "head_skeleton_angles_video2.mp4"

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
"""