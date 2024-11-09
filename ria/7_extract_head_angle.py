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





import numpy as np
from scipy import interpolate, ndimage
from collections import defaultdict





def normalize_skeleton_points(points, num_points=100):
    """
    Resample skeleton points to have uniform spacing.
    """
    # Calculate cumulative distances along the skeleton
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dists[-1]
    
    # Create evenly spaced points
    even_dists = np.linspace(0, total_length, num_points)
    
    # Interpolate new points
    new_points = np.zeros((num_points, 2))
    for i in range(2):  # For both x and y coordinates
        new_points[:, i] = np.interp(even_dists, cum_dists, points[:, i])
    
    return new_points, total_length

def calculate_head_angle_with_positions(skeleton, prev_angle=None, min_vector_length=5):
    """
    Calculate head angle and return vector positions on the skeleton.
    
    Parameters:
    -----------
    skeleton : ndarray
        Binary skeleton image
    prev_angle : float, optional
        Previous frame's angle for comparison
    min_vector_length : float
        Minimum required vector length in pixels
        
    Returns:
    --------
    dict containing:
        - angle_degrees: calculated angle
        - head_start_pos: (y,x) coordinates of head vector start
        - head_end_pos: (y,x) coordinates of head vector end
        - body_start_pos: (y,x) coordinates of body vector start
        - body_end_pos: (y,x) coordinates of body vector end
        - all skeleton points for reference
    """
    # Get ordered points along the skeleton
    points = np.column_stack(np.where(skeleton))
    ordered_points = points[np.argsort(points[:, 0])]
    
    # Normalize points spacing
    norm_points, total_length = normalize_skeleton_points(ordered_points, num_points=100)
    
    # Try different head sections to capture more extreme angles
    head_sections = [0.05, 0.08, 0.1, 0.15]  # Try 5%, 8%, 10%, and 15% for head
    body_section = 0.3   # Use last 30% for body direction
    
    best_result = None
    max_angle_magnitude = 0
    
    for head_section in head_sections:
        # Calculate indices for head and body vectors
        head_end_idx = max(2, int(head_section * len(norm_points)))  # Ensure at least 2 points
        body_start_idx = int((1 - body_section) * len(norm_points))
        
        # Get the actual points for vectors
        head_start = norm_points[0]
        head_end = norm_points[head_end_idx]
        body_start = norm_points[body_start_idx]
        body_end = norm_points[-1]
        
        # Calculate vectors
        head_vector = head_end - head_start
        body_vector = body_end - body_start
        
        # Check vector magnitudes
        head_mag = np.linalg.norm(head_vector)
        body_mag = np.linalg.norm(body_vector)
        
        if head_mag < min_vector_length or body_mag < min_vector_length:
            continue
        
        # Calculate angle
        dot_product = np.dot(head_vector, body_vector)
        cos_angle = np.clip(dot_product / (head_mag * body_mag), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # Determine direction using cross product
        cross_product = np.cross(body_vector, head_vector)
        if cross_product < 0:
            angle_deg = -angle_deg
        
        # If we have a previous angle, check if change is reasonable
        if prev_angle is not None:
            angle_change = abs(angle_deg - prev_angle)
            if angle_change > 25:  # Slightly increased threshold
                continue
        
        # Keep the result that gives us the largest angle magnitude
        if abs(angle_deg) > max_angle_magnitude:
            max_angle_magnitude = abs(angle_deg)
            best_result = {
                'angle_degrees': float(angle_deg),
                'head_start_pos': head_start.tolist(),  # (y,x) coordinates
                'head_end_pos': head_end.tolist(),      # (y,x) coordinates
                'body_start_pos': body_start.tolist(),  # (y,x) coordinates
                'body_end_pos': body_end.tolist(),      # (y,x) coordinates
                'head_vector': head_vector.tolist(),
                'body_vector': body_vector.tolist(),
                'head_mag': float(head_mag),
                'body_mag': float(body_mag),
                'head_section': head_section,
                'skeleton_points': norm_points.tolist(),  # All normalized points for reference
                'error': None
            }
    
    # If no valid result found, use previous angle
    if best_result is None:
        return {
            'angle_degrees': prev_angle if prev_angle is not None else 0,
            'error': 'No valid angle found',
            'head_mag': 0,
            'body_mag': 0
        }
    
    return best_result

# Processing loop
head_angle_results = {}
all_angles = []
frame_angles_dict = {}
prev_angles = {}
angle_data = []  # For DataFrame

for frame_idx in sorted(truncated_skeletons.keys()):
    frame_data = truncated_skeletons[frame_idx]
    frame_angles = {}
    frame_angles_dict[frame_idx] = []
    
    for obj_id, skeleton_data in frame_data.items():
        skeleton = skeleton_data[0]
        prev_angle = prev_angles.get(obj_id)
        
        result = calculate_head_angle_with_positions(
            skeleton,
            prev_angle=prev_angle,
            min_vector_length=5
        )
        
        if result['angle_degrees'] is not None:
            prev_angles[obj_id] = result['angle_degrees']
            frame_angles[obj_id] = result
            all_angles.append(result['angle_degrees'])
            frame_angles_dict[frame_idx].append(result['angle_degrees'])
            
            # Add data for DataFrame
            angle_data.append({
                'frame': frame_idx,
                'object_id': obj_id, 
                'angle_degrees': result['angle_degrees']
            })
    
    head_angle_results[frame_idx] = frame_angles

# Create DataFrame with angle data and plot head angles over time
angle_df = pd.DataFrame(angle_data)
#angle_df.to_csv('angle_data.csv', index=False)



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



smoothed_df = smooth_head_angles_with_validation(angle_df, 
            deviation_threshold=12)

# Convert DataFrame columns to numpy arrays before plotting
frame_data = smoothed_df['frame'].to_numpy()
angle_data = smoothed_df['angle_degrees'].to_numpy()

plt.figure(figsize=(12, 6))
plt.plot(frame_data, angle_data, '.-', alpha=0.7)
plt.xlabel('Frame')
plt.ylabel('Head Angle (degrees)')
plt.title('Head Angle Over Time')
plt.ylim(-40, 40)  # Set fixed y-axis limits
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('head_angles_over_time.png')
plt.close()




#Add bend position
import numpy.linalg as la

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
            'error': 'No valid angle found',
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


# Compare angles between old and new functions
print("Comparing angles between old and new functions...")

for frame_idx in sorted(truncated_skeletons.keys()):
    frame_data = truncated_skeletons[frame_idx]
    
    for obj_id, skeleton_data in frame_data.items():
        skeleton = skeleton_data[0]
        
        # Get angles from both functions
        result_with_bend = calculate_head_angle_with_positions_and_bend(
            skeleton,
            prev_angle=None,
            min_vector_length=5
        )
        
        result_without_bend = calculate_head_angle_with_positions(
            skeleton,
            prev_angle=None,
            min_vector_length=5
        )
        
        # Print angles for both functions
        print(f"Frame {frame_idx}, Object {obj_id}:")
        if result_with_bend['error'] is None:
            print(f"  With bend: {result_with_bend['angle_degrees']:.2f}°")
        else:
            print("  With bend: No valid angle found")
            
        if result_without_bend['error'] is None:
            print(f"  Without bend: {result_without_bend['angle_degrees']:.2f}°")
        else:
            print("  Without bend: No valid angle found")
        print()

print("Angle comparison complete.")


# Initialize storage for results
head_angle_results = {}
all_angles = []
all_bends = []
frame_angles_dict = {}
prev_angles = {}
analysis_data = []  # For DataFrame

# Process all frames
for frame_idx in sorted(truncated_skeletons.keys()):
    frame_data = truncated_skeletons[frame_idx]
    frame_angles = {}
    frame_angles_dict[frame_idx] = []
    
    for obj_id, skeleton_data in frame_data.items():
        skeleton = skeleton_data[0]
        prev_angle = prev_angles.get(obj_id)
        
        result = calculate_head_angle_with_positions_and_bend(
            skeleton,
            prev_angle=prev_angle,
            min_vector_length=5,
            restriction_point=0.5,
            straight_threshold=3
        )
        
        if result['angle_degrees'] is not None and result['error'] is None:
            prev_angles[obj_id] = result['angle_degrees']
            frame_angles[obj_id] = result
            all_angles.append(result['angle_degrees'])
            all_bends.append(result['bend_magnitude'])
            frame_angles_dict[frame_idx].append(result['angle_degrees'])
            
            # Add data for DataFrame
            analysis_data.append({
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': result['angle_degrees'],
                'bend_location': result['bend_location'],
                'bend_magnitude': result['bend_magnitude'],
                'bend_position_y': result['bend_position'][0],
                'bend_position_x': result['bend_position'][1],
                'head_mag': result['head_mag'],
                'body_mag': result['body_mag']
            })
    
    head_angle_results[frame_idx] = frame_angles

# Convert to DataFrame
df = pd.DataFrame(analysis_data)

# Get the original bend locations for comparison
original_bends = df['bend_location'].copy()

# Interpolate straight frames
df_interpolated = interpolate_straight_frames(df)

# Verify that only straight frames were modified
modified_mask = df_interpolated['bend_location'] != original_bends
straight_mask = df_interpolated['angle_degrees'].abs() <= 3
print(f"Modified {modified_mask.sum()} frames")
print(f"Found {straight_mask.sum()} straight frames")
if not (modified_mask == straight_mask).all():
    print("Warning: Some non-straight frames were modified or some straight frames were not modified")


# Calculate summary statistics
summary_stats = {
    'mean_angle': np.mean(df_interpolated['angle_degrees']),
    'std_angle': np.std(df_interpolated['angle_degrees']),
    'mean_bend_location': np.mean(df_interpolated['bend_location']),
    'std_bend_location': np.std(df_interpolated['bend_location'])
}

# Group by frame to see progression over time
frame_stats = df_interpolated.groupby('frame').agg({
    'angle_degrees': ['mean', 'std'],
    'bend_location': ['mean', 'std']
}).reset_index()

print("\nOverall Summary:")
for key, value in summary_stats.items():
    print(f"{key}: {value:.2f}")

print("\nFirst few rows of the interpolated dataframe:")
print(df_interpolated.head())



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


frame_idx = list(truncated_skeletons.keys())[156]  
obj_id = list(truncated_skeletons[frame_idx].keys())[0]  

plt.figure(figsize=(10, 10))
plot_skeleton_with_bend(frame_idx, obj_id, truncated_skeletons, head_angle_results)
plt.tight_layout()
plt.show()
plt.savefig('tst.png')
plt.close()





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
            
            if result['angle_degrees'] is not None and result['error'] is None:
                prev_angles[obj_id] = result['angle_degrees']
                
                initial_data.append({
                    'frame': frame_idx,
                    'object_id': obj_id,
                    'angle_degrees': result['angle_degrees'],
                    'head_mag': result['head_mag'],
                    'body_mag': result['body_mag']
                })
    
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

# Usage example:
"""
# Process all skeletons with integrated smoothing and bend recalculation
results_df = process_skeleton_batch(
    truncated_skeletons,
    min_vector_length=5,
    restriction_point=0.5,
    straight_threshold=3,
    smoothing_window=3,
    deviation_threshold=12
)
"""

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
ax1.axhspan(-5, 5, color='gray', alpha=0.2, label='Straight Region')

# Plot head angle on left y-axis
l1, = ax1.plot(frame_data, angle_data, 'b.-', alpha=0.7, label='Head Angle')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Head Angle (degrees)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(-40, 40)  # Set y-axis limits for head angle

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





# Plot bend location over time
# Convert df to numpy array before plotting
frame_data = df_interpolated['frame'].to_numpy()
bend_data = df_interpolated['bend_location'].to_numpy()
angle_data = df_interpolated['angle_degrees'].to_numpy()

plt.figure(figsize=(12, 6))

# Create twin axes sharing x-axis
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot bend location on left y-axis
l1, = ax1.plot(frame_data, bend_data, 'b.-', alpha=0.5, label='Bend Location')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Bend Location (0-1)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Plot angle on right y-axis
l2, = ax2.plot(frame_data, angle_data, 'r.-', alpha=0.5, label='Head Angle')
ax2.set_ylabel('Angle (degrees)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Bend Location and Head Angle Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bend_location_over_time.png')
plt.close()











def create_layered_mask_video(image_dir, bottom_masks_dict, top_masks_dict, output_path, fps=10, 
                          bottom_alpha=0.5, top_alpha=0.7):
    """
    Create a video with two layers of mask overlays from a directory of images and two dictionaries of masks.
    The bottom_masks_dict will be rendered first, with top_masks_dict overlaid on top.
    
    Args:
        image_dir (str): Directory containing the input images
        bottom_masks_dict (dict): Dictionary where keys are frame indices and values are
                                 dictionaries of mask_id: mask pairs for the bottom layer
        top_masks_dict (dict): Dictionary where keys are frame indices and values are
                              dictionaries of mask_id: mask pairs for the top layer
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        bottom_alpha (float): Transparency of the bottom mask overlay (0-1)
        top_alpha (float): Transparency of the top mask overlay (0-1)
    """
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

    # Create color mapping for mask IDs (separate for each layer)
    bottom_mask_ids = set()
    top_mask_ids = set()
    for masks in bottom_masks_dict.values():
        bottom_mask_ids.update(masks.keys())
    for masks in top_masks_dict.values():
        top_mask_ids.update(masks.keys())
    
    # Use first half of colors for bottom masks, second half for top masks
    mid_point = len(COLORS) // 2
    bottom_colors = COLORS[:mid_point]
    top_colors = COLORS[mid_point:] + COLORS[:max(0, len(top_mask_ids) - len(COLORS) // 2)]
    
    bottom_mask_colors = {mask_id: bottom_colors[i % len(bottom_colors)] 
                         for i, mask_id in enumerate(bottom_mask_ids)}
    top_mask_colors = {mask_id: top_colors[i % len(top_colors)] 
                      for i, mask_id in enumerate(top_mask_ids)}

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
            
            # Apply top masks if available
            if frame_idx in top_masks_dict:
                top_overlay = create_mask_overlay(frame,
                                               top_masks_dict[frame_idx],
                                               top_mask_colors,
                                               top_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, top_overlay, top_alpha, 0)

            # Write frame
            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue

    # Clean up
    out.release()
    print(f"Video saved to {output_path}")

# Example usage:
"""
image_dir = video_dir
bottom_masks = head_segments
top_masks = truncated_skeletons
output_path = "head_skeleton_video.mp4"

create_layered_mask_video(image_dir, bottom_masks, top_masks, output_path, 
                         fps=10, bottom_alpha=0.5, top_alpha=0.7)
"""

from PIL import Image

frame = 156
mask = 2
mask = truncated_skeletons[frame][2][0]

image_array = np.uint8(mask * 255)
image = Image.fromarray(image_array)
image.save('tst.png')



def create_layered_mask_vector_video(image_dir, bottom_masks_dict, top_masks_dict, angle_results_dict, 
                                output_path, fps=10, bottom_alpha=0.5, top_alpha=0.7,
                                head_vector_color=(255, 0, 0), body_vector_color=(0, 0, 255),
                                vector_scale=1.0, vector_width=2):
    """
    Create a video with mask overlays and skeleton direction vectors.
    
    Args:
        image_dir (str): Directory containing the input images
        bottom_masks_dict (dict): Dictionary of bottom layer masks
        top_masks_dict (dict): Dictionary of top layer masks
        angle_results_dict (dict): Dictionary of angle calculation results with head/body positions and vectors
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        bottom_alpha (float): Transparency of bottom mask overlay (0-1)
        top_alpha (float): Transparency of top mask overlay (0-1)
        head_vector_color (tuple): RGB color for head vector
        body_vector_color (tuple): RGB color for body vector
        vector_scale (float): Scaling factor for vector lengths
        vector_width (int): Width of vector lines
    """
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

    def draw_vectors(image, angle_info, vector_scale=1.0):
        """Helper function to draw head and body vectors"""
        if angle_info.get('error'):
            return image
            
        # Get vector information
        head_start = angle_info['head_start_pos']  # [y, x] format
        head_vector = angle_info['head_vector']
        body_start = angle_info['body_start_pos']
        body_vector = angle_info['body_vector']
        
        # Scale vectors
        head_vector = [v * vector_scale for v in head_vector]
        body_vector = [v * vector_scale for v in body_vector]
        
        # Draw head vector
        # Convert from [y, x] to [x, y] for cv2
        start_point = (int(head_start[1]), int(head_start[0]))
        end_point = (int(head_start[1] + head_vector[1]), 
                    int(head_start[0] + head_vector[0]))
        cv2.arrowedLine(image, start_point, end_point, 
                       head_vector_color, vector_width, tipLength=0.3)
        
        # Draw body vector
        start_point = (int(body_start[1]), int(body_start[0]))
        end_point = (int(body_start[1] + body_vector[1]),
                    int(body_start[0] + body_vector[0]))
        cv2.arrowedLine(image, start_point, end_point,
                       body_vector_color, vector_width, tipLength=0.3)
        
        # Add angle text near head vector tip
        # Calculate text position near the head vector tip
        text_offset = 40  # Offset from the vector tip
        text_x = int(head_start[1] + head_vector[1] + text_offset)
        text_y = int(head_start[0] + head_vector[0])
        
        # Format angle text
        angle_text = f"{angle_info['angle_degrees']:.1f} deg"
        
        # Add background to make text more readable
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            angle_text, font, font_scale, font_thickness)
        
        # Draw background rectangle
        padding = 5
        cv2.rectangle(image, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Draw text
        cv2.putText(image, angle_text,
                   (text_x, text_y),
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
            
            # Apply top masks if available
            if frame_idx in top_masks_dict:
                top_overlay = create_mask_overlay(frame,
                                               top_masks_dict[frame_idx],
                                               top_mask_colors,
                                               top_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, top_overlay, top_alpha, 0)
            
            # Draw vectors if angle results are available for this frame
            if frame_idx in angle_results_dict:
                for obj_id, angle_info in angle_results_dict[frame_idx].items():
                    final_frame = draw_vectors(final_frame, angle_info, vector_scale)

            # Write frame
            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue

    # Clean up
    out.release()
    print(f"Video saved to {output_path}")

# Example usage:
"""
video_dir = "/home/lilly/phd/ria/data_foranalysis/AG_WT/videotojpg/AG_WT-MMH99_10s_20190221_04"
image_dir = video_dir
bottom_masks = head_segments
top_masks = truncated_skeletons
angle_results = head_angle_results  # Results with new format
output_path = "head_skeleton_vectors_video.mp4"

create_layered_mask_vector_video(
    image_dir=image_dir,
    bottom_masks_dict=bottom_masks,
    top_masks_dict=top_masks,
    angle_results_dict=angle_results,
    output_path=output_path,
    fps=10,
    bottom_alpha=0.5,
    top_alpha=0.7,
    vector_scale=1.0  # Adjust this value to make vectors longer or shorter
)
"""






####Compare segmentations
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

#Get mean and std of brightness values from masked regions using average mask
def extract_brightness_values(aligned_images: List[np.ndarray],
                            masks_dict: Dict[int, Dict[int, np.ndarray]], 
                            object_id: int) -> pd.DataFrame:
    """
    Extracts mean and standard deviation of brightness values from masked regions.
    
    Parameters:
        aligned_images: List of grayscale images (numpy arrays)
        masks_dict: Dictionary where keys are frame indices and values are
                   dictionaries of mask_id: mask pairs for that frame
        object_id: ID of the object whose mask should be used
    
    Returns:
        DataFrame containing frame number, mean and std brightness values for each frame
    """
    data = []
    
    for frame_idx, image in enumerate(aligned_images):
        if frame_idx in masks_dict and object_id in masks_dict[frame_idx]:
            # Get mask for this frame and object
            mask = masks_dict[frame_idx][object_id][0]
            
            # Extract pixels within mask
            masked_pixels = image[mask]
            
            # Calculate statistics
            mean_val = np.mean(masked_pixels)
            std_val = np.std(masked_pixels)
            
            data.append({
                'frame': frame_idx,
                'mean_brightness': mean_val,
                'std_brightness': std_val
            })
    
    return pd.DataFrame(data)


def extract_top_percent_brightness(aligned_images: List[np.ndarray],
                                 masks_dict: Dict[int, Dict[int, np.ndarray]], 
                                 object_id: int,
                                 percent: float) -> pd.DataFrame:
    """
    Extracts mean brightness of the top X% brightest pixels from masked regions.
    
    Parameters:
        aligned_images: List of grayscale images (numpy arrays)
        masks_dict: Dictionary where keys are frame indices and values are
                   dictionaries of mask_id: mask pairs for that frame
        object_id: ID of the object whose mask should be used
        percent: Percentage of brightest pixels to consider (0-100)
        
    Returns:
        DataFrame containing frame number and mean brightness of top X% pixels for each frame
    """
    # Validate percentage input
    if not 0 < percent <= 100:
        raise ValueError("Percentage must be between 0 and 100")
        
    data = []
    
    for frame_idx, image in enumerate(aligned_images):
        if frame_idx in masks_dict and object_id in masks_dict[frame_idx]:
            # Get mask for this frame and object
            mask = masks_dict[frame_idx][object_id][0]
            
            # Extract pixels within mask
            masked_pixels = image[mask]
            
            # Calculate number of pixels to use based on percentage
            n_pixels = int(round(len(masked_pixels) * (percent / 100)))
            # Ensure at least 1 pixel is used
            n_pixels = max(1, n_pixels)
            
            # Get top percentage of brightest pixels
            top_n_pixels = np.sort(masked_pixels)[-n_pixels:]
            
            # Calculate mean of top pixels
            mean_top_percent = np.mean(top_n_pixels)
            
            data.append({
                'frame': frame_idx,
                'mean_top_percent_brightness': mean_top_percent,
                'n_pixels_used': n_pixels,
                'total_pixels': len(masked_pixels),
                'percent_used': percent
            })
    
    return pd.DataFrame(data)


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

output_filename = "/home/lilly/phd/ria/data_analyzed/aligned_segments/AG-MMH99_10s_20190306_02_crop_riasegmentation_alignedsegments.h5"
loaded_segments = load_cleaned_segments_from_h5(output_filename)
img_dir = "/home/lilly/phd/ria/data_foranalysis/riacrop/AG-MMH99_10s_20190306_02_crop"

# Get list of image files sorted by frame number
image_files = sorted(os.listdir(img_dir))

# Load and preprocess images
aligned_images = []
for img_file in image_files:
    if img_file.endswith('.jpg'):
        img_path = os.path.join(img_dir, img_file)
        # Read as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            aligned_images.append(img)

final_masks = load_cleaned_segments_from_h5(output_filename)
brightness_loop = extract_brightness_values(aligned_images, final_masks, 4)
brightness_nrd = extract_brightness_values(aligned_images, final_masks, 2)
brightness_nrv = extract_brightness_values(aligned_images, final_masks, 3)

loop_top_10percent = extract_top_percent_brightness(aligned_images, final_masks, 4, percent=10)
nrd_top_10percent = extract_top_percent_brightness(aligned_images, final_masks, 2, percent=10)
nrv_top_10percent = extract_top_percent_brightness(aligned_images, final_masks, 3, percent=10)

loop_top_50percent = extract_top_percent_brightness(aligned_images, final_masks, 4, percent=50)
nrd_top_50percent = extract_top_percent_brightness(aligned_images, final_masks, 2, percent=50)
nrv_top_50percent = extract_top_percent_brightness(aligned_images, final_masks, 3, percent=50)

loop_top_25percent = extract_top_percent_brightness(aligned_images, final_masks, 4, percent=25)
nrd_top_25percent = extract_top_percent_brightness(aligned_images, final_masks, 2, percent=25)
nrv_top_25percent = extract_top_percent_brightness(aligned_images, final_masks, 3, percent=25)

# Load all Fiji data files
fiji_dir = "/home/lilly/phd/ria/MMH99_10s_20190306_02"
fiji_files = [f for f in os.listdir(fiji_dir) if f.endswith('.xlsx')]

all_fiji_data = []
for file in fiji_files:
    df = pd.read_excel(os.path.join(fiji_dir, file))
    all_fiji_data.append(df)

# Combine all Fiji data
fiji_combined = pd.concat(all_fiji_data)

# Group by frame and calculate mean and std for each segment
fiji_stats = fiji_combined.groupby('Frame').agg({
    'loop': ['mean', 'std'],
    'nrD': ['mean', 'std'],
    'nrV': ['mean', 'std']
}).reset_index()

# Normalize means and calculate normalized std and 95% confidence intervals
# For each segment, calculate std relative to the range of the mean
for segment in ['loop', 'nrD', 'nrV']:
    mean_range = fiji_stats[(segment, 'mean')].max() - fiji_stats[(segment, 'mean')].min()
    fiji_stats[(segment, 'std_norm')] = fiji_stats[(segment, 'std')] / mean_range
    
    # Calculate 95% confidence interval (1.96 * std error)
    n = len(fiji_combined[fiji_combined['Frame'] == fiji_stats['Frame'][0]][segment])  # samples per frame
    fiji_stats[(segment, 'ci_norm')] = (1.96 * fiji_stats[(segment, 'std')] / np.sqrt(n)) / mean_range
    fiji_stats[(segment, 'mean_norm')] = normalize(fiji_stats[(segment, 'mean')])


# Create plot with ribbons
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(40, 32))

# Define colors
loop_color = 'blue'
nrv_color = 'red'
nrd_color = 'green'
fiji_color = 'purple'
alpha_ribbon = 0.3

# Create a twin axis for angles and bend location on each subplot
ax1_angle = ax1.twinx()
ax1_bend = ax1.twinx()
ax1_bend.spines['right'].set_position(('outward', 60))

ax2_angle = ax2.twinx()
ax2_bend = ax2.twinx()
ax2_bend.spines['right'].set_position(('outward', 60))

ax3_angle = ax3.twinx()
ax3_bend = ax3.twinx()
ax3_bend.spines['right'].set_position(('outward', 60))

ax4_angle = ax4.twinx()
ax4_bend = ax4.twinx()
ax4_bend.spines['right'].set_position(('outward', 60))

# Add light blue background regions to all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.axvspan(0, 100, color='lightblue', alpha=0.2)
    ax.axvspan(200, 300, color='lightblue', alpha=0.2)
    ax.axvspan(400, 500, color='lightblue', alpha=0.2)
    ax.axvspan(600, 613, color='lightblue', alpha=0.2)

# Top subplot - loop
ax1.plot(brightness_loop['frame'].to_numpy(), normalize(brightness_loop['mean_brightness']).to_numpy(),
         color=loop_color, linewidth=2, label='Aligned Loop')

ax1.plot(fiji_stats['Frame'].to_numpy(), fiji_stats[('loop', 'mean_norm')].to_numpy(),
         color=fiji_color, linewidth=2, label='Fiji Loop Mean', alpha=0.3)
ax1.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_stats[('loop', 'mean_norm')].to_numpy() - fiji_stats[('loop', 'std_norm')].to_numpy(),
                 fiji_stats[('loop', 'mean_norm')].to_numpy() + fiji_stats[('loop', 'std_norm')].to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

# Add angle data and bend location to first subplot
ax1_angle.plot(np.array(angle_df['frame']), np.array(angle_df['angle_degrees']), 'k--', alpha=0.95, label='Head Angle')
ax1_angle.set_ylabel('Head Angle (degrees)', color='k')
# Convert df_interpolated to numpy arrays before plotting
bend_data = df_interpolated['bend_location'].to_numpy()
frame_data = df_interpolated['frame'].to_numpy()
ax1_bend.plot(frame_data, bend_data, 'g:', alpha=0.95, label='Bend Location')
ax1_bend.set_ylabel('Bend Location', color='g')

ax1.set_xlabel('Frame Number', fontsize=14)
ax1.set_ylabel('Normalized Brightness', fontsize=14)
ax1.set_title('Normalized Brightness Over Frames (Loop)', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks(np.arange(0, len(brightness_loop), 10))
ax1.tick_params(axis='both', labelsize=12)
ax1.legend(fontsize=12)
ax1.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Middle subplot - nrV
ax2.plot(brightness_nrv['frame'].to_numpy(), normalize(brightness_nrv['mean_brightness']).to_numpy(),
         color=nrv_color, linewidth=2, label='Aligned nrV')

ax2.plot(fiji_stats['Frame'].to_numpy(), fiji_stats[('nrV', 'mean_norm')].to_numpy(),
         color=fiji_color, linewidth=2, label='Fiji nrV Mean', alpha=0.3)
ax2.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_stats[('nrV', 'mean_norm')].to_numpy() - fiji_stats[('nrV', 'std_norm')].to_numpy(),
                 fiji_stats[('nrV', 'mean_norm')].to_numpy() + fiji_stats[('nrV', 'std_norm')].to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

# Add angle data and bend location to second subplot
ax2_angle.plot(np.array(angle_df['frame']), np.array(angle_df['angle_degrees']), 'k--', alpha=0.95, label='Head Angle')
ax2_angle.set_ylabel('Head Angle (degrees)', color='k')
ax2_bend.plot(frame_data, bend_data, 'g:', alpha=0.95, label='Bend Location')
ax2_bend.set_ylabel('Bend Location', color='g')

ax2.set_xlabel('Frame Number', fontsize=14)
ax2.set_ylabel('Normalized Brightness', fontsize=14)
ax2.set_title('Normalized Brightness Over Frames (nrV)', fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks(np.arange(0, len(brightness_nrv), 10))
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(fontsize=12)
ax2.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Bottom subplot - nrD
ax3.plot(brightness_nrd['frame'].to_numpy(), normalize(brightness_nrd['mean_brightness']).to_numpy(),
         color=nrd_color, linewidth=2, label='Aligned nrD')

ax3.plot(fiji_stats['Frame'].to_numpy(), fiji_stats[('nrD', 'mean_norm')].to_numpy(),
         color=fiji_color, linewidth=2, label='Fiji nrD Mean', alpha=0.3)
ax3.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_stats[('nrD', 'mean_norm')].to_numpy() - fiji_stats[('nrD', 'std_norm')].to_numpy(),
                 fiji_stats[('nrD', 'mean_norm')].to_numpy() + fiji_stats[('nrD', 'std_norm')].to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

# Add angle data and bend location to third subplot
ax3_angle.plot(np.array(angle_df['frame']), np.array(angle_df['angle_degrees']), 'k--', alpha=0.95, label='Head Angle')
ax3_angle.set_ylabel('Head Angle (degrees)', color='k')
ax3_bend.plot(frame_data, bend_data, 'g:', alpha=0.95, label='Bend Location')
ax3_bend.set_ylabel('Bend Location', color='g')

ax3.set_xlabel('Frame Number', fontsize=14)
ax3.set_ylabel('Normalized Brightness', fontsize=14)
ax3.set_title('Normalized Brightness Over Frames (nrD)', fontsize=18)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_xticks(np.arange(0, len(brightness_nrd), 10))
ax3.tick_params(axis='both', labelsize=12)
ax3.legend(fontsize=12)
ax3.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax3.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax3.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Fourth subplot - sum of all segments
sum_aligned = brightness_loop['mean_brightness'].to_numpy() + \
             brightness_nrv['mean_brightness'].to_numpy() + \
             brightness_nrd['mean_brightness'].to_numpy()
sum_aligned_normalized = normalize(sum_aligned)

# For Fiji data, sum the means first, then normalize
fiji_sum_mean = fiji_stats[('loop', 'mean')] + fiji_stats[('nrV', 'mean')] + fiji_stats[('nrD', 'mean')]
# Calculate normalized std for sum using error propagation
fiji_sum_std = np.sqrt(
    (fiji_stats[('loop', 'std_norm')])**2 + 
    (fiji_stats[('nrV', 'std_norm')])**2 + 
    (fiji_stats[('nrD', 'std_norm')])**2
)
fiji_sum_normalized = normalize(fiji_sum_mean)

ax4.plot(brightness_loop['frame'].to_numpy(), sum_aligned_normalized,
         color='black', linewidth=2, label='Sum of Aligned Segments')
ax4.plot(fiji_stats['Frame'].to_numpy(), fiji_sum_normalized.to_numpy(),
         color=fiji_color, linewidth=2, label='Sum of Fiji Segments Mean', alpha=0.3)
ax4.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_sum_normalized.to_numpy() - fiji_sum_std.to_numpy(),
                 fiji_sum_normalized.to_numpy() + fiji_sum_std.to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

# Calculate and plot nrD minus nrV brightness
nrd_minus_nrv = brightness_nrd['mean_brightness'].to_numpy() - brightness_nrv['mean_brightness'].to_numpy()
nrd_minus_nrv_normalized = normalize(nrd_minus_nrv)
ax4.plot(brightness_loop['frame'].to_numpy(), nrd_minus_nrv_normalized,
         color='orange', linewidth=2, label='nrD - nrV')

# Add angle data and bend location to fourth subplot
ax4_angle.plot(np.array(angle_df['frame']), np.array(angle_df['angle_degrees']), 'k--', alpha=0.95, label='Head Angle')
ax4_angle.set_ylabel('Head Angle (degrees)', color='k')
ax4_bend.plot(frame_data, bend_data, 'g:', alpha=0.95, label='Bend Location')
ax4_bend.set_ylabel('Bend Location', color='g')

ax4.set_xlabel('Frame Number', fontsize=14)
ax4.set_ylabel('Normalized Sum of Brightness', fontsize=14)
ax4.set_title('Normalized Sum of All Segments Over Frames', fontsize=18)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.set_xticks(np.arange(0, len(brightness_loop), 10))
ax4.tick_params(axis='both', labelsize=12)
ax4.legend(fontsize=12)
ax4.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax4.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax4.axvline(x=500, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('normalized_brightness_over_frames_ria9902_withheadangle.png', dpi=300, bbox_inches='tight')
plt.close()