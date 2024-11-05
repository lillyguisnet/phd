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

video_dir = '/home/lilly/phd/ria/data_foranalysis/videotojpg/AG-MMH99_10s_20190306_02'
output_filename = '/home/lilly/phd/ria/data_analyzed/head_segmentation/AG-MMH99_10s_20190306_02_headsegmentation.h5'

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

head_segments = load_cleaned_segments_from_h5(output_filename)


# region [test skeletonization]
def get_mask_boundary_points(mask):
    """
    Get all boundary points of the mask.
    """
    boundary = morphology.binary_dilation(mask) ^ mask
    boundary_points = np.where(boundary)
    points = np.column_stack((boundary_points[0], boundary_points[1]))
    return points[~np.any(np.isnan(points), axis=1)]

def get_farthest_boundary_points(mask):
    """
    Find the two most distant boundary points of the mask.
    """
    points = get_mask_boundary_points(mask)
    if len(points) < 2:
        return []
    
    # Calculate pairwise distances
    distances = cdist(points, points)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    
    # Convert to integer coordinates
    point1 = points[i].astype(int)
    point2 = points[j].astype(int)
    
    # Verify points are within mask bounds
    shape = mask.shape
    if (0 <= point1[0] < shape[0] and 0 <= point1[1] < shape[1] and
        0 <= point2[0] < shape[0] and 0 <= point2[1] < shape[1]):
        return [point1, point2]
    return []

def get_endpoints_from_skeleton(skel):
    """
    Get endpoints from a skeleton by finding points with exactly one neighbor.
    """
    neighbors = np.zeros_like(skel, dtype=int)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            neighbors += np.roll(np.roll(skel, i, axis=0), j, axis=1)
    
    endpoints = np.where((skel > 0) & (neighbors == 1))
    points = np.column_stack((endpoints[0], endpoints[1]))
    return points[~np.any(np.isnan(points), axis=1)]

def connect_points_in_mask(mask, start, end):
    """
    Connect two points within a mask, ensuring the connection stays within the mask.
    """
    # Ensure points are within bounds
    shape = np.array(mask.shape)
    start = np.clip(start, 0, shape - 1)
    end = np.clip(end, 0, shape - 1)
    
    # Draw initial line
    rr, cc = draw.line(int(start[0]), int(start[1]), 
                      int(end[0]), int(end[1]))
    
    # Keep only points within mask
    valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    rr = rr[valid]
    cc = cc[valid]
    
    # Return only points that lie within the mask
    mask_points = mask[rr, cc]
    return rr[mask_points], cc[mask_points]

def extend_skeleton_to_point(skeleton, mask, start, end):
    """
    Extend skeleton from start point to end point.
    """
    start = np.array(start).astype(int)
    end = np.array(end).astype(int)
    
    # Try path finding first
    try:
        costs = np.where(mask, 1, 1000)
        path = graph.route_through_array(
            costs,
            start=tuple(start),
            end=tuple(end),
            fully_connected=True
        )[0]
        if path is not None:
            path = np.array(path)
            skeleton[path[:, 0], path[:, 1]] = True
            return skeleton
    except Exception:
        pass
    
    # If path finding fails, try direct connection
    try:
        rr, cc = connect_points_in_mask(mask, start, end)
        skeleton[rr, cc] = True
    except Exception:
        pass
    
    return skeleton

def enhanced_skeleton_v4(mask):
    """
    Generate an enhanced skeleton that extends to the mask endpoints.
    """
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    
    # Generate initial skeleton
    skeleton = morphology.skeletonize(mask)
    
    # Find boundary points
    boundary_points = get_farthest_boundary_points(mask)
    if not boundary_points:
        return skeleton
    
    # Get skeleton endpoints or points
    skeleton_points = get_endpoints_from_skeleton(skeleton)
    if len(skeleton_points) == 0:
        coords = np.where(skeleton)
        skeleton_points = np.column_stack((coords[0], coords[1]))
        if len(skeleton_points) == 0:
            return skeleton
    
    # Extend skeleton
    extended = skeleton.copy()
    for target in boundary_points:
        target_reshaped = target.reshape(1, 2)
        distances = cdist(target_reshaped, skeleton_points)
        start = skeleton_points[np.argmin(distances)]
        extended = extend_skeleton_to_point(extended, mask, start, target)
    
    # Final cleanup
    final = morphology.thin(extended)
    return final

def process_all_frames_enhanced_v4(head_segments):
    """
    Process all frames with enhanced skeletonization.
    """
    enhanced_skeletons = {}
    
    for frame_idx, frame_data in head_segments.items():
        frame_skeletons = {}
        print(f"Processing frame {frame_idx}")
        
        for obj_id, mask in frame_data.items():
            try:
                skeleton = enhanced_skeleton_v4(mask)
                frame_skeletons[obj_id] = skeleton
            except Exception as e:
                print(f"Error processing frame {frame_idx}, object {obj_id}: {str(e)}")
                # Create a 2D mask by squeezing first
                fallback_mask = np.squeeze(mask)
                frame_skeletons[obj_id] = morphology.skeletonize(fallback_mask)
        
        enhanced_skeletons[frame_idx] = frame_skeletons
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}")
    
    return enhanced_skeletons


enhanced_skeletons = process_all_frames_enhanced_v4(head_segments)



def find_tip_points(mask):
    """
    Find the two most distant points along the mask boundary.
    """
    # Get boundary points
    boundary = morphology.binary_dilation(mask) ^ mask
    boundary_points = np.array(np.where(boundary)).T
    
    if len(boundary_points) < 2:
        return []
    
    # Calculate all pairwise distances between boundary points
    dist_matrix = cdist(boundary_points, boundary_points)
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    return [boundary_points[i], boundary_points[j]]

def get_single_path_skeleton(mask):
    """
    Generate a single continuous path skeleton from one end to the other.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    
    # Find the two most distant points
    endpoints = find_tip_points(mask)
    if len(endpoints) < 2:
        return morphology.skeletonize(mask)
    
    # Create cost matrix using distance transform
    dt = distance_transform_edt(mask)
    max_dist = np.max(dt)
    if max_dist > 0:
        # Create costs that prefer the medial axis
        costs = 1 + (1 - dt/max_dist) * 2
    else:
        costs = np.ones_like(mask, dtype=float)
    
    costs[~mask] = np.inf
    
    # Find the optimal path between endpoints
    try:
        path = graph.route_through_array(
            costs,
            start=tuple(endpoints[0]),
            end=tuple(endpoints[1]),
            fully_connected=True
        )[0]
        
        # Convert path to skeleton
        skeleton = np.zeros_like(mask, dtype=bool)
        path = np.array(path)
        skeleton[path[:, 0], path[:, 1]] = True
        
        # Optional: slight smoothing while maintaining endpoints
        skeleton = morphology.thin(morphology.binary_dilation(skeleton, morphology.disk(1)))
        
        return skeleton
        
    except Exception as e:
        print(f"Path finding failed: {str(e)}")
        return morphology.skeletonize(mask)

def process_frames_continuous(head_segments):
    """
    Process all frames with continuous skeleton generation.
    """
    continuous_skeletons = {}
    
    for frame_idx, frame_data in head_segments.items():
        frame_skeletons = {}
        print(f"Processing frame {frame_idx}")
        
        for obj_id, mask in frame_data.items():
            try:
                mask = np.squeeze(mask)
                skeleton = get_single_path_skeleton(mask)
                frame_skeletons[obj_id] = skeleton
            except Exception as e:
                print(f"Error processing frame {frame_idx}, object {obj_id}: {str(e)}")
                fallback_mask = np.squeeze(mask)
                frame_skeletons[obj_id] = morphology.skeletonize(fallback_mask)
        
        continuous_skeletons[frame_idx] = frame_skeletons
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}")
    
    return continuous_skeletons



enhanced_skeletons = process_frames_continuous(head_segments)
# endregion [test skeletonization]

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
            cutoff_point = y_min + keep_pixels
            
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


def gaussian_weighted_curvature(points, window_size, sigma):
    """
    Calculate curvature using Gaussian-weighted windows.
    """
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    pad_width = window_size // 2
    padded_points = np.pad(points, ((pad_width, pad_width), (0, 0)), mode='edge')
    
    weights = ndimage.gaussian_filter1d(np.ones(window_size), sigma)
    weights /= np.sum(weights)
    
    curvatures = []
    directions = []  # Store direction vectors
    
    for i in range(len(points)):
        window = padded_points[i:i+window_size]
        centroid = np.sum(window * weights[:, np.newaxis], axis=0) / np.sum(weights)
        centered = window - centroid
        cov = np.dot(centered.T, centered * weights[:, np.newaxis]) / np.sum(weights)
        eigvals, eigvecs = np.linalg.eig(cov)
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[:, sort_indices]
        
        curvature = eigvals[1] / (eigvals[0] + eigvals[1])
        
        # Calculate direction vector (perpendicular to major axis)
        direction = eigvecs[:, 1]  # Second eigenvector
        
        # Ensure consistent direction relative to the body
        if i > 0 and np.dot(directions[-1], direction) < 0:
            direction = -direction
            
        directions.append(direction)
        curvatures.append(curvature)
    
    # Convert to numpy arrays
    curvatures = np.array(curvatures)
    directions = np.array(directions)
    
    # Determine sign of curvature based on direction change
    for i in range(1, len(directions)-1):
        # Calculate cross product to determine turn direction
        prev_dir = directions[i-1]
        curr_dir = directions[i]
        cross_z = np.cross([prev_dir[0], prev_dir[1], 0], [curr_dir[0], curr_dir[1], 0])[2]
        curvatures[i] *= np.sign(cross_z)
    
    return curvatures

def order_segments(segments):
    """
    Order skeleton segments from one endpoint to the other.
    """
    segments_set = set(map(tuple, segments))
    
    graph = defaultdict(list)
    for x, y in segments_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in segments_set:
                    graph[(x, y)].append(neighbor)
    
    endpoints = [point for point, neighbors in graph.items() if len(neighbors) == 1]
    
    if len(endpoints) != 2:
        raise ValueError("Expected exactly two endpoints")
    
    # Always start from the topmost point (head)
    start = min(endpoints, key=lambda x: x[0])
    end = max(endpoints, key=lambda x: x[0]) if endpoints[0] == start else endpoints[0]
    
    ordered = [start]
    current = start
    
    while current != end:
        next_point = [p for p in graph[current] if p not in ordered][0]
        ordered.append(next_point)
        current = next_point
    
    return np.array(ordered)

def calculate_worm_curvature(skeleton, window_size=50, sigma=10, n_points=200):
    """
    Calculate curvature along the worm's body.
    """
    try:
        # Get skeleton points and order them
        points = np.column_stack(np.where(skeleton))
        ordered_points = order_segments(points)
        
        # Interpolate smooth curve
        x, y = ordered_points[:, 1], ordered_points[:, 0]  # Swap x,y for image coordinates
        tck, u = interpolate.splprep([x, y], s=0)
        unew = np.linspace(0, 1, num=n_points)
        smooth_points = np.column_stack(interpolate.splev(unew, tck))
        
        # Calculate curvature
        curvature = gaussian_weighted_curvature(smooth_points, window_size, sigma)
        
        # Find maximum curvature and its position
        max_idx = np.argmax(np.abs(curvature))
        max_curvature = curvature[max_idx]
        max_position = max_idx / len(curvature)
        
        # Determine bend direction based on sign of curvature
        # Negative curvature means bending right (when viewed from head to tail)
        # Positive curvature means bending left
        bend_direction = 'left' if max_curvature > 0 else 'right'
        
        results = {
            'curvature_profile': curvature.tolist(),
            'max_curvature': float(np.abs(max_curvature)),
            'signed_curvature': float(max_curvature),
            'max_position': float(max_position),
            'bend_direction': bend_direction,
            'mean_curvature': float(np.mean(np.abs(curvature))),
            'std_curvature': float(np.std(curvature)),
            'smooth_points': smooth_points.tolist()
        }
        
        return results
        
    except Exception as e:
        print(f"Error in curvature calculation: {str(e)}")
        return None


# Calculate curvature for all frames and objects
curvature_results = {}
max_curvatures = []
for frame_idx, frame_data in truncated_skeletons.items():
    frame_results = {}
    for obj_id, skeleton in frame_data.items():
        # Get the 2D array from the 3D input (taking the first channel)
        skeleton_2d = skeleton[0]
        
        # Calculate curvature with custom parameters
        curvature_info = calculate_worm_curvature(
            skeleton_2d,
            window_size=150,
            sigma=24,
            n_points=300
        )
        
        if curvature_info:
            frame_results[obj_id] = curvature_info
            max_curvatures.append(curvature_info['max_curvature'])
            
    curvature_results[frame_idx] = frame_results

print(f"Range: {np.min(max_curvatures):.3f} to {np.max(max_curvatures):.3f}")
print(f"Mean: {np.mean(max_curvatures):.3f}")


frame_idx = 285
obj_id = 2
skeleton = truncated_skeletons[frame_idx][obj_id][0]

image_array = np.uint8(skeleton * 255)
image = Image.fromarray(image_array)
image.save('tst.png')

# Calculate curvature with custom parameters
curvature_info = calculate_worm_curvature(
    skeleton,
    window_size=50,
    sigma=10,
    n_points=100
)

if curvature_info:
    print(f"Max curvature: {curvature_info['max_curvature']:.3f}")
    print(f"Bend direction: {curvature_info['bend_direction']}")
    print(f"Position: {curvature_info['max_position']:.2f} along body (0=head, 1=tail)")



def curvature_to_angle(curvature_info, window_size=20):
    """
    Convert curvature measurements to angles in degrees.
    
    Args:
        curvature_info: Dictionary containing curvature data
        window_size: Size of window for angle calculation (default=20)
        
    Returns:
        dict: Angle information including:
            - angles: Array of angles along the body in degrees
            - max_angle: Maximum bend angle in degrees
            - max_angle_position: Position of maximum angle (0-1)
            - mean_angle: Mean absolute angle in degrees
    """
    # Convert smooth_points back to numpy array
    points = np.array(curvature_info['smooth_points'])
    
    # Calculate angles using a sliding window
    angles = []
    positions = []
    
    for i in range(0, len(points) - window_size * 2, window_size // 2):
        # Get two adjacent segments
        seg1 = points[i+window_size] - points[i]
        seg2 = points[i+window_size*2] - points[i+window_size]
        
        # Calculate angle between segments
        cos_angle = np.dot(seg1, seg2) / (np.linalg.norm(seg1) * np.linalg.norm(seg2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cos_angle)
        
        # Determine sign based on cross product
        cross_z = np.cross([seg1[0], seg1[1], 0], [seg2[0], seg2[1], 0])[2]
        angle = np.degrees(angle) if cross_z > 0 else -np.degrees(angle)
        
        angles.append(angle)
        positions.append((i + window_size) / len(points))
    
    angles = np.array(angles)
    positions = np.array(positions)
    
    # Find maximum angle and its position
    max_idx = np.argmax(np.abs(angles))
    max_angle = angles[max_idx]
    max_angle_position = positions[max_idx]
    
    # Use the same position as the curvature calculation
    curvature_max_pos = curvature_info['max_position']
    # Find the angle at the position of maximum curvature
    pos_idx = np.argmin(np.abs(positions - curvature_max_pos))
    angle_at_max_curvature = angles[pos_idx]
    
    return {
        'angles': angles.tolist(),
        'max_angle': float(max_angle),
        'max_angle_position': float(max_angle_position),
        'angle_at_max_curvature': float(angle_at_max_curvature),
        'mean_angle': float(np.mean(np.abs(angles))),
        'angle_positions': positions.tolist()
    }



# Calculate angles for all frames and objects
angle_results = {}
max_positions = []
max_angles = []
for frame_idx, frame_data in curvature_results.items():
    frame_angles = {}
    for obj_id, curvature_info in frame_data.items():
        angle_info = curvature_to_angle(curvature_info, window_size=50)
        frame_angles[obj_id] = angle_info
        max_positions.append(curvature_info['max_position'])
        max_angles.append(angle_info['angle_at_max_curvature'])
    angle_results[frame_idx] = frame_angles

max_positions = np.array(max_positions)
max_angles = np.array(max_angles)
q1_pos, q3_pos = np.percentile(max_positions, [25, 75])
q1_ang, q3_ang = np.percentile(max_angles, [25, 75])

print(f"Max position statistics:")
print(f"Range: {np.min(max_positions):.3f} to {np.max(max_positions):.3f}")
print(f"Mean: {np.mean(max_positions):.3f}")
print(f"Median: {np.median(max_positions):.3f}")
print(f"Std dev: {np.std(max_positions):.3f}")
print(f"Q1: {q1_pos:.3f}")
print(f"Q3: {q3_pos:.3f}")

print(f"\nMax angle statistics:")
print(f"Range: {np.min(max_angles):.3f} to {np.max(max_angles):.3f}")
print(f"Mean: {np.mean(max_angles):.3f}")
print(f"Median: {np.median(max_angles):.3f}")
print(f"Std dev: {np.std(max_angles):.3f}")
print(f"Q1: {q1_ang:.3f}")
print(f"Q3: {q3_ang:.3f}")


curvature_info = calculate_worm_curvature(skeleton)

if curvature_info:
    angle_info = curvature_to_angle(curvature_info)
    print(f"Max curvature: {curvature_info['max_curvature']:.3f}")
    print(f"Position: {curvature_info['max_position']:.2f} along body")
    print(f"Angle at max curvature: {angle_info['angle_at_max_curvature']:.1f}°")


#Find angle with both tip of skeleton
def calculate_head_angle_v2(skeleton, head_segment_length=20, body_segment_length=40):
    """
    Calculate the angle between the head tip and body of the worm.
    Uses the bottom portion of the skeleton for body direction.
    
    Parameters:
    -----------
    skeleton : ndarray
        Binary skeleton image
    head_segment_length : int
        Number of points to use for determining head direction
    body_segment_length : int
        Number of points to use for determining body direction
    
    Returns:
    --------
    angle : float
        Angle in degrees between head and body directions
    """
    # Get ordered points along the skeleton
    points = np.column_stack(np.where(skeleton))
    
    # Order points from head to tail (assuming head is at the top)
    ordered_points = points[np.argsort(points[:, 0])]
    
    # Interpolate to get smooth curve
    x, y = ordered_points[:, 1], ordered_points[:, 0]  # Swap x,y for image coordinates
    tck, u = splprep([x, y], s=0)
    u_new = np.linspace(0, 1, num=500)
    smooth_x, smooth_y = splev(u_new, tck)
    smooth_points = np.column_stack((smooth_y, smooth_x))  # Back to image coordinates
    
    # Get head direction vector (using first few points)
    head_vector = smooth_points[head_segment_length] - smooth_points[0]
    
    # Get body direction vector from the bottom portion of the skeleton
    # Use the last body_segment_length points
    body_vector = smooth_points[-1] - smooth_points[-body_segment_length]
    
    # Calculate angle between vectors
    dot_product = np.dot(head_vector, body_vector)
    norms = np.linalg.norm(head_vector) * np.linalg.norm(body_vector)
    
    cos_angle = dot_product / norms
    # Clip to handle floating point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Determine if the head is bent left or right using cross product
    cross_product = np.cross(body_vector, head_vector)
    if cross_product < 0:
        angle_deg = -angle_deg
    
    return {
        'angle_degrees': float(angle_deg),
        'head_vector': head_vector.tolist(),
        'body_vector': body_vector.tolist(),
        'direction': 'left' if cross_product > 0 else 'right',
        'head_point': smooth_points[0].tolist(),
        'body_start': smooth_points[-body_segment_length].tolist(),
        'body_end': smooth_points[-1].tolist(),
        'skeleton': skeleton.tolist()
    }

# Calculate angles for all frames and objects
head_angle_results = {}
all_angles = []
frame_angles_dict = {}  # Store angles per frame for plotting

for frame_idx, frame_data in truncated_skeletons.items():
    frame_angles = {}
    frame_angles_dict[frame_idx] = []  # Initialize list for this frame's angles
    for obj_id, skeleton_data in frame_data.items():
        skeleton = skeleton_data[0]  # Get the skeleton array
        angle_info = calculate_head_angle_v2(skeleton, head_segment_length=15, body_segment_length=200)
        frame_angles[obj_id] = angle_info
        all_angles.append(angle_info['angle_degrees'])
        frame_angles_dict[frame_idx].append(angle_info['angle_degrees'])
    head_angle_results[frame_idx] = frame_angles

min_angle = min(all_angles)
max_angle = max(all_angles) 
mean_angle = sum(all_angles) / len(all_angles)

def calculate_head_angle_adaptive(skeleton, prev_angle=None, base_head_length=15, 
                              body_segment_length=200, max_angle_change=20, 
                              max_head_length=50):
    """
    Calculate the angle between head tip and body with adaptive head segment length.
    
    Parameters:
    -----------
    skeleton : ndarray
        Binary skeleton image
    prev_angle : float, optional
        Angle from previous frame for comparison
    base_head_length : int
        Initial number of points to use for head direction
    body_segment_length : int
        Number of points to use for body direction
    max_angle_change : float
        Maximum allowed angle change between frames
    max_head_length : int
        Maximum allowed head segment length
    
    Returns:
    --------
    dict
        Dictionary containing angle information and parameters used
    """
    def calc_angle_with_length(points, head_length, body_length):
        """Helper function to calculate angle with specific head length"""
        # Get head direction vector
        head_vector = points[head_length] - points[0]
        
        # Get body direction vector
        body_vector = points[-1] - points[-body_length]
        
        # Calculate angle between vectors
        dot_product = np.dot(head_vector, body_vector)
        norms = np.linalg.norm(head_vector) * np.linalg.norm(body_vector)
        
        if norms == 0:
            return None, head_vector, body_vector
            
        cos_angle = dot_product / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # Determine direction using cross product
        cross_product = np.cross(body_vector, head_vector)
        if cross_product < 0:
            angle_deg = -angle_deg
            
        return angle_deg, head_vector, body_vector

    # Get ordered points along the skeleton
    points = np.column_stack(np.where(skeleton))
    ordered_points = points[np.argsort(points[:, 0])]
    
    # Interpolate to get smooth curve
    x, y = ordered_points[:, 1], ordered_points[:, 0]
    tck, u = splprep([x, y], s=0)
    u_new = np.linspace(0, 1, num=500)
    smooth_x, smooth_y = splev(u_new, tck)
    smooth_points = np.column_stack((smooth_y, smooth_x))
    
    # Start with base head length
    current_head_length = base_head_length
    angle_deg, head_vector, body_vector = calc_angle_with_length(
        smooth_points, current_head_length, body_segment_length)
    
    # If we have a previous angle, check if we need to adjust head length
    if prev_angle is not None and angle_deg is not None:
        while (abs(angle_deg - prev_angle) > max_angle_change and 
               current_head_length < max_head_length):
            current_head_length += 5
            new_angle, head_vector, body_vector = calc_angle_with_length(
                smooth_points, current_head_length, body_segment_length)
            
            if new_angle is None:
                break
                
            angle_deg = new_angle
            
            # If we're not improving, break
            if abs(angle_deg - prev_angle) >= abs(angle_deg - prev_angle):
                break
    
    return {
        'angle_degrees': float(angle_deg) if angle_deg is not None else None,
        'head_vector': head_vector.tolist(),
        'body_vector': body_vector.tolist(),
        'direction': 'left' if np.cross(body_vector, head_vector) > 0 else 'right',
        'head_segment_length': current_head_length,
        'head_point': smooth_points[0].tolist(),
        'body_start': smooth_points[-body_segment_length].tolist(),
        'body_end': smooth_points[-1].tolist(),
        'skeleton': skeleton.tolist()
    }

# Modified processing loop
head_angle_results = {}
all_angles = []
frame_angles_dict = {}
prev_angles = {}  # Store previous angles for each object

for frame_idx in sorted(truncated_skeletons.keys()):
    frame_data = truncated_skeletons[frame_idx]
    frame_angles = {}
    frame_angles_dict[frame_idx] = []
    
    for obj_id, skeleton_data in frame_data.items():
        skeleton = skeleton_data[0]
        prev_angle = prev_angles.get(obj_id, None)
        
        angle_info = calculate_head_angle_adaptive(
            skeleton,
            prev_angle=prev_angle,
            base_head_length=15,
            body_segment_length=200,
            max_angle_change=20,
            max_head_length=50
        )
        
        if angle_info['angle_degrees'] is not None:
            prev_angles[obj_id] = angle_info['angle_degrees']
            frame_angles[obj_id] = angle_info
            all_angles.append(angle_info['angle_degrees'])
            frame_angles_dict[frame_idx].append(angle_info['angle_degrees'])
    
    head_angle_results[frame_idx] = frame_angles

# Get min, max and mean angles
min_angle = min(all_angles)
max_angle = max(all_angles)
mean_angle = sum(all_angles) / len(all_angles)

# Plot angles over time
plt.figure(figsize=(12, 6))
frames = sorted(frame_angles_dict.keys())
for obj_id in set(prev_angles.keys()):
    angles = []
    frames_with_obj = []
    for frame in frames:
        if obj_id in head_angle_results[frame]:
            angles.append(head_angle_results[frame][obj_id]['angle_degrees'])
            frames_with_obj.append(frame)
    plt.plot(frames_with_obj, angles, label=f'Object {obj_id}')

plt.axhline(y=mean_angle, color='r', linestyle='--', label='Mean angle')
plt.xlabel('Frame')
plt.ylabel('Head Angle (degrees)')
plt.title('Head Angles Over Time')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('head_angles_over_time.png')
plt.close()

print(f"Min angle: {min_angle:.2f}°")
print(f"Max angle: {max_angle:.2f}°") 
print(f"Mean angle: {mean_angle:.2f}°")




def calculate_head_angle_diagnostic(skeleton, prev_angle=None, prev_vectors=None,
                              base_head_length=15, body_segment_length=200,
                              max_angle_change=20, max_head_length=50):
    """
    Enhanced version with diagnostic information and additional checks.
    """
    def calc_angle_with_length(points, head_length, body_length):
        """Helper function with additional vector consistency checks"""
        if len(points) < max(head_length + 1, body_length + 1):
            return {
                'angle': None,
                'head_vector': None,
                'body_vector': None,
                'error': 'Insufficient points'
            }
        
        # Get vectors
        head_vector = points[head_length] - points[0]
        body_vector = points[-1] - points[-body_length]
        
        # Check vector magnitudes
        head_mag = np.linalg.norm(head_vector)
        body_mag = np.linalg.norm(body_vector)
        
        if head_mag < 1e-6 or body_mag < 1e-6:
            return {
                'angle': None,
                'head_vector': head_vector,
                'body_vector': body_vector,
                'error': 'Zero magnitude vector'
            }
            
        # Normalize vectors
        head_vector_norm = head_vector / head_mag
        body_vector_norm = body_vector / body_mag
        
        # Calculate angle
        dot_product = np.dot(head_vector_norm, body_vector_norm)
        cos_angle = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # Determine direction using cross product
        cross_product = np.cross(body_vector_norm, head_vector_norm)
        if cross_product < 0:
            angle_deg = -angle_deg
            
        return {
            'angle': angle_deg,
            'head_vector': head_vector,
            'body_vector': body_vector,
            'head_mag': head_mag,
            'body_mag': body_mag,
            'cross_product': cross_product,
            'dot_product': dot_product,
            'error': None
        }

    # Get ordered points along the skeleton
    points = np.column_stack(np.where(skeleton))
    
    # Check if we have enough points
    if len(points) < base_head_length + body_segment_length:
        return {
            'angle_degrees': None,
            'error': 'Insufficient skeleton points',
            'diagnostics': {
                'points_count': len(points),
                'required_points': base_head_length + body_segment_length
            }
        }
    
    # Order points and ensure consistent orientation
    ordered_points = points[np.argsort(points[:, 0])]
    
    # Interpolate to get smooth curve
    try:
        x, y = ordered_points[:, 1], ordered_points[:, 0]
        tck, u = splprep([x, y], s=0)
        u_new = np.linspace(0, 1, num=500)
        smooth_x, smooth_y = splev(u_new, tck)
        smooth_points = np.column_stack((smooth_y, smooth_x))
    except Exception as e:
        return {
            'angle_degrees': None,
            'error': f'Interpolation failed: {str(e)}',
            'diagnostics': {
                'points_shape': points.shape,
                'ordered_points_shape': ordered_points.shape
            }
        }
    
    # Calculate initial angle
    current_head_length = base_head_length
    result = calc_angle_with_length(smooth_points, current_head_length, body_segment_length)
    
    # Store diagnostic information
    diagnostics = {
        'initial_head_length': base_head_length,
        'final_head_length': current_head_length,
        'vector_magnitudes': {
            'head': result['head_mag'] if result['head_vector'] is not None else None,
            'body': result['body_mag'] if result['body_vector'] is not None else None
        },
        'dot_product': result.get('dot_product'),
        'cross_product': result.get('cross_product')
    }
    
    # Check for extreme angle changes
    if prev_angle is not None and result['angle'] is not None:
        angle_change = abs(result['angle'] - prev_angle)
        diagnostics['angle_change'] = angle_change
        
        # Try to detect potential orientation flips
        if angle_change > 150:  # Likely orientation flip
            diagnostics['possible_flip'] = True
            
        # Check vector consistency with previous frame
        if prev_vectors is not None:
            prev_head_vec = np.array(prev_vectors['head'])
            prev_body_vec = np.array(prev_vectors['body'])
            current_head_vec = result['head_vector']
            current_body_vec = result['body_vector']
            
            head_consistency = np.dot(prev_head_vec, current_head_vec) / (np.linalg.norm(prev_head_vec) * np.linalg.norm(current_head_vec))
            body_consistency = np.dot(prev_body_vec, current_body_vec) / (np.linalg.norm(prev_body_vec) * np.linalg.norm(current_body_vec))
            
            diagnostics['vector_consistency'] = {
                'head': head_consistency,
                'body': body_consistency
            }
    
    return {
        'angle_degrees': result['angle'],
        'head_vector': result['head_vector'].tolist() if result['head_vector'] is not None else None,
        'body_vector': result['body_vector'].tolist() if result['body_vector'] is not None else None,
        'diagnostics': diagnostics,
        'error': result.get('error')
    }


head_angle_results = {}
diagnostics_log = []
prev_angles = {}
prev_vectors = {}

for frame_idx in sorted(truncated_skeletons.keys()):
    frame_data = truncated_skeletons[frame_idx]
    frame_angles = {}
    
    for obj_id, skeleton_data in frame_data.items():
        skeleton = skeleton_data[0]
        prev_angle = prev_angles.get(obj_id)
        prev_vec = prev_vectors.get(obj_id)
        
        result = calculate_head_angle_diagnostic(
            skeleton,
            prev_angle=prev_angle,
            prev_vectors=prev_vec,
            base_head_length=15,
            body_segment_length=200
        )
        
        # Log diagnostic information
        diagnostics_log.append({
            'frame': frame_idx,
            'object_id': obj_id,
            'diagnostics': result['diagnostics'],
            'error': result.get('error'),
            'angle': result['angle_degrees']
        })
        
        if result['angle_degrees'] is not None:
            prev_angles[obj_id] = result['angle_degrees']
            prev_vectors[obj_id] = {
                'head': result['head_vector'],
                'body': result['body_vector']
            }
            
        frame_angles[obj_id] = result
    
    head_angle_results[frame_idx] = frame_angles

# Function to analyze diagnostic log and identify problematic frames
def analyze_diagnostics(log):
    problems = []
    for entry in log:
        frame = entry['frame']
        diagnostics = entry['diagnostics']
        
        # Check for potential issues
        if 'angle_change' in diagnostics and diagnostics['angle_change'] > 20:
            problems.append({
                'frame': frame,
                'issue': 'Large angle change',
                'change': diagnostics['angle_change'],
                'vectors': diagnostics.get('vector_consistency')
            })
        
        if diagnostics.get('possible_flip'):
            problems.append({
                'frame': frame,
                'issue': 'Possible orientation flip',
                'vector_consistency': diagnostics.get('vector_consistency')
            })
            
        # Check vector magnitudes
        vec_mags = diagnostics.get('vector_magnitudes', {})
        if vec_mags.get('head', 0) < 5 or vec_mags.get('body', 0) < 5:
            problems.append({
                'frame': frame,
                'issue': 'Small vector magnitude',
                'magnitudes': vec_mags
            })
    
    return problems




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

def calculate_head_angle_normalized(skeleton, prev_angle=None, min_vector_length=10):
    """
    Calculate head angle using normalized skeleton points and relative distances.
    
    Parameters:
    -----------
    skeleton : ndarray
        Binary skeleton image
    prev_angle : float, optional
        Previous frame's angle for comparison
    min_vector_length : float
        Minimum required vector length in pixels
    """
    # Get ordered points along the skeleton
    points = np.column_stack(np.where(skeleton))
    ordered_points = points[np.argsort(points[:, 0])]
    
    # Normalize points spacing
    norm_points, total_length = normalize_skeleton_points(ordered_points, num_points=100)
    
    # Define head and body sections as percentages of total length
    head_section = 0.15  # Use first 15% for head
    body_section = 0.4   # Use last 40% for body direction
    
    # Calculate indices for head and body vectors
    head_end_idx = int(head_section * len(norm_points))
    body_start_idx = int((1 - body_section) * len(norm_points))
    
    # Calculate vectors
    head_vector = norm_points[head_end_idx] - norm_points[0]
    body_vector = norm_points[-1] - norm_points[body_start_idx]
    
    # Check vector magnitudes
    head_mag = np.linalg.norm(head_vector)
    body_mag = np.linalg.norm(body_vector)
    
    if head_mag < min_vector_length or body_mag < min_vector_length:
        return {
            'angle_degrees': prev_angle if prev_angle is not None else 0,
            'error': 'Vector magnitude too small',
            'head_mag': head_mag,
            'body_mag': body_mag
        }
    
    # Calculate angle
    dot_product = np.dot(head_vector, body_vector)
    cos_angle = np.clip(dot_product / (head_mag * body_mag), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Determine direction using cross product
    cross_product = np.cross(body_vector, head_vector)
    if cross_product < 0:
        angle_deg = -angle_deg
    
    # If we have a previous angle and the change is too large, 
    # use the previous angle
    if prev_angle is not None:
        angle_change = abs(angle_deg - prev_angle)
        if angle_change > 20:  # Maximum allowed change
            return {
                'angle_degrees': prev_angle,
                'error': 'Angle change too large',
                'proposed_angle': angle_deg,
                'angle_change': angle_change
            }
    
    return {
        'angle_degrees': float(angle_deg),
        'head_vector': head_vector.tolist(),
        'body_vector': body_vector.tolist(),
        'head_mag': float(head_mag),
        'body_mag': float(body_mag),
        'error': None
    }

# Modified processing loop
head_angle_results = {}
all_angles = []
frame_angles_dict = {}
prev_angles = {}

for frame_idx in sorted(truncated_skeletons.keys()):
    frame_data = truncated_skeletons[frame_idx]
    frame_angles = {}
    frame_angles_dict[frame_idx] = []
    
    for obj_id, skeleton_data in frame_data.items():
        skeleton = skeleton_data[0]
        prev_angle = prev_angles.get(obj_id)
        
        result = calculate_head_angle_normalized(
            skeleton,
            prev_angle=prev_angle,
            min_vector_length=10
        )
        
        if result['angle_degrees'] is not None:
            prev_angles[obj_id] = result['angle_degrees']
            frame_angles[obj_id] = result
            all_angles.append(result['angle_degrees'])
            frame_angles_dict[frame_idx].append(result['angle_degrees'])
    
    head_angle_results[frame_idx] = frame_angles





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

# Create DataFrame with angle data
angle_df = pd.DataFrame(angle_data)
angle_df.to_csv('angle_data.csv', index=False)

# Print min/max angles
if all_angles:
    print(f"Min angle: {min(all_angles):.2f}°")
    print(f"Max angle: {max(all_angles):.2f}°")

# Plot angles over time
plt.figure(figsize=(12, 6))
frames = sorted(frame_angles_dict.keys())
for obj_id in set(prev_angles.keys()):
    angles = []
    frames_with_obj = []
    for frame in frames:
        if obj_id in head_angle_results[frame]:
            angles.append(head_angle_results[frame][obj_id]['angle_degrees'])
            frames_with_obj.append(frame)
    
    # Plot raw angles
    plt.plot(frames_with_obj, angles, label=f'Object {obj_id} (raw)')
    
    # Calculate and plot rolling mean with small window
    angles_array = np.array(angles)
    window_size = 5
    rolling_mean = np.convolve(angles_array, np.ones(window_size)/window_size, mode='valid')
    rolling_mean_frames = frames_with_obj[window_size-1:]
    plt.plot(rolling_mean_frames, rolling_mean, 
             label=f'Object {obj_id} (rolling mean w={window_size})', 
             linewidth=2)
             
    # Calculate and plot rolling mean with larger window
    large_window = 10  # Larger window size
    large_rolling_mean = np.convolve(angles_array, np.ones(large_window)/large_window, mode='valid')
    large_rolling_frames = frames_with_obj[large_window-1:]
    plt.plot(large_rolling_frames, large_rolling_mean,
             label=f'Object {obj_id} (rolling mean w={large_window})',
             linewidth=2)

plt.xlabel('Frame')
plt.ylabel('Head Angle (degrees)')
plt.title('Head Angles Over Time')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('head_angles_over_time.png')
plt.close()

# Print min/max angles
if all_angles:
    print(f"Min angle: {min(all_angles):.2f}°")
    print(f"Max angle: {max(all_angles):.2f}°")



#Check one frame
frame_idx = 544
obj_id = 2
skeleton = truncated_skeletons[frame_idx][obj_id][0]

# Calculate angles and vectors
angle_info = calculate_head_angle_v2(skeleton, head_segment_length=15, body_segment_length=200)

plt.figure(figsize=(8, 8))
plt.imshow(skeleton, cmap='gray')

head_point = angle_info['head_point']
head_vector = angle_info['head_vector']
body_start = angle_info['body_start']
body_vector = angle_info['body_vector']

# Plot vectors
# Head vector (red)
plt.arrow(head_point[1], head_point[0], 
          head_vector[1], head_vector[0],
          color='red', width=0.5, label='Head direction')

# Body vector (blue)
plt.arrow(body_start[1], body_start[0],
          body_vector[1], body_vector[0],
          color='blue', width=0.5, label='Body direction')

plt.title(f"Head Angle: {angle_info['angle_degrees']:.1f}°\nBend Direction: {angle_info['direction']}")
plt.legend()
plt.axis('equal')
plt.show()
plt.savefig('tst.png')
plt.close()

print(angle_info['angle_degrees'])




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


def interpolate_straight_frames(df):
    """
    Interpolate bend locations ONLY for straight frames, keeping all other values unchanged.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Work on each object separately
    for obj_id in df['object_id'].unique():
        # Get data for this object
        mask = df['object_id'] == obj_id
        obj_data = df[mask].copy()
        
        # Identify straight frames
        straight_mask = (obj_data['angle_degrees'].abs() <= 10)
        
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
straight_mask = df_interpolated['angle_degrees'].abs() <= 10
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


frame_idx = list(truncated_skeletons.keys())[5]  # First frame
obj_id = list(truncated_skeletons[frame_idx].keys())[0]  # First object

plt.figure(figsize=(10, 10))
plot_skeleton_with_bend(frame_idx, obj_id, truncated_skeletons, head_angle_results)
plt.tight_layout()
plt.show()
plt.savefig('tst.png')
plt.close()






# Plot bend location over time
# Convert df to numpy array before plotting
frame_data = df_interpolated['frame'].to_numpy()
bend_data = df_interpolated['bend_location'].to_numpy()
angle_data = np.abs(df_interpolated['angle_degrees'].to_numpy())  # Take absolute value

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

frame = 0
mask = 4
mask = skeletons[0][2][0]

image_array = np.uint8(skeleton[0] * 255)
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