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

def calculate_worm_curvature(skeleton, window_size=50, sigma=10, n_points=100):
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


frame_idx = 0
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