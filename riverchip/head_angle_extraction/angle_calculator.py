"""
Head angle calculation utilities.
Contains the original complex business logic for calculating head angles from skeletons.
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
from scipy import interpolate, ndimage
from scipy.spatial.distance import cdist
from .config import Config

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

def order_skeleton_points_by_path(skeleton):
    """
    Order skeleton points by following the connected path from one tip to the other.
    This ensures proper head-to-tail ordering regardless of curvature.
    
    Parameters:
    -----------
    skeleton : ndarray
        Binary skeleton image
        
    Returns:
    --------
    ordered_points : ndarray
        Points ordered from one tip to the other
    success : bool
        Whether the ordering was successful
    """
    # Get all skeleton points
    points = np.column_stack(np.where(skeleton))
    if len(points) == 0:
        return points, False
    
    # If only one or two points, return as is
    if len(points) <= 2:
        return points, True
    
    try:
        # Build adjacency information by finding neighbors for each point
        # Two points are neighbors if they are within distance sqrt(2) (diagonal neighbors)
        distances = cdist(points, points)
        adjacency = distances <= np.sqrt(2)
        np.fill_diagonal(adjacency, False)  # Remove self-connections
        
        # Count neighbors for each point
        neighbor_counts = np.sum(adjacency, axis=1)
        
        # Find endpoints (points with only 1 neighbor)
        endpoints = np.where(neighbor_counts == 1)[0]
        
        # If we don't have exactly 2 endpoints, try to handle gracefully
        if len(endpoints) == 0:
            # No clear endpoints - this might be a loop or very noisy skeleton
            # Fall back to using the points that are furthest apart
            distances_between_points = cdist(points, points)
            max_dist_idx = np.unravel_index(np.argmax(distances_between_points), distances_between_points.shape)
            start_idx = max_dist_idx[0]
        elif len(endpoints) == 1:
            # Only one endpoint - start from there
            start_idx = endpoints[0]
        elif len(endpoints) >= 2:
            # Multiple endpoints - choose the two that are furthest apart
            endpoint_distances = cdist(points[endpoints], points[endpoints])
            max_dist_idx = np.unravel_index(np.argmax(endpoint_distances), endpoint_distances.shape)
            start_idx = endpoints[max_dist_idx[0]]
        
        # Follow the path from the starting point
        ordered_indices = []
        current_idx = start_idx
        visited = set()
        
        while current_idx is not None and current_idx not in visited:
            ordered_indices.append(current_idx)
            visited.add(current_idx)
            
            # Find unvisited neighbors
            neighbors = np.where(adjacency[current_idx])[0]
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if len(unvisited_neighbors) == 0:
                # No more unvisited neighbors - we're done
                break
            elif len(unvisited_neighbors) == 1:
                # Only one choice - continue along the path
                current_idx = unvisited_neighbors[0]
            else:
                # Multiple unvisited neighbors - choose the one that continues the path most smoothly
                if len(ordered_indices) >= 2:
                    # Calculate direction from previous point to current point
                    prev_point = points[ordered_indices[-2]]
                    curr_point = points[current_idx]
                    current_direction = curr_point - prev_point
                    current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
                    
                    # Choose neighbor that best continues this direction
                    best_neighbor = None
                    best_alignment = -2  # Worst possible dot product
                    
                    for neighbor_idx in unvisited_neighbors:
                        neighbor_point = points[neighbor_idx]
                        neighbor_direction = neighbor_point - curr_point
                        neighbor_direction = neighbor_direction / (np.linalg.norm(neighbor_direction) + 1e-10)
                        
                        # Calculate alignment with current direction
                        alignment = np.dot(current_direction, neighbor_direction)
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_neighbor = neighbor_idx
                    
                    current_idx = best_neighbor
                else:
                    # Not enough history - just pick the first unvisited neighbor
                    current_idx = unvisited_neighbors[0]
        
        # If we didn't visit all points, there might be disconnected components
        # Add any remaining points at the end (this handles noisy skeletons)
        remaining_points = set(range(len(points))) - visited
        if remaining_points:
            # Sort remaining points by distance to the last ordered point
            if ordered_indices:
                last_point = points[ordered_indices[-1]]
                remaining_indices = list(remaining_points)
                remaining_distances = [np.linalg.norm(points[idx] - last_point) for idx in remaining_indices]
                sorted_remaining = [x for _, x in sorted(zip(remaining_distances, remaining_indices))]
                ordered_indices.extend(sorted_remaining)
            else:
                ordered_indices.extend(list(remaining_points))
        
        # Return the ordered points
        ordered_points = points[ordered_indices]
        
        # Verify we have a reasonable result
        if len(ordered_points) != len(points):
            # Something went wrong - fall back to coordinate-based sorting
            return points[np.argsort(points[:, 0])], False
        
        return ordered_points, True
        
    except Exception as e:
        # If anything goes wrong, fall back to coordinate-based sorting
        Config.debug_print(f"Path-based ordering failed: {e}, falling back to coordinate sorting")
        return points[np.argsort(points[:, 0])], False

def determine_head_end(ordered_points, skeleton_shape):
    """
    Determine which end of the ordered skeleton is the head.
    Uses heuristics based on position and skeleton shape.
    
    Parameters:
    -----------
    ordered_points : ndarray
        Skeleton points ordered from tip to tip
    skeleton_shape : tuple
        Shape of the skeleton image (height, width)
        
    Returns:
    --------
    head_first : bool
        True if the head is at the beginning of ordered_points, False if at the end
    """
    if len(ordered_points) < 2:
        return True  # Default assumption
    
    # Get the two endpoints
    start_point = ordered_points[0]
    end_point = ordered_points[-1]
    
    # Heuristic 1: Head is usually closer to the top of the image
    # (assuming worms are oriented with head up or at least not at bottom)
    start_y, start_x = start_point
    end_y, end_x = end_point
    
    # The point with smaller y-coordinate (closer to top) is more likely to be the head
    if start_y < end_y:
        return True  # Start is head
    elif end_y < start_y:
        return False  # End is head
    
    # If y-coordinates are similar, use x-coordinate as tiebreaker
    # This is less reliable but better than random
    if start_x < end_x:
        return True  # Prefer left side as head (arbitrary choice)
    else:
        return False
    
    # Could add more sophisticated heuristics here:
    # - Analyze local curvature at endpoints
    # - Use information from previous frames
    # - Analyze thickness patterns
    # But for now, position-based heuristics should work reasonably well

def calculate_head_angle_with_positions_and_bend(skeleton, prev_angle=None, min_vector_length=5, 
                                             restriction_point=0.4, straight_threshold=3):
    """
    Calculate head angle and bend location along the skeleton.
    Never drops frames - always returns a result dictionary with appropriate error handling.
    For straight worms (angle <= straight_threshold), bend values are set to 0.
    
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
        - angle_degrees: calculated angle (0 if calculation fails)
        - head positions and vectors
        - bend_location: relative position along skeleton (0-1), 0 if straight
        - bend_magnitude: strength of the bend, 0 if straight
        - bend_position: (y,x) coordinates of maximum bend point, [0,0] if straight
        - error: error message if calculation fails, None otherwise
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
                'skeleton_points': [[0, 0]],  # Default point
                'curvature_profile': [0],
                'head_start_pos': [0, 0],
                'head_end_pos': [0, 0],
                'body_start_pos': [0, 0],
                'body_end_pos': [0, 0],
                'head_vector': [0, 0],
                'body_vector': [0, 0]
            }
        
        # Use path-based ordering instead of coordinate-based sorting
        ordered_points, ordering_success = order_skeleton_points_by_path(skeleton)
        
        # Determine which end is the head
        head_first = determine_head_end(ordered_points, skeleton.shape)
        
        # If head is at the end, reverse the order so head is at the beginning
        if not head_first:
            ordered_points = ordered_points[::-1]
        
        # Add debug information about ordering
        if Config.DEBUG_MODE and not ordering_success:
            Config.debug_print("Path-based ordering failed, using coordinate fallback")
        
        norm_points, total_length = normalize_skeleton_points(ordered_points, num_points=100)
        
        # Calculate head angle - EXACT SAME LOGIC AS WORKING VERSION
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
            
            # EXACT SAME ANGLE CALCULATION AS WORKING VERSION
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
                    'error': None
                }
        
        if best_result is None:
            # No valid angle found - use previous angle or default
            best_result = {
                'angle_degrees': prev_angle if prev_angle is not None else 0,
                'error': 'No valid angle found with current parameters',
                'head_mag': 0,
                'body_mag': 0,
                'head_start_pos': norm_points[0].tolist(),
                'head_end_pos': norm_points[min(5, len(norm_points)-1)].tolist(),
                'body_start_pos': norm_points[max(0, len(norm_points)-10)].tolist(),
                'body_end_pos': norm_points[-1].tolist(),
                'head_vector': [0, 0],
                'body_vector': [0, 0],
                'skeleton_points': norm_points.tolist()
            }
        
        # If the worm is straight (angle within threshold), set all bend-related values to 0
        if abs(best_result['angle_degrees']) <= straight_threshold:
            best_result.update({
                'bend_location': 0,
                'bend_magnitude': 0,
                'bend_position': [0, 0],
                'curvature_profile': np.zeros(len(norm_points)).tolist(),
                'is_straight': True
            })
        else:
            # Calculate curvature only for non-straight worms
            curvatures = gaussian_weighted_curvature(norm_points, window_size=25, sigma=8, 
                                                   restriction_point=restriction_point)
            
            # Find the location of maximum bend (only consider points up to restriction)
            valid_range = int(len(curvatures) * restriction_point)
            max_curvature_idx = np.argmax(np.abs(curvatures[:valid_range]))
            bend_location = max_curvature_idx / len(curvatures)  # Normalized position (0-1)
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
            'skeleton_points': [[0, 0]],  # Default point
            'curvature_profile': [0],
            'head_start_pos': [0, 0],
            'head_end_pos': [0, 0],
            'body_start_pos': [0, 0],
            'body_end_pos': [0, 0],
            'head_vector': [0, 0],
            'body_vector': [0, 0],
            'is_straight': True
        }

def interpolate_results(prev_result, next_result, weight_prev, weight_next, straight_threshold=3):
    """
    Interpolate between two results based on weights.
    If the interpolated angle is within the straight threshold, all bend values are set to 0.
    """
    def interpolate_value(prev_val, next_val):
        if isinstance(prev_val, (list, np.ndarray)):
            # Handle the case where elements might be nested lists
            result = []
            for p, n in zip(prev_val, next_val):
                if isinstance(p, (list, np.ndarray)):
                    # Handle nested lists recursively
                    result.append(interpolate_value(p, n))
                else:
                    # Handle simple number case
                    result.append(weight_prev * p + weight_next * n)
            return result
        return weight_prev * prev_val + weight_next * next_val
    
    interpolated = {}
    # First interpolate the angle to check if result should be straight
    angle_degrees = interpolate_value(prev_result['angle_degrees'], next_result['angle_degrees'])
    
    # If interpolated angle is within straight threshold, return straight result
    if abs(angle_degrees) <= straight_threshold:
        return {
            'angle_degrees': angle_degrees,
            'head_mag': interpolate_value(prev_result['head_mag'], next_result['head_mag']),
            'body_mag': interpolate_value(prev_result['body_mag'], next_result['body_mag']),
            'bend_location': 0,
            'bend_magnitude': 0,
            'bend_position': [0, 0],
            'head_start_pos': interpolate_value(prev_result['head_start_pos'], next_result['head_start_pos']),
            'head_end_pos': interpolate_value(prev_result['head_end_pos'], next_result['head_end_pos']),
            'body_start_pos': interpolate_value(prev_result['body_start_pos'], next_result['body_start_pos']),
            'body_end_pos': interpolate_value(prev_result['body_end_pos'], next_result['body_end_pos']),
            'head_vector': interpolate_value(prev_result['head_vector'], next_result['head_vector']),
            'body_vector': interpolate_value(prev_result['body_vector'], next_result['body_vector']),
            'is_straight': True
        }
    
    # Otherwise interpolate all values
    for key in prev_result.keys():
        if key in ['error', 'is_straight']:
            continue
        interpolated[key] = interpolate_value(prev_result[key], next_result[key])
    
    interpolated['is_straight'] = False
    return interpolated

def decay_result(base_result, decay_factor, straight_threshold=3):
    """
    Apply decay to a result, making it trend toward default values.
    If the decayed angle is within the straight threshold, all bend values are set to 0.
    """
    def apply_decay(value):
        """Helper function to recursively apply decay to nested lists"""
        if isinstance(value, (list, np.ndarray)):
            return [apply_decay(v) for v in value]
        else:
            return value * decay_factor
    
    # First decay the angle to check if result should be straight
    angle_degrees = base_result['angle_degrees'] * decay_factor
    
    # If decayed angle is within straight threshold, return straight result
    if abs(angle_degrees) <= straight_threshold:
        return {
            'angle_degrees': angle_degrees,
            'head_mag': base_result['head_mag'] * decay_factor,
            'body_mag': base_result['body_mag'] * decay_factor,
            'bend_location': 0,
            'bend_magnitude': 0,
            'bend_position': [0, 0],
            'head_start_pos': apply_decay(base_result['head_start_pos']),
            'head_end_pos': apply_decay(base_result['head_end_pos']),
            'body_start_pos': apply_decay(base_result['body_start_pos']),
            'body_end_pos': apply_decay(base_result['body_end_pos']),
            'head_vector': apply_decay(base_result['head_vector']),
            'body_vector': apply_decay(base_result['body_vector']),
            'is_straight': True
        }
    
    # Otherwise decay all values
    decayed = {}
    for key, value in base_result.items():
        if key in ['error', 'is_straight']:
            continue
        decayed[key] = apply_decay(value)
    
    decayed['is_straight'] = False
    return decayed

def calculate_head_angles(truncated_skeletons):
    """
    Process a batch of skeletons with head angle smoothing and bend recalculation.
    Handles all frames gracefully with smooth interpolation between valid frames.
    Uses the EXACT SAME LOGIC as the working version.
    """
    Config.debug_print("Processing skeleton batch using exact working algorithm")
    
    # Use the exact same parameters as the working version
    min_vector_length = 5
    restriction_point = 0.5  # Main call uses 0.5
    straight_threshold = 3
    smoothing_window = 3
    deviation_threshold = 50  # Main call uses 50, not 15
    
    # First pass: Calculate initial angles with graceful error handling
    initial_data = []
    frame_results = {}  # Store results by frame and object
    
    # First pass to collect all valid results
    for frame_idx in sorted(truncated_skeletons.keys()):
        frame_data = truncated_skeletons[frame_idx]
        frame_results[frame_idx] = {}
        
        for obj_id, skeleton_data in frame_data.items():
            # Handle different skeleton data formats
            if isinstance(skeleton_data, np.ndarray):
                if skeleton_data.ndim > 2:
                    skeleton = skeleton_data[0]  # Take first channel
                else:
                    skeleton = skeleton_data
            else:
                skeleton = skeleton_data
            
            result = calculate_head_angle_with_positions_and_bend(
                skeleton,
                prev_angle=None,  # Don't use prev_angle in first pass
                min_vector_length=min_vector_length,
                restriction_point=restriction_point,
                straight_threshold=straight_threshold
            )
            
            # Store result regardless of error status
            frame_results[frame_idx][obj_id] = result
    
    # Second pass to interpolate invalid frames
    for frame_idx in sorted(truncated_skeletons.keys()):
        for obj_id in frame_results[frame_idx].keys():
            result = frame_results[frame_idx][obj_id]
            
            if result['error'] is not None:
                # Find previous and next valid frames
                prev_valid_frame = None
                prev_valid_result = None
                next_valid_frame = None
                next_valid_result = None
                
                # Look for previous valid frame
                for prev_frame in range(frame_idx - 1, -1, -1):
                    if prev_frame in frame_results and obj_id in frame_results[prev_frame]:
                        prev_result = frame_results[prev_frame][obj_id]
                        if prev_result['error'] is None:
                            prev_valid_frame = prev_frame
                            prev_valid_result = prev_result
                            break
                
                # Look for next valid frame
                for next_frame in range(frame_idx + 1, max(frame_results.keys()) + 1):
                    if next_frame in frame_results and obj_id in frame_results[next_frame]:
                        next_result = frame_results[next_frame][obj_id]
                        if next_result['error'] is None:
                            next_valid_frame = next_frame
                            next_valid_result = next_result
                            break
                
                # Interpolate based on available valid frames
                if prev_valid_result and next_valid_result:
                    total_frames = next_valid_frame - prev_valid_frame
                    weight_next = (frame_idx - prev_valid_frame) / total_frames
                    weight_prev = 1 - weight_next
                    
                    interpolated_result = interpolate_results(
                        prev_valid_result, next_valid_result, 
                        weight_prev, weight_next,
                        straight_threshold=straight_threshold)
                    interpolated_result['error'] = f"Interpolated between frames {prev_valid_frame} and {next_valid_frame}"
                    frame_results[frame_idx][obj_id] = interpolated_result
                    
                elif prev_valid_result:
                    frames_since_valid = frame_idx - prev_valid_frame
                    decay_factor = 0.9 ** frames_since_valid
                    
                    interpolated_result = decay_result(
                        prev_valid_result, decay_factor,
                        straight_threshold=straight_threshold)
                    interpolated_result['error'] = f"Decayed from previous frame {prev_valid_frame}"
                    frame_results[frame_idx][obj_id] = interpolated_result
                    
                elif next_valid_result:
                    frames_until_valid = next_valid_frame - frame_idx
                    decay_factor = 0.9 ** frames_until_valid
                    
                    interpolated_result = decay_result(
                        next_valid_result, decay_factor,
                        straight_threshold=straight_threshold)
                    interpolated_result['error'] = f"Decayed from next frame {next_valid_frame}"
                    frame_results[frame_idx][obj_id] = interpolated_result
                    
                else:
                    # No valid frames available - use default values with straight configuration
                    frame_results[frame_idx][obj_id] = {
                        'angle_degrees': 0.0,
                        'head_mag': 0.0,
                        'body_mag': 0.0,
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
                        'error': 'No valid frames available for interpolation'
                    }
            
            # Add to initial_data for further processing
            initial_data.append({
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': frame_results[frame_idx][obj_id]['angle_degrees'],
                'bend_location': frame_results[frame_idx][obj_id]['bend_location'],
                'bend_magnitude': frame_results[frame_idx][obj_id]['bend_magnitude'],
                'bend_position_y': frame_results[frame_idx][obj_id]['bend_position'][0],
                'bend_position_x': frame_results[frame_idx][obj_id]['bend_position'][1],
                'head_mag': frame_results[frame_idx][obj_id]['head_mag'],
                'body_mag': frame_results[frame_idx][obj_id]['body_mag'],
                'is_straight': frame_results[frame_idx][obj_id].get('is_straight', abs(frame_results[frame_idx][obj_id]['angle_degrees']) <= straight_threshold),
                'error': frame_results[frame_idx][obj_id].get('error', None)
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
    
    # Final pass: Recalculate bend positions based on smoothed angles
    final_data = []
    
    # Group by object_id to maintain continuity
    for obj_id in smoothed_df['object_id'].unique():
        obj_data = smoothed_df[smoothed_df['object_id'] == obj_id]
        
        prev_angle = None
        for _, row in obj_data.iterrows():
            frame_idx = row['frame']
            # Get the corresponding skeleton
            skeleton_data = truncated_skeletons[frame_idx][obj_id]
            if isinstance(skeleton_data, np.ndarray):
                if skeleton_data.ndim > 2:
                    skeleton = skeleton_data[0]  # Take first channel
                else:
                    skeleton = skeleton_data
            else:
                skeleton = skeleton_data
            
            # Calculate bend position with the smoothed angle
            new_result = calculate_head_angle_with_positions_and_bend(
                skeleton,
                prev_angle=prev_angle,
                min_vector_length=min_vector_length,
                restriction_point=restriction_point,
                straight_threshold=straight_threshold
            )
            prev_angle = row['angle_degrees']
            
            # Use the smoothed angle from the dataframe but updated bend calculations
            is_straight = abs(row['angle_degrees']) <= straight_threshold
            
            # Set bend values based on straight vs non-straight
            if is_straight:
                bend_location = 0
                bend_magnitude = 0
                bend_position_y = 0
                bend_position_x = 0
            else:
                # Use calculated values
                bend_location = new_result['bend_location']
                bend_magnitude = new_result['bend_magnitude']
                bend_position_y = new_result['bend_position'][0]
                bend_position_x = new_result['bend_position'][1]
            
            final_result = {
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': row['angle_degrees'],  # Use smoothed angle
                'bend_location': bend_location,
                'bend_magnitude': bend_magnitude,
                'bend_position_y': bend_position_y,
                'bend_position_x': bend_position_x,
                'head_mag': new_result['head_mag'],
                'body_mag': new_result['body_mag'],
                'is_noise_peak': row['is_noise_peak'],
                'peak_deviation': row['peak_deviation'],
                'window_size_used': row['window_size_used'],
                'is_straight': is_straight,
                'error': new_result.get('error', None)
            }
            
            final_data.append(final_result)
    
    final_df = pd.DataFrame(final_data)
    
    # Final validation check for correct values
    straight_count = final_df[final_df['is_straight'] == True].shape[0]
    non_straight_with_bends = final_df[(final_df['is_straight'] == False) & (final_df['bend_location'] > 0)].shape[0]
    print(f"Final validation - Straight frames: {straight_count}, Non-straight with bend values: {non_straight_with_bends}")
    
    # Add warning column for interpolated/default values
    final_df['has_warning'] = final_df['error'].notna()
    
    # Log summary of warnings
    warning_count = final_df['has_warning'].sum()
    if warning_count > 0:
        print(f"\nWarning Summary:")
        print(f"Total frames with warnings: {warning_count}")
        print("\nSample of warnings:")
        for error in final_df[final_df['has_warning']]['error'].unique()[:5]:
            count = final_df[final_df['error'] == error].shape[0]
            print(f"- {error}: {count} frames")
    
    print(f"✅ Calculated angles for {len(final_df)} measurements using original algorithm")
    if len(final_df) > 0:
        print(f"📊 Angle range: {final_df['angle_degrees'].min():.1f}° to {final_df['angle_degrees'].max():.1f}°")
        print(f"📊 Mean angle: {final_df['angle_degrees'].mean():.1f}°")
        
        # Count straight vs bent
        straight_count = final_df[final_df['is_straight'] == True].shape[0]
        bent_count = len(final_df) - straight_count
        print(f"📊 Straight frames: {straight_count}, Bent frames: {bent_count}")
        
        # Report smoothing statistics
        noise_peaks = final_df[final_df['is_noise_peak'] == True].shape[0]
        if noise_peaks > 0:
            print(f"📊 Noise peaks smoothed: {noise_peaks}")
    
    return final_df 