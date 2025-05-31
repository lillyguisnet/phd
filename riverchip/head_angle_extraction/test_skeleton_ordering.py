"""
Test script to verify the new skeleton ordering functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

# Import the functions we need to test
def order_skeleton_points_by_path(skeleton):
    """
    Simplified version for testing - copy the function here to avoid import issues
    """
    from scipy.spatial.distance import cdist
    
    # Get all skeleton points
    points = np.column_stack(np.where(skeleton))
    if len(points) == 0:
        return points, False
    
    # If only one or two points, return as is
    if len(points) <= 2:
        return points, True
    
    try:
        # Build adjacency information by finding neighbors for each point
        distances = cdist(points, points)
        adjacency = distances <= np.sqrt(2)
        np.fill_diagonal(adjacency, False)
        
        # Count neighbors for each point
        neighbor_counts = np.sum(adjacency, axis=1)
        
        # Find endpoints (points with only 1 neighbor)
        endpoints = np.where(neighbor_counts == 1)[0]
        
        # Choose starting point
        if len(endpoints) == 0:
            distances_between_points = cdist(points, points)
            max_dist_idx = np.unravel_index(np.argmax(distances_between_points), distances_between_points.shape)
            start_idx = max_dist_idx[0]
        elif len(endpoints) == 1:
            start_idx = endpoints[0]
        else:
            endpoint_distances = cdist(points[endpoints], points[endpoints])
            max_dist_idx = np.unravel_index(np.argmax(endpoint_distances), endpoint_distances.shape)
            start_idx = endpoints[max_dist_idx[0]]
        
        # Follow the path
        ordered_indices = []
        current_idx = start_idx
        visited = set()
        
        while current_idx is not None and current_idx not in visited:
            ordered_indices.append(current_idx)
            visited.add(current_idx)
            
            neighbors = np.where(adjacency[current_idx])[0]
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if len(unvisited_neighbors) == 0:
                break
            elif len(unvisited_neighbors) == 1:
                current_idx = unvisited_neighbors[0]
            else:
                # Choose the neighbor that continues the path most smoothly
                if len(ordered_indices) >= 2:
                    prev_point = points[ordered_indices[-2]]
                    curr_point = points[current_idx]
                    current_direction = curr_point - prev_point
                    current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
                    
                    best_neighbor = None
                    best_alignment = -2
                    
                    for neighbor_idx in unvisited_neighbors:
                        neighbor_point = points[neighbor_idx]
                        neighbor_direction = neighbor_point - curr_point
                        neighbor_direction = neighbor_direction / (np.linalg.norm(neighbor_direction) + 1e-10)
                        
                        alignment = np.dot(current_direction, neighbor_direction)
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_neighbor = neighbor_idx
                    
                    current_idx = best_neighbor
                else:
                    current_idx = unvisited_neighbors[0]
        
        # Add any remaining points
        remaining_points = set(range(len(points))) - visited
        if remaining_points:
            if ordered_indices:
                last_point = points[ordered_indices[-1]]
                remaining_indices = list(remaining_points)
                remaining_distances = [np.linalg.norm(points[idx] - last_point) for idx in remaining_indices]
                sorted_remaining = [x for _, x in sorted(zip(remaining_distances, remaining_indices))]
                ordered_indices.extend(sorted_remaining)
            else:
                ordered_indices.extend(list(remaining_points))
        
        ordered_points = points[ordered_indices]
        return ordered_points, True
        
    except Exception as e:
        print(f"Path-based ordering failed: {e}")
        return points[np.argsort(points[:, 0])], False

def determine_head_end(ordered_points, skeleton_shape):
    """Determine which end is the head."""
    if len(ordered_points) < 2:
        return True
    
    start_point = ordered_points[0]
    end_point = ordered_points[-1]
    
    start_y, start_x = start_point
    end_y, end_x = end_point
    
    # Head is usually closer to the top
    if start_y < end_y:
        return True
    elif end_y < start_y:
        return False
    
    # Tiebreaker
    return start_x < end_x

def create_test_skeleton():
    """Create a simple curved skeleton for testing."""
    skeleton = np.zeros((100, 100), dtype=bool)
    
    # Create a curved path
    for i in range(80):
        y = 10 + i
        x = 10 + int(20 * np.sin(i * 0.1))
        if 0 <= y < 100 and 0 <= x < 100:
            skeleton[y, x] = True
    
    return skeleton

def test_ordering():
    """Test the skeleton ordering function."""
    print("🧪 Testing skeleton ordering...")
    
    skeleton = create_test_skeleton()
    ordered_points, success = order_skeleton_points_by_path(skeleton)
    
    print(f"Ordering successful: {success}")
    print(f"Number of points: {len(ordered_points)}")
    
    if len(ordered_points) > 0:
        print(f"First point: {ordered_points[0]}")
        print(f"Last point: {ordered_points[-1]}")
        
        head_first = determine_head_end(ordered_points, skeleton.shape)
        print(f"Head at beginning: {head_first}")
        
        # Compare with coordinate-based ordering
        coord_ordered = np.column_stack(np.where(skeleton))
        coord_ordered = coord_ordered[np.argsort(coord_ordered[:, 0])]
        
        print(f"\nComparison:")
        print(f"Path-based first point: {ordered_points[0]}")
        print(f"Coordinate-based first point: {coord_ordered[0]}")
        print(f"Path-based last point: {ordered_points[-1]}")
        print(f"Coordinate-based last point: {coord_ordered[-1]}")
        
        print("✅ Test completed!")
    else:
        print("❌ No points found in skeleton")

if __name__ == "__main__":
    test_ordering() 