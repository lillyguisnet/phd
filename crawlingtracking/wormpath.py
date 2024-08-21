import numpy as np
from scipy.ndimage import center_of_mass
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from matplotlib.patches import Patch
from scipy.signal import medfilt

with open('propagation_fullframe.pkl', 'rb') as file:
    video_segments = pickle.load(file)

with open('wormshape_results.pkl', 'rb') as file:
    cropped_analysis = pickle.load(file)   

def get_centroid(mask):
    # Ensure the mask is 2D
    if mask.ndim == 3:
        mask = mask.squeeze()
    # Calculate centroid
    cy, cx = center_of_mass(mask)
    return (int(cx), int(cy))

# Process all frames
centroids = {}
for frame, segmentation in video_segments.items():
    print(frame)
    mask = segmentation[1]  # Assuming the mask is always under key 1
    centroids[frame] = get_centroid(mask)


def smooth_path(centroids, window_length=11, poly_order=3):
    frames = sorted(centroids.keys())
    x_coords = [centroids[f][0] for f in frames]
    y_coords = [centroids[f][1] for f in frames]
    
    x_smooth = savgol_filter(x_coords, window_length, poly_order)
    y_smooth = savgol_filter(y_coords, window_length, poly_order)
    
    return {f: (x, y) for f, x, y in zip(frames, x_smooth, y_smooth)}

def calculate_path_metrics(centroids):
    sorted_frames = sorted(centroids.keys())
    coordinates = np.array([centroids[frame] for frame in sorted_frames])
    
    # Calculate total distance traveled
    distances = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
    total_distance = np.sum(distances)
    
    # Calculate maximum distance from starting point
    start_point = coordinates[0]
    distances_from_start = np.sqrt(np.sum((coordinates - start_point)**2, axis=1))
    max_distance_from_start = np.max(distances_from_start)
    
    return total_distance, max_distance_from_start

# Print first few centroids
print(dict(list(centroids.items())[:5]))
# Use this in place of padded_moving_average
smooth_centroids = smooth_path(centroids)
print(dict(list(smooth_centroids.items())[:5]))
# Calculate metrics for both original and smoothed paths
original_total_distance, original_max_distance = calculate_path_metrics(centroids)
smooth_total_distance, smooth_max_distance = calculate_path_metrics(smooth_centroids)

print(f"Original path:")
print(f"Total distance traveled: {original_total_distance:.2f} pixels")
print(f"Maximum distance from start: {original_max_distance:.2f} pixels")
print(f"\nSmoothed path:")
print(f"Total distance traveled: {smooth_total_distance:.2f} pixels")
print(f"Maximum distance from start: {smooth_max_distance:.2f} pixels")

def plot_paths_with_time_gradient(original, filtered):
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Create a custom colormap for 6 segments
    colors = ['purple', 'blue', 'cyan', 'green', 'yellow', 'red']
    n_bins = 6  # 6 color segments
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Sort frames and get coordinates
    sorted_frames = sorted(original.keys())
    x_orig, y_orig = zip(*[original[frame] for frame in sorted_frames])
    x_filt, y_filt = zip(*[filtered[frame] for frame in sorted_frames])
    
    # Plot original path with color gradient
    for i in range(n_bins):
        start = i * len(sorted_frames) // n_bins
        end = (i + 1) * len(sorted_frames) // n_bins
        ax.plot(x_orig[start:end], y_orig[start:end], color=cmap(i/n_bins), alpha=0.5, linewidth=2)
    
    # Plot filtered path with color gradient
    for i in range(n_bins):
        start = i * len(sorted_frames) // n_bins
        end = (i + 1) * len(sorted_frames) // n_bins
        ax.plot(x_filt[start:end], y_filt[start:end], color=cmap(i/n_bins), alpha=0.9, linewidth=2)
    
    ax.set_title("Worm Path: Original vs Filtered (Padded Moving Average)\nColor shows time progression")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    
    # Add a colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=600))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Frame number')
    cbar.set_ticks([50 + i*100 for i in range(6)])  # Set ticks at the middle of each color segment
    cbar.set_ticklabels([f'{i*100}-{(i+1)*100}' for i in range(6)])
    
    # Add legend
    ax.plot([], [], color='black', alpha=0.5, linewidth=2, label='Original')
    ax.plot([], [], color='black', alpha=0.5, linewidth=2, label='Filtered')
    ax.legend()

     # Highlight start and end points
    ax.plot(x_orig[0], y_orig[0], 'go', markersize=10, label='Start')
    ax.plot(x_orig[-1], y_orig[-1], 'ro', markersize=10, label='End')
    
    # Highlight furthest point
    distances_from_start = [euclidean(original[sorted_frames[0]], original[frame]) for frame in sorted_frames]
    furthest_frame = sorted_frames[np.argmax(distances_from_start)]
    furthest_point = original[furthest_frame]
    ax.plot(furthest_point[0], furthest_point[1], 'yo', markersize=10, label='Furthest Point')
    
    # Add metrics to the plot
    ax.text(0.05, 0.95, f"Original Total Distance: {original_total_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.92, f"Original Max Distance: {original_max_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.89, f"Smooth Total Distance: {smooth_total_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.86, f"Smooth Max Distance: {smooth_max_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('plotpath.png', dpi=300, bbox_inches='tight')
    plt.close()

# Assuming you have already calculated 'centroids' and 'filtered_centroids'
plot_paths_with_time_gradient(centroids, smooth_centroids)


def calculate_velocity(smooth_centroids, fps=10):
    frames = sorted(smooth_centroids.keys())
    positions = np.array([smooth_centroids[f] for f in frames])
    
    # Calculate velocity using central differences
    velocity = np.gradient(positions, axis=0) * fps
    
    # Smooth the velocity
    v_x = savgol_filter(velocity[:, 0], window_length=11, polyorder=3)
    v_y = savgol_filter(velocity[:, 1], window_length=11, polyorder=3)
    
    return {f: (vx, vy) for f, vx, vy in zip(frames, v_x, v_y)}


def align_data(full_frame_centroids, cropped_analysis):
    """Align the full frame centroids with the cropped video analysis."""
    aligned_data = {}
    for frame in full_frame_centroids.keys():
        if frame in cropped_analysis['frames']:
            aligned_data[frame] = {
                'centroid': full_frame_centroids[frame],
                'head_bend': cropped_analysis['smoothed_head_bends'][cropped_analysis['frames'].index(frame)],
                'smooth_points': cropped_analysis['smooth_points'][cropped_analysis['frames'].index(frame)]
            }
    return aligned_data

def calculate_orientation_vector(smooth_points):
    """Calculate the orientation vector of the worm using the head and the point at the first quarter."""
    head = smooth_points[0]
    segment_point = smooth_points[len(smooth_points) // 10]  # Point at the first quarter of the worm's body
    return np.array(head) - np.array(segment_point)


def calculate_movement_features(velocities, orientations, window_size=5):
    # Compute speed (magnitude of velocity)
    speed = np.linalg.norm(velocities, axis=1)
    
    # Normalize the vectors to avoid issues with magnitude
    velocities_norm = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
    orientations_norm = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    
    # Compute the dot product and get the angles in degrees
    dot_products = np.einsum('ij,ij->i', velocities_norm, orientations_norm)
    angles = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))  # Clip to handle any rounding errors
    
    # Apply median filtering to smooth the results
    smooth_speed = medfilt(speed, kernel_size=window_size)
    smooth_angles = medfilt(angles, kernel_size=window_size)
    
    return smooth_speed, smooth_angles

def classify_movement_window(speeds, angles, speed_threshold=0.4, angle_threshold=130):
    """Classify movement based on windowed features."""
    if np.mean(speeds) < speed_threshold:
        return 'stationary'
    elif np.mean(np.abs(angles)) < angle_threshold:
        return 'forward'
    else:
        return 'backward'
    



def analyze_worm_movement(full_frame_centroids, cropped_analysis, fps=10, window_size=5):
    smooth_centroids = smooth_path(full_frame_centroids)
    velocities = calculate_velocity(smooth_centroids, fps)
    aligned_data = align_data(smooth_centroids, cropped_analysis)
    
    frames = sorted(aligned_data.keys())
    velocity_vectors = np.array([velocities[f] for f in frames])
    orientation_vectors = np.array([calculate_orientation_vector(aligned_data[f]['smooth_points']) for f in frames])
    
    speeds, angles = calculate_movement_features(velocity_vectors, orientation_vectors, window_size)
    
    movement_classification = {}
    for i, frame in enumerate(frames):
        start = max(0, i - window_size // 2)
        end = min(len(frames), i + window_size // 2 + 1)
        movement_classification[frame] = classify_movement_window(
            speeds[start:end], angles[start:end]
        )
    
    forward_frames = sum(1 for v in movement_classification.values() if v == 'forward')
    backward_frames = sum(1 for v in movement_classification.values() if v == 'backward')
    stationary_frames = sum(1 for v in movement_classification.values() if v == 'stationary')
    
    # Calculate additional metrics
    total_frames = len(aligned_data)
    positions = np.array([aligned_data[f]['centroid'] for f in sorted(aligned_data.keys())])
    total_distance = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    avg_speed = total_distance / total_frames
    
    # Calculate sinuosity (total distance / straight-line distance)
    start_point = positions[0]
    end_point = positions[-1]
    straight_line_distance = np.linalg.norm(end_point - start_point)
    sinuosity = total_distance / straight_line_distance if straight_line_distance > 0 else 0
    
    # Calculate average velocity and acceleration
    velocity_vectors = np.array([velocities[f] for f in sorted(velocities.keys())])
    avg_velocity = np.mean(velocity_vectors, axis=0)
    acceleration = np.gradient(velocity_vectors, axis=0) * fps
    avg_acceleration = np.mean(acceleration, axis=0)
    
    return {
        'forward_frames': forward_frames,
        'backward_frames': backward_frames,
        'stationary_frames': stationary_frames,
        'total_frames': total_frames,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'sinuosity': sinuosity,
        'avg_velocity': avg_velocity,
        'avg_acceleration': avg_acceleration,
        'movement_classification': movement_classification,
        'smooth_centroids': smooth_centroids,
        'velocities': velocities
    }

# Assuming you have 'centroids' from full frame analysis and 'cropped_analysis' from cropped video analysis
results = analyze_worm_movement(centroids, cropped_analysis)

print(f"Forward frames: {results['forward_frames']}")
print(f"Backward frames: {results['backward_frames']}")
print(f"Stationary frames: {results['stationary_frames']}")
print(f"Total distance: {results['total_distance']:.2f} pixels")
print(f"Average speed: {results['avg_speed']:.2f} pixels/frame")
print(f"Sinuosity: {results['sinuosity']:.2f}")
print(f"Average velocity: ({results['avg_velocity'][0]:.2f}, {results['avg_velocity'][1]:.2f}) pixels/frame")
print(f"Average acceleration: ({results['avg_acceleration'][0]:.2f}, {results['avg_acceleration'][1]:.2f}) pixels/frame^2")

# Visualize the movement classification
frames = sorted(results['movement_classification'].keys())
classifications = [results['movement_classification'][f] for f in frames]


def plot_worm_path_with_metrics(results):
    # Extract necessary data
    smooth_centroids = results['smooth_centroids']
    movement_classification = results['movement_classification']
    velocities = results['velocities']

    # Prepare data for plotting
    frames = sorted(smooth_centroids.keys())
    x, y = zip(*[smooth_centroids[f] for f in frames])
    colors = []
    for f in frames:
        if f in movement_classification:
            if movement_classification[f] == 'forward':
                colors.append('green')
            elif movement_classification[f] == 'backward':
                colors.append('red')
            else:
                colors.append('blue')
        else:
            colors.append('gray')

    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1]})

    # Plot the worm's path
    scatter = ax1.scatter(x, y, c=colors, s=5)
    ax1.plot(x, y, color='gray', alpha=0.5, linewidth=1)
    ax1.set_title("Worm Path with Movement Classification")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.invert_yaxis()

    # Add legend
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label='Forward'),
        Patch(facecolor='red', edgecolor='red', label='Backward'),
        Patch(facecolor='blue', edgecolor='blue', label='Stationary')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Highlight start and end points
    ax1.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax1.plot(x[-1], y[-1], 'ro', markersize=10, label='End')

    # Add metrics to the plot
    metrics_text = (
        f"Total Frames: {results['total_frames']}\n"
        f"Forward Frames: {results['forward_frames']}\n"
        f"Backward Frames: {results['backward_frames']}\n"
        f"Stationary Frames: {results['stationary_frames']}\n"
        f"Total Distance: {results['total_distance']:.2f} px\n"
        f"Average Speed: {results['avg_speed']:.2f} px/frame\n"
        f"Sinuosity: {results['sinuosity']:.2f}"
    )
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # Plot velocity over time
    v_x = [velocities[f][0] for f in frames]
    v_y = [velocities[f][1] for f in frames]
    speed = np.sqrt(np.array(v_x)**2 + np.array(v_y)**2)
    ax2.plot(frames, speed, label='Speed')
    ax2.set_title("Worm Speed Over Time")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Speed (px/frame)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('plotspeed.png', dpi=300, bbox_inches='tight')
    plt.close()

# Use the function
plot_worm_path_with_metrics(results)






def classify_movement_midbody(smooth_points, head_positions, window_size=5):
    num_frames = len(smooth_points)
    classifications = []

    for i in range(1, num_frames):
        prev_head = head_positions[i-1]
        curr_head = head_positions[i]
        
        prev_midbody = smooth_points[i-1][len(smooth_points[i-1])//2]
        curr_midbody = smooth_points[i][len(smooth_points[i])//2]
        
        head_displacement = curr_head - prev_head
        midbody_displacement = curr_midbody - prev_midbody
        
        # Calculate dot product
        dot_product = np.dot(head_displacement, midbody_displacement)
        
        # Classify based on dot product
        if np.linalg.norm(head_displacement) < 0.5:  # Adjust threshold as needed
            classifications.append('stationary')
        elif dot_product > 0:
            classifications.append('forward')
        else:
            classifications.append('backward')
    
    # Add classification for the first frame (assume same as second frame)
    classifications.insert(0, classifications[0])
    
    # Apply median filter to smooth classifications
    smoothed_classifications = medfilt(
        [['stationary', 'forward', 'backward'].index(c) for c in classifications], 
        kernel_size=window_size
    )
    
    return [['stationary', 'forward', 'backward'][i] for i in smoothed_classifications]



# Extract head positions (assuming the first point of each skeleton is the head)
head_positions = np.array([points[0] for points in cropped_analysis['smooth_points']])
# Classify movement
movement_classifications = classify_movement_midbody(cropped_analysis['smooth_points'], head_positions)
# Add classifications to results
cropped_analysis['movement_classifications'] = movement_classifications

def plot_movement_classification(results):
    frames = results['frames']
    classifications = results['movement_classifications']
    
    plt.figure(figsize=(12, 6))
    plt.plot(frames, [{'forward': 1, 'backward': -1, 'stationary': 0}[c] for c in classifications])
    plt.yticks([-1, 0, 1], ['Backward', 'Stationary', 'Forward'])
    plt.xlabel('Frame')
    plt.ylabel('Movement Classification')
    plt.title('Worm Movement Classification Over Time')
    plt.grid(True)
    plt.show()
    plt.savefig('plotspeedmidb.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_movement_classification(cropped_analysis)