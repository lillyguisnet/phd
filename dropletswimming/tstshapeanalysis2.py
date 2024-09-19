import pickle
import numpy as np
from scipy import interpolate, signal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage import morphology, graph, draw
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import stft
from scipy.signal import welch, find_peaks
from scipy.spatial.distance import cdist
import os
from collections import Counter
from collections import defaultdict


with open('tstswimcrop.pkl', 'rb') as file:
    hd_video_segments = pickle.load(file)


""" with open('/home/maxime/prg/phd/wormshape_hdresults.pkl', 'rb') as file:
    tst = pickle.load(file) """

def smooth_metric(data, window_length=11, poly_order=3):
    return savgol_filter(data, window_length, poly_order)

def clean_mask(mask):
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = sizes.argmax() + 1
    cleaned_mask = (labeled == largest_label)
    return cleaned_mask

def get_skeleton(mask):
    return morphology.skeletonize(mask)

def find_endpoints_and_junctions(coords):
    if isinstance(coords, np.ndarray) and coords.dtype == bool:
        coords = np.argwhere(coords)
    # Create a dictionary to count neighbors for each point
    neighbor_count = defaultdict(int)
    
    # Function to check if two points are neighbors
    def are_neighbors(p1, p2):
        return np.all(np.abs(p1 - p2) <= 1) and not np.all(p1 == p2)

    # Count neighbors for each point
    for i, p1 in enumerate(coords):
        for j, p2 in enumerate(coords):
            if i != j and are_neighbors(p1, p2):
                neighbor_count[tuple(p1)] += 1

    endpoints = []
    junctions = []

    # Classify points based on neighbor count
    for point in coords:
        point_tuple = tuple(point)
        if neighbor_count[point_tuple] == 1:
            endpoints.append(point)
        elif neighbor_count[point_tuple] > 2:
            junctions.append(point)

    return np.array(endpoints), np.array(junctions)


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def find_furthest_endpoints_along_skeleton(skeleton):
    # Find all endpoints
    endpoints, _ = find_endpoints_and_junctions(skeleton)
    
    if len(endpoints) <= 2:
        return None  # Not enough endpoints
    
    max_distance = 0
    furthest_pair = None
    
    # Check all pairs of endpoints
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            
            cost_array = np.where(skeleton, 1, np.inf)
            start = tuple(endpoints[i])
            end = tuple(endpoints[j]) 
            path_indices, cost = graph.route_through_array(cost_array, start, end)

            path = np.array(path_indices)

            distance = len(path)  # The length of the path is the distance along the skeleton
            
            if distance > max_distance:
                max_distance = distance
                furthest_pair = (endpoints[i], endpoints[j])
    
    return furthest_pair

def order_segments(segments):
    ### Order segments so that endpoints are at the beginning and end
    # Convert segments to a set of tuples for easier lookup
    segments_set = set(map(tuple, segments))
    
    # Create a graph representation
    graph = defaultdict(list)
    for x, y in segments_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in segments_set:
                    graph[(x, y)].append(neighbor)
    
    # Find endpoints (points with only one neighbor)
    endpoints = [point for point, neighbors in graph.items() if len(neighbors) == 1]
    
    if len(endpoints) != 2:
        raise ValueError("Expected exactly two endpoints")
    
    # Traverse the graph from one endpoint to the other
    start, end = endpoints
    ordered = [start]
    current = start
    
    while current != end:
        next_point = [p for p in graph[current] if p not in ordered][0]
        ordered.append(next_point)
        current = next_point
    
    return np.array(ordered)



def calculate_orientation_difference(segment1, segment2, p1, p2):
    idx1 = np.where((segment1 == p1).all(axis=1))[0][0]
    idx2 = np.where((segment2 == p2).all(axis=1))[0][0]

    # Use up to 5 points (including the endpoint) in the direction of the segment
    if idx1 == 0:
        points1 = segment1[:3]  # First 5 points if at start
    else:
        points1 = segment1[max(0, idx1-2):idx1+1]  # Up to 5 points ending at idx1
    
    if idx2 == 0:
        points2 = segment2[:3]  # First 5 points if at start
    else:
        points2 = segment2[max(0, idx2-2):idx2+1]  # Up to 5 points ending at idx2
       
    # Calculate direction vectors using the first and last points of our selections
    vec1 = points1[-1] - points1[0]
    vec2 = points2[-1] - points2[0]
    
    dot_product = np.dot(vec1, vec2)
    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    cos_angle = dot_product / norms
    
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def adjust_self_touching_skeleton(skeleton):
    endpoints, junctions = find_endpoints_and_junctions(skeleton)
    
    if len(junctions) > 0:
        # Remove junctions
        for junction in junctions:
            skeleton[junction[0], junction[1]] = 0
        
        # Find all segments
        labeled_skeleton, num_segments = morphology.label(skeleton, connectivity=2, return_num=True)
        segments = [np.argwhere(labeled_skeleton == i) for i in range(1, num_segments+1)]
        
        while len(segments) > 1:
            
            # Order segments so that endpoints are at the beginning and end
            ordered_segments = []
            for seg in (segments):
                ordered_seg = order_segments(seg)
                ordered_segments.append(ordered_seg)
            segments = ordered_segments
            
            # Calculate distances between all pairs of endpoints
            endpoints_segs = []
            for _, segs in enumerate(segments):
                #seg =segs
                endpoints, junc = find_endpoints_and_junctions(segs)
                endpoints_segs.append(endpoints)
            #endpoints_segs, _ = find_endpoints_and_junctions(segments)
            all_endpoints = np.vstack(endpoints_segs)
            #distances = cdist(all_endpoints, all_endpoints,'euclidean')
            distances = np.linalg.norm(all_endpoints[:, None] - all_endpoints, axis=2)
            np.fill_diagonal(distances, np.inf)

            num_segments = len(segments)
            endpoint_to_segment = {}

            # Map each endpoint to its segment
            for seg_idx, segment in enumerate(segments):
                for point in segment:
                    endpoint_to_segment[tuple(point)] = seg_idx

            # Find the closest pair of endpoints that belong to different segments
            connect_p1 = None
            connect_p2 = None
            original_min_distance = np.min(distances)
            original_closest_pair = np.unravel_index(distances.argmin(), distances.shape)
            while connect_p1 is None and connect_p2 is None:
                if np.isinf(distances).all():
                    # If all distances are inf, use the original closest pair
                    i, j = original_closest_pair
                    connect_p1 = all_endpoints[i]
                    connect_p2 = all_endpoints[j]
                    break

                i, j = np.unravel_index(distances.argmin(), distances.shape)
                p1, p2 = all_endpoints[i], all_endpoints[j]

                # Find which segments these endpoints belong to
                seg1_idx = endpoint_to_segment[tuple(p1)]
                seg2_idx = endpoint_to_segment[tuple(p2)]

                if seg1_idx != seg2_idx:
                    angle = calculate_orientation_difference(segments[seg1_idx], segments[seg2_idx], p1, p2)
                    if angle > 120 or angle < 20:
                        connect_p1 = p1
                        connect_p2 = p2
                    else:
                        distances[i, j] = distances[j, i] = np.inf
                else:
                    # If endpoints are from the same segment, set this distance to infinity and continue
                    distances[i, j] = distances[j, i] = np.inf


            # Connect the closest pair of endpoints
            rr, cc = draw.line(p1[0], p1[1], p2[0], p2[1])
            skeleton[rr, cc] = 1

            # Recompute segments
            labeled_skeleton, num_segments = morphology.label(skeleton, connectivity=2, return_num=True)
            segments = [np.argwhere(labeled_skeleton == i) for i in range(1, num_segments+1)]                              
    
    # Ensure exactly two endpoints
    endpoints, _ = find_endpoints_and_junctions(skeleton)
    if len(endpoints) > 2:
        furthest_pair = find_furthest_endpoints_along_skeleton(skeleton)
        if furthest_pair:           
            cost_array = np.where(skeleton, 1, np.inf)
            start = tuple(furthest_pair[0])
            end = tuple(furthest_pair[1]) 
            path_indices, cost = graph.route_through_array(cost_array, start, end)

            path = np.array(path_indices)
            
            return path                      
    
    return np.argwhere(skeleton)  



def gaussian_weighted_curvature(points, window_size, sigma):
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    pad_width = window_size // 2
    padded_points = np.pad(points, ((pad_width, pad_width), (0, 0)), mode='edge')
    
    weights = ndimage.gaussian_filter1d(np.ones(window_size), sigma)
    weights /= np.sum(weights)
    
    curvatures = []
    for i in range(len(points)):
        window = padded_points[i:i+window_size]
        centroid = np.sum(window * weights[:, np.newaxis], axis=0) / np.sum(weights)
        centered = window - centroid
        cov = np.dot(centered.T, centered * weights[:, np.newaxis]) / np.sum(weights)
        eigvals, eigvecs = np.linalg.eig(cov)
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        curvature = eigvals[1] / (eigvals[0] + eigvals[1])
        curvatures.append(curvature)
    
    return np.array(curvatures)

def calculate_swimamplitude(smooth_points):
    # Calculate the centerline (straight line from head to tail)
    centerline = np.array([smooth_points[0], smooth_points[-1]])
    
    # Calculate the vector of the centerline
    centerline_vector = centerline[1] - centerline[0]
    
    # Normalize the centerline vector
    centerline_unit = centerline_vector / np.linalg.norm(centerline_vector)
    
    # Calculate perpendicular distances from each point to the centerline
    perpendicular_distances = []
    for point in smooth_points:
        v = point - centerline[0]
        proj = np.dot(v, centerline_unit)
        proj_point = centerline[0] + proj * centerline_unit
        distance = np.linalg.norm(point - proj_point)
        perpendicular_distances.append(distance)
    
    # Maximum amplitude is half of the maximum perpendicular distance
    max_amplitude = max(perpendicular_distances)
    avg_amplitude = np.mean(perpendicular_distances)
    
    return max_amplitude, avg_amplitude



def classify_shape(smooth_points, threshold=8, c_shape_ratio=0.98, epsilon=1e-10):
    centerline_start = smooth_points[0]
    centerline_end = smooth_points[-1]
    centerline_vector = centerline_end - centerline_start
    
    centerline_unit = centerline_vector / np.linalg.norm(centerline_vector)
    
    perpendicular_distances = []
    sides = []
    for point in smooth_points:
        v = point - centerline_start
        proj = np.dot(v, centerline_unit)
        proj_point = centerline_start + proj * centerline_unit
        distance = np.linalg.norm(point - proj_point)
        perpendicular_distances.append(distance)
        
        cross_product = np.cross(centerline_vector, v)
        # Use a small epsilon value to handle near-zero cases
        if abs(cross_product) < epsilon:
            sides.append(0)  # Point is very close to the centerline
        else:
            sides.append(1 if cross_product > 0 else -1)
    
    unique_sides = np.unique(sides)
    max_distance = max(perpendicular_distances)
    
    # Count positive and negative sides separately
    positive_count = sum(1 for s in sides if s > 0)
    negative_count = sum(1 for s in sides if s < 0)
    total_points = len(sides)
    
    dominant_side_ratio = max(positive_count, negative_count) / total_points
    
    if dominant_side_ratio >= c_shape_ratio and max_distance > threshold:
        return "C-shape"
    elif max_distance <= threshold:
        return "Straight"
    elif len(unique_sides) > 1:
        return "S-shape"
    else:
        return "Unknown"


def analyze_shape(skeleton, frame_num):
    longest_path = adjust_self_touching_skeleton(skeleton)
    
    longest_path = order_segments(longest_path)
    t = np.arange(len(longest_path))
    x, y = longest_path[:, 1], longest_path[:, 0]
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.linspace(0, 1, num=100)
    smooth_points = np.column_stack(interpolate.splev(unew, tck))
    
    # Curvature calculation
    window_size, sigma = 50, 10
    curvature = gaussian_weighted_curvature(smooth_points, window_size, sigma)

    # Amplitude calculation (using swim centerline distances)
    max_amplitude, avg_amplitude = calculate_swimamplitude(smooth_points)

    # Worm length
    worm_length = np.sum(np.sqrt(np.sum(np.diff(smooth_points, axis=0)**2, axis=1)))

    shape = classify_shape(smooth_points)
    # Calculate wavelength based on the shape
    if shape == "C-shape":
        wavelength = worm_length * 2  # One full wave is twice the worm length
    elif shape == "Straight":
        wavelength = worm_length * 4  # Assume a very long wavelength
    elif shape == "S-shape":
        peaks, _ = find_peaks(curvature)
        if len(peaks) > 1:
            wavelengths = np.diff(peaks)
            avg_wavelength = np.mean(wavelengths) * 2  # multiply by 2 for full wave
            wavelength = avg_wavelength
        else:
            wavelength = 0  # or some default value
    else:
        wavelength = 0


    wave_number = worm_length / wavelength #Number of waves along the worm body
    normalized_wavelength = wavelength / worm_length #Wavelength as proportion of worm length

    # Frequency analysis (spatial)
    spatial_freq = np.abs(np.fft.fft(curvature))
    dominant_spatial_freq = np.abs(np.fft.fftfreq(len(curvature))[np.argmax(spatial_freq[1:]) + 1])   

    return {
        'frame': frame_num,
        'shape': shape,
        'smooth_points': smooth_points,
        'curvature': curvature,
        'max_amplitude': max_amplitude,
        'avg_amplitude': avg_amplitude,
        'wavelength': wavelength,
        'worm_length': worm_length,
        'wave_number': wave_number,
        'normalized_wavelength': normalized_wavelength,
        'dominant_spatial_freq': dominant_spatial_freq,
    }


def analyze_video(segmentation_dict, fps=10, window_size=5, overlap=2.5):
    # Initialize result containers
    frames = []
    shape = []
    smooth_points = []
    curvatures = []
    max_amplitudes = []
    avg_amplitudes = []
    wavelengths = []
    worm_lengths = []
    wave_numbers = []
    normalized_wavelengths = []
    dominant_spatial_freqs = []
    masks = []
    
    # Get shape info, per frame
    for frame_num, frame_data in segmentation_dict.items():
        print("Frame: " + str(frame_num))
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)
        
        frame_results = analyze_shape(skeleton, frame_num) #per frame
        frames.append(frame_results['frame'])
        shape.append(frame_results['shape'])
        smooth_points.append(frame_results['smooth_points'])
        curvatures.append(frame_results['curvature'])
        max_amplitudes.append(frame_results['max_amplitude'])
        avg_amplitudes.append(frame_results['avg_amplitude'])
        wavelengths.append(frame_results['wavelength'])
        worm_lengths.append(frame_results['worm_length'])
        wave_numbers.append(frame_results['wave_number'])
        normalized_wavelengths.append(frame_results['normalized_wavelength'])
        dominant_spatial_freqs.append(frame_results['dominant_spatial_freq'])
        masks.append(cleaned_mask)

    # Temporal frequency analysis
    curvature_1d = np.array([np.mean(c) for c in curvatures])
    curvature_1d = (curvature_1d - np.mean(curvature_1d)) / np.std(curvature_1d)
    
    nperseg = int(window_size * fps)
    noverlap = int(overlap * fps)
    
    f, psd = welch(curvature_1d, fs=fps, nperseg=nperseg, noverlap=noverlap)
    
    dominant_freqs = []
    time_points = []
    for i in range(0, len(curvature_1d) - nperseg, nperseg - noverlap):
        segment = curvature_1d[i:i+nperseg]
        f_segment, psd_segment = welch(segment, fs=fps, nperseg=nperseg, noverlap=noverlap)
        peaks, _ = find_peaks(psd_segment, height=np.max(psd_segment) * 0.1)
        if len(peaks) > 0:
            dominant_freq_idx = peaks[np.argmax(psd_segment[peaks])]
            dominant_freqs.append(f_segment[dominant_freq_idx])
        else:
            dominant_freqs.append(0)
        time_points.append(i / fps)
    
    # Interpolate dominant temporal frequencies for all frames
    frame_numbers = np.arange(len(frames))
    interpolated_freqs = np.interp(frame_numbers / fps, time_points, dominant_freqs)
    
    # Apply smoothing
    smoothed_max_amplitudes = smooth_metric(max_amplitudes)
    smoothed_avg_amplitudes = smooth_metric(avg_amplitudes)
    smoothed_wavelengths = smooth_metric(wavelengths)
    smoothed_worm_lengths = smooth_metric(worm_lengths)
    smoothed_wave_numbers = smooth_metric(wave_numbers)
    smoothed_normalized_wavelengths = smooth_metric(normalized_wavelengths)
    
    # Prepare final results dictionary
    final_results = {
        'frames': frames,
        'shape': shape,
        'smooth_points': smooth_points,
        'curvatures': curvatures,
        'max_amplitudes': max_amplitudes,
        'avg_amplitudes': avg_amplitudes,
        'wavelengths': wavelengths,
        'worm_lengths': worm_lengths,
        'wave_numbers': wave_numbers,
        'normalized_wavelengths': normalized_wavelengths,
        'dominant_spatial_freqs': dominant_spatial_freqs,
        'smoothed_max_amplitudes': smoothed_max_amplitudes,
        'smoothed_avg_amplitudes': smoothed_avg_amplitudes,
        'smoothed_wavelengths': smoothed_wavelengths,
        'smoothed_worm_lengths': smoothed_worm_lengths,
        'smoothed_wave_numbers': smoothed_wave_numbers,
        'smoothed_normalized_wavelengths': smoothed_normalized_wavelengths,
        'interpolated_freqs': interpolated_freqs,
        'f': f,
        'psd': psd,
        'fps': fps,
        'curvature_time_series': curvature_1d
    }

    final_results['masks'] = masks
    
    return final_results



swim_shapeanalysis = analyze_video(hd_video_segments)

shape_counts = Counter(swim_shapeanalysis['shape'])
shape_counts


with open('swim_shapeanalysis.pkl', 'wb') as file:
    pickle.dump(swim_shapeanalysis, file)


image_array = np.uint8(skeleton * 255)
image = Image.fromarray(image_array)
image.save('skeleton.png')
from PIL import Image

def find_endpoints_and_junctions(skeleton):
    points = np.transpose(np.nonzero(skeleton))
    neighbors = np.sum([skeleton[p[0]-1:p[0]+2, p[1]-1:p[1]+2] for p in points], axis=(1,2))
    endpoints = points[neighbors == 2]
    junctions = points[neighbors > 3]
    return endpoints, junctions

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def find_segment_endpoints(segment):
    distances = cdist(segment, segment)
    i, j = np.unravel_index(distances.argmax(), distances.shape)
    return segment[i], segment[j]

def find_furthest_endpoints_along_skeleton(skeleton):
    # Find all endpoints
    endpoints, _ = find_endpoints_and_junctions(skeleton)
    
    if len(endpoints) < 2:
        return None  # Not enough endpoints
    
    # Create a graph from the skeleton
    g = graph.pixel_graph(skeleton)
    
    max_distance = 0
    furthest_pair = None
    
    # Check all pairs of endpoints
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            path = graph.shortest_path(g, tuple(endpoints[i]), tuple(endpoints[j]))
            distance = len(path)  # The length of the path is the distance along the skeleton
            
            if distance > max_distance:
                max_distance = distance
                furthest_pair = (endpoints[i], endpoints[j])
    
    return furthest_pair

def adjust_self_touching_skeleton(skeleton):
    endpoints, junctions = find_endpoints_and_junctions(skeleton)
    
    if len(junctions) > 0:
        # Remove junctions
        for junction in junctions:
            skeleton[junction[0], junction[1]] = 0
        
        # Find all segments
        labeled_skeleton, num_segments = morphology.label(skeleton, connectivity=2, return_num=True)
        segments = [np.argwhere(labeled_skeleton == i) for i in range(1, num_segments+1)]
        
        while len(segments) > 1:
            # Find endpoints of all segments
            segment_endpoints = [find_segment_endpoints(seg) for seg in segments]
            
            # Calculate distances between all pairs of endpoints
            all_endpoints = np.vstack(segment_endpoints)
            distances = cdist(all_endpoints, all_endpoints)
            np.fill_diagonal(distances, np.inf)
            
            connected = False
            while not connected:
                # Find the closest pair of endpoints
                i, j = np.unravel_index(distances.argmin(), distances.shape)
                p1, p2 = all_endpoints[i], all_endpoints[j]
                
                # Find which segments these endpoints belong to
                seg1_idx = i // 2
                seg2_idx = j // 2
                
                if seg1_idx != seg2_idx:  # Ensure we're connecting different segments
                    # Check the angle
                    seg1, seg2 = segments[seg1_idx], segments[seg2_idx]
                    idx1 = np.where((seg1 == p1).all(axis=1))[0][0]
                    idx2 = np.where((seg2 == p2).all(axis=1))[0][0]
                    
                    if idx1 in [0, len(seg1)-1] and idx2 in [0, len(seg2)-1]:
                        # Ensure we have enough points to calculate angle
                        if len(seg1) > 2 and len(seg2) > 2:
                            p1_neighbor = seg1[1] if idx1 == 0 else seg1[-2]
                            p2_neighbor = seg2[1] if idx2 == 0 else seg2[-2]
                            angle = calculate_angle(p1_neighbor, p1, p2_neighbor)
                            
                            if angle >= 90:
                                # Accept this connection
                                rr, cc = draw.line(p1[0], p1[1], p2[0], p2[1])
                                skeleton[rr, cc] = 1
                                
                                # Merge connected segments
                                merged_segment = np.vstack((seg1, seg2))
                                segments = [seg for k, seg in enumerate(segments) if k not in [seg1_idx, seg2_idx]]
                                segments.append(merged_segment)
                                
                                connected = True
                            else:
                                # Reject this connection and try the next closest pair
                                distances[i, j] = distances[j, i] = np.inf
                        else:
                            # If segments are too short, connect them anyway
                            rr, cc = draw.line(p1[0], p1[1], p2[0], p2[1])
                            skeleton[rr, cc] = 1
                            
                            merged_segment = np.vstack((seg1, seg2))
                            segments = [seg for k, seg in enumerate(segments) if k not in [seg1_idx, seg2_idx]]
                            segments.append(merged_segment)
                            
                            connected = True
                    else:
                        # If we're not connecting endpoints, try the next closest pair
                        distances[i, j] = distances[j, i] = np.inf
                else:
                    # If we're trying to connect endpoints of the same segment, try the next closest pair
                    distances[i, j] = distances[j, i] = np.inf
    
    # Ensure exactly two endpoints
    endpoints, _ = find_endpoints_and_junctions(skeleton)
    if len(endpoints) > 2:
        furthest_pair = find_furthest_endpoints_along_skeleton(skeleton)
        if furthest_pair:
            g = graph.pixel_graph(skeleton)
            path = graph.shortest_path(g, tuple(furthest_pair[0]), tuple(furthest_pair[1]))
            
            new_skeleton = np.zeros_like(skeleton)
            for point in path:
                new_skeleton[point] = 1
            
            return new_skeleton
    
    return skeleton


adjusted_skeleton = adjust_self_touching_skeleton(skeleton)


plt.figure(figsize=(10, 6))
plt.plot(smooth_points[:, 0], smooth_points[:, 1], 'b-')
plt.scatter(smooth_points[:, 0], smooth_points[:, 1], c='r', s=20)
plt.title('Smooth Points Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()
plt.savefig('skeleton.png')
plt.close()


def coordinates_to_bool_array(coordinates, shape=(100, 100)):
    # Create an empty boolean array filled with False
    bool_array = np.zeros(shape, dtype=bool)
    
    # Round the coordinates to integers
    rounded_coords = np.round(coordinates).astype(int)
    
    # Clip the coordinates to ensure they're within the array bounds
    rounded_coords = np.clip(rounded_coords, 0, np.array(shape) - 1)
    
    # Set the corresponding pixels to True
    bool_array[rounded_coords[:, 1], rounded_coords[:, 0]] = True
    
    return bool_array

result = coordinates_to_bool_array(longest_path)

from PIL import Image
image_array = np.uint8(result * 255)
image = Image.fromarray(image_array)
image.save('skeleton.png')


def visualize_worm_analysis(results):
    frames = results['frames']
    fps = results['fps']
    times = [frame / fps for frame in frames]

    metrics = ['max_amplitudes', 'avg_amplitudes', 'wavelengths', 'worm_lengths', 
               'dominant_spatial_freqs', 'wave_numbers', 'normalized_wavelengths']
    smoothed_metrics = ['max_amplitudes', 'avg_amplitudes', 'wavelengths', 'worm_lengths', 
                        'wave_numbers', 'normalized_wavelengths']
    
    data = {metric: results[metric] for metric in metrics}
    smoothed_data = {f'smoothed_{metric}': results[f'smoothed_{metric}'] for metric in smoothed_metrics}

    # Create the main figure and subplots
    fig = plt.figure(figsize=(25, 30))
    gs = GridSpec(6, 4, figure=fig)

    # Plot max and avg amplitudes
    ax1 = fig.add_subplot(gs[0, :2])
    #ax1.plot(frames, data['max_amplitudes'], label='Max Amplitude')
    #ax1.plot(frames, data['avg_amplitudes'], label='Avg Amplitude')
    ax1.plot(frames, smoothed_data['smoothed_max_amplitudes'], label='Smoothed Max Amplitude', linestyle='--')
    ax1.plot(frames, smoothed_data['smoothed_avg_amplitudes'], label='Smoothed Avg Amplitude', linestyle='--')
    ax1.set_title('Amplitudes over time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Amplitude')
    ax1.legend()

    # Plot wavelength
    ax2 = fig.add_subplot(gs[3, :2])
    ax2.plot(frames, data['wavelengths'], label='Wavelength')
    ax2.plot(frames, smoothed_data['smoothed_wavelengths'], label='Smoothed Wavelength', linestyle='--')
    ax2.set_title('Wavelength over time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Wavelength')
    ax2.legend()

    # Plot worm length
    ax3 = fig.add_subplot(gs[4, :2])
    ax3.plot(frames, data['worm_lengths'], label='Worm Length')
    ax3.plot(frames, smoothed_data['smoothed_worm_lengths'], label='Smoothed Worm Length', linestyle='--')
    ax3.set_title('Worm Length over time')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Length')
    ax3.legend()

    # Plot spatial frequency
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.plot(frames, data['dominant_spatial_freqs'], label='Spatial Frequency')
    ax4.set_title('Spatial Frequency over time')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # Plot worm shape for a sample frame
    sample_frame_index = len(frames) // 2
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(results['smooth_points'][sample_frame_index][:, 0], results['smooth_points'][sample_frame_index][:, 1])
    ax5.set_title(f'Worm Shape (Frame {frames[sample_frame_index]})')
    ax5.set_aspect('equal', 'box')

    # Plot curvature for the sample frame
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(results['curvatures'][sample_frame_index])
    ax6.set_title(f'Curvature (Frame {frames[sample_frame_index]})')
    ax6.set_xlabel('Position along worm')
    ax6.set_ylabel('Curvature')

    # Plot wave number
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(frames, data['wave_numbers'], label='Wave Number')
    ax7.plot(frames, smoothed_data['smoothed_wave_numbers'], label='Smoothed Wave Number', linestyle='--')
    ax7.set_title('Wave Number over time')
    ax7.set_xlabel('Frame')
    ax7.set_ylabel('Wave Number')
    ax7.set_ylim(0, 4)
    ax7.legend()

    # Plot normalized wavelength
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.plot(frames, data['normalized_wavelengths'], label='Normalized Wavelength')
    ax8.plot(frames, smoothed_data['smoothed_normalized_wavelengths'], label='Smoothed Normalized Wavelength', linestyle='--')
    ax8.set_title('Normalized Wavelength over time')
    ax8.set_xlabel('Frame')
    ax8.set_ylabel('Normalized Wavelength')
    ax8.set_ylim(0, 2)
    ax8.legend()
    
    # Plot dominant frequency
    ax9 = fig.add_subplot(gs[4, 2:])
    ax9.plot(frames, results['interpolated_freqs'])
    ax9.set_title('Dominant Temporal Frequency Over Time')
    ax9.set_xlabel('Frame Number')
    ax9.set_ylabel('Frequency (cycles per frame)')
    ax9.set_ylim(0, max(results['interpolated_freqs']) * 1.1)
    ax9.grid(True)

    # Plot power spectral density
    ax10 = fig.add_subplot(gs[3, 2:])
    ax10.semilogy(results['f'], results['psd'])
    ax10.set_title('Power Spectral Density')
    ax10.set_xlabel('Frequency (Hz)')
    ax10.set_ylabel('Power/Frequency')
    ax10.grid(True)

    # Plot curvature across frames
    ax11 = fig.add_subplot(gs[1, :2])  # New subplot for curvature
    curvature_data = np.array(results['curvatures'])
    im = ax11.imshow(curvature_data.T, aspect='auto', cmap='viridis', 
                     extent=[0, len(frames), 0, curvature_data.shape[1]])
    ax11.set_title('Curvature Across Frames')
    ax11.set_xlabel('Frame')
    ax11.set_ylabel('Position Along Worm')
    #cbar = plt.colorbar(im, ax=ax11)
    #cbar.set_label('Curvature')   

    # New subplot for curvature_1d
    ax12 = fig.add_subplot(gs[2, :2])  # Use the entire width for the new subplot
    ax12.plot(frames, results['curvature_time_series'])
    ax12.set_title('Curvature 1D Time Series')
    ax12.set_xlabel('Frame')
    ax12.set_ylabel('Curvature')
    ax12.grid(True)

    # Align x-axis for curvature plots
    ax1.set_xlim(0, len(frames))
    ax2.set_xlim(0, len(frames))
    ax3.set_xlim(0, len(frames))
    ax11.set_xlim(0, len(frames))
    ax12.set_xlim(0, len(frames))  

    plt.tight_layout()
    plt.savefig('swim_worm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
 
 
visualize_worm_analysis(swim_shapeanalysis)

# Analyze the video
cropped_analysis = analyze_video(video_segments)

fframe_analysis = analyze_video(hd_video_segments)

with open('wormshape_hdresults.pkl', 'wb') as file:
    pickle.dump(fframe_analysis, file)


##h5 stuff

import json
from collections.abc import Mapping, Sequence
import numpy as np

def analyze_dict_structure(data, indent="", max_depth=None, current_depth=0):
    if max_depth is not None and current_depth > max_depth:
        return "..."

    if isinstance(data, Mapping):
        result = "{\n"
        for key, value in data.items():
            result += f"{indent}  {repr(key)}: {analyze_dict_structure(value, indent + '  ', max_depth, current_depth + 1)},\n"
        result += indent + "}"
        return result
    elif isinstance(data, np.ndarray):
        return f"ndarray(shape={data.shape}, dtype={data.dtype})"
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) > 10:
            sample = data[:3]
            sample_str = ", ".join(str(analyze_dict_structure(item, indent + "  ", max_depth, current_depth + 1)) for item in sample)
            return f"{type(data).__name__}(length={len(data)}, sample=[{sample_str}, ...])"
        result = "[\n"
        for item in data:
            result += f"{indent}  {analyze_dict_structure(item, indent + '  ', max_depth, current_depth + 1)},\n"
        result += indent + "]"
        return result
    else:
        return f"{type(data).__name__}"

def print_dict_structure(data, max_depth=None):
    print(analyze_dict_structure(data, max_depth=max_depth))



print_dict_structure(swim_shapeanalysis)


