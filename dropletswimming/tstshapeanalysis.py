import pickle
import numpy as np
from scipy import interpolate, signal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage import morphology, graph
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

with open('tstswimcrop.pkl', 'rb') as file:
    hd_video_segments = pickle.load(file)

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

def find_longest_path(skeleton):
    points = np.argwhere(skeleton)
    n = len(points)
    distances = np.sqrt(np.sum((points[:, None, :] - points[None, :, :])**2, axis=-1))
    adj_matrix = csr_matrix(np.where((distances <= np.sqrt(2)) & (distances > 0), distances, 0))
    
    dist_matrix = shortest_path(adj_matrix, directed=False, method='D')
    i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    
    path = []
    predecessors = shortest_path(adj_matrix, directed=False, indices=i, return_predecessors=True)[1]
    current = j
    while current != i:
        path.append(current)
        current = predecessors[current]
    path.append(i)
    
    return points[path[::-1]]

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

def analyze_periodicity(curvature):
    fft = np.fft.fft(curvature)
    frequencies = np.fft.fftfreq(len(curvature))
    dominant_freq = frequencies[np.argmax(np.abs(fft[1:])) + 1]
    return dominant_freq

def track_endpoints(frames, smooth_points):
    endpoints = []
    for skeleton in smooth_points:
        endpoints.append((skeleton[0], skeleton[-1]))
    return np.array(endpoints)

from scipy.optimize import linear_sum_assignment

def group_endpoints(endpoints, window_size):
    """
    Group endpoints based on their proximity across frames.
    Returns grouped endpoints and the mapping of original indices to group indices.
    """
    grouped_endpoints = np.zeros((2, window_size, 2))
    index_mapping = np.zeros((window_size, 2), dtype=int)
    
    # Use the first frame as initial groups
    grouped_endpoints[0, 0] = endpoints[0, 0]
    grouped_endpoints[1, 0] = endpoints[0, 1]
    index_mapping[0] = [0, 1]
    
    for i in range(1, window_size):
        # Calculate distances between current endpoints and existing groups
        distances = np.linalg.norm(endpoints[i][:, np.newaxis] - grouped_endpoints[:, i-1], axis=2)
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(distances)
        
        # Assign endpoints to groups based on the optimal assignment
        grouped_endpoints[col_ind, i] = endpoints[i, row_ind]
        index_mapping[i] = col_ind
    
    return grouped_endpoints, index_mapping

def continuous_head_identification(endpoints, frames, window_size=10, error_threshold=5):
    head_positions = []
    tail_positions = []
    confidences = []
    current_head_index = 0
    
    for i in range(len(frames)):

        # Calculate the start and end indices for the sliding window
        start = max(0, i - window_size // 2)
        end = min(len(frames), start + window_size)
             # Adjust start if end is at the boundary
        if end == len(frames):
            start = max(0, end - window_size)
        # For initial frames, use movement-based identification
        window_endpoints = endpoints[start:end]
        grouped_endpoints, index_mapping = group_endpoints(window_endpoints, end - start)
        movements = np.diff(grouped_endpoints, axis=1)
        cumulative_movement = np.sum(np.abs(movements), axis=(1, 2))
        head_group_index = np.argmax(cumulative_movement)
        # Map the head group index back to the original endpoint index for the current frame
        frame_index_in_window = i - start
        current_head_index = np.where(index_mapping[frame_index_in_window] == head_group_index)[0][0]
        current_head_position = endpoints[i, current_head_index]
        current_tail_position = endpoints[i, 1 - current_head_index]  # The other endpoint is the tail
        confidence = 1.0  # High confidence for initial determination
               
        # Check for sudden changes
        if len(head_positions) > 0:
            prev_head_position = head_positions[-1]
            distance = np.linalg.norm(current_head_position - prev_head_position)
            if distance > 25:  # If head moved more than portion of worm length
                # Potential error detected
                confidence = 0.5  # Lower confidence due to change
                
                # Error correction
                if len(head_positions) >= error_threshold:
                    # Check if this change is consistent with recent history
                    recent_head_positions = head_positions[-error_threshold:]
                    #if currend head coords is not within 25 pix of one of the last threshold positions
                    if any(np.linalg.norm(pos - current_head_position) > 25 for pos in recent_head_positions):
                        # This change is not supported by recent history, likely an error
                        # Find the closest endpoint in the current frame to the previous head position
                        distances = np.linalg.norm(endpoints[i] - prev_head_position, axis=1)
                        closest_index = np.argmin(distances)
                        current_head_position = endpoints[i, closest_index]
                        current_tail_position = endpoints[i, 1 - closest_index]
                        current_head_index = closest_index
                        confidence = 0.7  # Moderate confidence after correction
        
            else:
                confidence = 0.8  # High confidence when consistent
                
        else:
            confidence = 0.9  # High confidence for first comparison
            
        previous_head_index = current_head_index
        head_positions.append(current_head_position)
        tail_positions.append(current_tail_position)
        confidences.append(confidence)
    
    return head_positions, tail_positions, confidences


def apply_head_correction(head_positions, confidences, correction_window=5):
    corrected_positions = head_positions.copy()
    for i in range(len(head_positions)):
        if confidences[i] < 0.7:  # If confidence is low
            # Look at surrounding frames
            start = max(0, i - correction_window)
            end = min(len(head_positions), i + correction_window + 1)
            surrounding_positions = head_positions[start:end]
            # Choose the median head position in the surrounding frames
            corrected_positions[i] = np.median(surrounding_positions, axis=0)
    return corrected_positions


def calculate_head_bend(skeleton, head_index, segment_length=5):
    head_segment = skeleton[0:segment_length] if head_index == 0 else skeleton[-segment_length:]
    body_vector = skeleton[-1] - skeleton[0]
    head_vector = head_segment[-1] - head_segment[0]
    angle = np.arctan2(np.cross(head_vector, body_vector), np.dot(head_vector, body_vector))
    return np.degrees(angle)

def analyze_head_bends(smoothed_head_bends, fps):
    # Find peaks and troughs
    peak_threshold = np.std(smoothed_head_bends)/3
    peaks, _ = find_peaks(smoothed_head_bends, prominence=peak_threshold, distance=8)
    troughs, _ = find_peaks(-np.array(smoothed_head_bends), prominence=peak_threshold, distance=8)
    
    # Calculate metrics
    num_peaks = len(peaks)
    num_troughs = len(troughs)
    avg_peak_depth = np.mean(np.array(smoothed_head_bends)[peaks])
    avg_trough_depth = np.mean(np.array(smoothed_head_bends)[troughs])
    max_peak_depth = np.max(np.array(smoothed_head_bends)[peaks])
    max_trough_depth = np.min(np.array(smoothed_head_bends)[troughs])
    
    # Calculate bending frequency
    all_extrema = sorted(np.concatenate([peaks, troughs]))
    if len(all_extrema) > 1:
        times = np.arange(len(smoothed_head_bends)) / fps
        extrema_times = times[all_extrema]
        bend_intervals = np.diff(extrema_times)
        avg_bend_frequency = 1 / np.mean(bend_intervals)
    else:
        avg_bend_frequency = 0

    # Perform FFT on smoothed data
    fft = np.fft.fft(smoothed_head_bends)
    freqs = np.fft.fftfreq(len(smoothed_head_bends), 1/fps)
    dominant_freq = freqs[np.argmax(np.abs(fft[1:]) + 1)]
    
    return {
        'num_peaks': num_peaks,
        'num_troughs': num_troughs,
        'avg_peak_depth': avg_peak_depth,
        'avg_trough_depth': avg_trough_depth,
        'max_peak_depth': max_peak_depth,
        'max_trough_depth': max_trough_depth,
        'avg_bend_frequency': avg_bend_frequency,
        'dominant_freq': dominant_freq,
        'peaks': peaks,
        'troughs': troughs,
        'fft': fft,
        'freqs': freqs
    }


def analyze_shape(skeleton, frame_num, head_position):
    longest_path = find_longest_path(skeleton)
    
    t = np.arange(len(longest_path))
    x, y = longest_path[:, 1], longest_path[:, 0]
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.linspace(0, 1, num=100)
    smooth_points = np.column_stack(interpolate.splev(unew, tck))
    
    # Curvature calculation
    window_size, sigma = 50, 10
    curvature = gaussian_weighted_curvature(smooth_points, window_size, sigma)

    # Wavelength calculation
    peaks, _ = find_peaks(curvature)
    if len(peaks) > 1:
        wavelengths = np.diff(peaks)
        avg_wavelength = np.mean(wavelengths) * 2  # multiply by 2 for full wave
    else:
        avg_wavelength = None  # or some default value

    # Amplitude calculation (using peak-to-trough)
    if len(peaks) > 0:
        troughs, _ = find_peaks(-curvature)
        if len(troughs) > 0:
            amplitudes = curvature[peaks] - curvature[troughs[np.searchsorted(troughs, peaks) - 1]]
            max_amplitude = np.max(amplitudes)
            avg_amplitude = np.mean(amplitudes)
        else:
            max_amplitude = np.max(curvature) - np.min(curvature)
            avg_amplitude = max_amplitude / 2
    else:
        max_amplitude = np.max(curvature) - np.min(curvature)
        avg_amplitude = max_amplitude / 2


    # Worm length
    worm_length = np.sum(np.sqrt(np.sum(np.diff(smooth_points, axis=0)**2, axis=1)))

    wave_number = worm_length / avg_wavelength #Number of waves along the worm body
    normalized_wavelength = avg_wavelength / worm_length #Wavelength as proportion of worm length

    # Frequency analysis (spatial)
    spatial_freq = np.abs(np.fft.fft(curvature))
    dominant_spatial_freq = np.abs(np.fft.fftfreq(len(curvature))[np.argmax(spatial_freq[1:]) + 1])

    if head_position is None:
        return {
            'frame': frame_num,
            'smooth_points': smooth_points,
            'curvature': curvature,
            'max_amplitude': max_amplitude,
            'avg_amplitude': avg_amplitude,
            'wavelength': avg_wavelength,
            'worm_length': worm_length,
            'wave_number': wave_number,
            'normalized_wavelength': normalized_wavelength,
            'dominant_spatial_freq': dominant_spatial_freq,
            'head_bend': None
        }
        
    # Find the index of the point closest to the head position
    head_index = np.argmin(np.sum((smooth_points - head_position)**2, axis=1))
    head_bend = calculate_head_bend(smooth_points, head_index)

    return {
        'frame': frame_num,
        'smooth_points': smooth_points,
        'curvature': curvature,
        'max_amplitude': max_amplitude,
        'avg_amplitude': avg_amplitude,
        'wavelength': avg_wavelength,
        'worm_length': worm_length,
        'wave_number': wave_number,
        'normalized_wavelength': normalized_wavelength,
        'dominant_spatial_freq': dominant_spatial_freq,
        'head_bend': head_bend
    }


def analyze_video(segmentation_dict, fps=10, window_size=5, overlap=2.5):
    # Initialize result containers
    frames = []
    smooth_points = []
    curvatures = []
    max_amplitudes = []
    avg_amplitudes = []
    wavelengths = []
    worm_lengths = []
    wave_numbers = []
    normalized_wavelengths = []
    dominant_spatial_freqs = []
    head_bends = []
    masks = []
    
    # First pass to get endpoints
    for frame_num, frame_data in segmentation_dict.items():
        print("Find head: " + str(frame_num))
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)
        
        frame_results = analyze_shape(skeleton, frame_num, None)
        frames.append(frame_results['frame'])
        smooth_points.append(frame_results['smooth_points'])
        curvatures.append(frame_results['curvature'])
        max_amplitudes.append(frame_results['max_amplitude'])
        avg_amplitudes.append(frame_results['avg_amplitude'])
        wavelengths.append(frame_results['wavelength'])
        worm_lengths.append(frame_results['worm_length'])
        wave_numbers.append(frame_results['wave_number'])
        normalized_wavelengths.append(frame_results['normalized_wavelength'])
        dominant_spatial_freqs.append(frame_results['dominant_spatial_freq'])
        head_bends.append(frame_results['head_bend'])
        masks.append(cleaned_mask)

    # Identify head
    endpoints = track_endpoints(frames, smooth_points)
    #head_index = identify_head(endpoints)
    # Apply continuous head identification with error detection and correction
    head_positions, tail_positions, confidences = continuous_head_identification(endpoints, frames)
    #corrected_head_positions = apply_head_correction(head_positions, confidences)
    corrected_head_positions = head_positions

    # Second pass with head information
    frames, smooth_points, curvatures, max_amplitudes, avg_amplitudes, wavelengths, worm_lengths, wave_numbers, normalized_wavelengths, dominant_spatial_freqs, head_bends = ([] for _ in range(11))
    
    for frame_num, frame_data in segmentation_dict.items():
        print("Analyze shape: " + str(frame_num))
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)

        head_index = corrected_head_positions[frame_num]
        #head_index = head_positions[frame_num]
        
        frame_results = analyze_shape(skeleton, frame_num, head_index)
        frames.append(frame_results['frame'])
        smooth_points.append(frame_results['smooth_points'])
        curvatures.append(frame_results['curvature'])
        max_amplitudes.append(frame_results['max_amplitude'])
        avg_amplitudes.append(frame_results['avg_amplitude'])
        wavelengths.append(frame_results['wavelength'])
        worm_lengths.append(frame_results['worm_length'])
        wave_numbers.append(frame_results['wave_number'])
        normalized_wavelengths.append(frame_results['normalized_wavelength'])
        dominant_spatial_freqs.append(frame_results['dominant_spatial_freq'])
        head_bends.append(frame_results['head_bend'])

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
    
    # Interpolate dominant frequencies for all frames
    frame_numbers = np.arange(len(frames))
    interpolated_freqs = np.interp(frame_numbers / fps, time_points, dominant_freqs)
    
    # Apply smoothing
    smoothed_max_amplitudes = smooth_metric(max_amplitudes)
    smoothed_avg_amplitudes = smooth_metric(avg_amplitudes)
    smoothed_wavelengths = smooth_metric(wavelengths)
    smoothed_worm_lengths = smooth_metric(worm_lengths)
    smoothed_wave_numbers = smooth_metric(wave_numbers)
    smoothed_normalized_wavelengths = smooth_metric(normalized_wavelengths)
    smoothed_head_bends = smooth_metric(head_bends)

    # Analyze head bends
    head_bend_analysis = analyze_head_bends(smoothed_head_bends, fps)
    
    # Prepare final results dictionary
    final_results = {
        'frames': frames,
        'smooth_points': smooth_points,
        'curvatures': curvatures,
        'max_amplitudes': max_amplitudes,
        'avg_amplitudes': avg_amplitudes,
        'wavelengths': wavelengths,
        'worm_lengths': worm_lengths,
        'wave_numbers': wave_numbers,
        'normalized_wavelengths': normalized_wavelengths,
        'dominant_spatial_freqs': dominant_spatial_freqs,
        'head_bends': head_bends,
        'smoothed_max_amplitudes': smoothed_max_amplitudes,
        'smoothed_avg_amplitudes': smoothed_avg_amplitudes,
        'smoothed_wavelengths': smoothed_wavelengths,
        'smoothed_worm_lengths': smoothed_worm_lengths,
        'smoothed_wave_numbers': smoothed_wave_numbers,
        'smoothed_normalized_wavelengths': smoothed_normalized_wavelengths,
        'smoothed_head_bends': smoothed_head_bends,
        'interpolated_freqs': interpolated_freqs,
        'f': f,
        'psd': psd,
        'head_bend_analysis': head_bend_analysis,
        'fps': fps,
        'curvature_time_series': curvature_1d
    }

    # Add head tracking information to the final results
    final_results['head_positions'] = corrected_head_positions
    final_results['tail_positions'] = tail_positions
    final_results['head_position_confidences'] = confidences
    final_results['masks'] = masks
    
    return final_results



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

result = coordinates_to_bool_array(junctions)

image_array = np.uint8(skeleton * 255)
image = Image.fromarray(image_array)
image.save('skeleton.png')


def visualize_worm_analysis(results):
    frames = results['frames']
    fps = results['fps']
    times = [frame / fps for frame in frames]

    metrics = ['max_amplitudes', 'avg_amplitudes', 'wavelengths', 'worm_lengths', 
               'dominant_spatial_freqs', 'wave_numbers', 'normalized_wavelengths', 'head_bends']
    smoothed_metrics = ['max_amplitudes', 'avg_amplitudes', 'wavelengths', 'worm_lengths', 
                        'wave_numbers', 'normalized_wavelengths', 'head_bends']
    
    data = {metric: results[metric] for metric in metrics}
    smoothed_data = {f'smoothed_{metric}': results[f'smoothed_{metric}'] for metric in smoothed_metrics}

    # Create the main figure and subplots
    fig = plt.figure(figsize=(25, 30))  # Increased figure height
    gs = GridSpec(6, 4, figure=fig)  # Added one more row

    # Plot max and avg amplitudes
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(frames, data['max_amplitudes'], label='Max Amplitude')
    ax1.plot(frames, data['avg_amplitudes'], label='Avg Amplitude')
    ax1.plot(frames, smoothed_data['smoothed_max_amplitudes'], label='Smoothed Max Amplitude', linestyle='--')
    ax1.plot(frames, smoothed_data['smoothed_avg_amplitudes'], label='Smoothed Avg Amplitude', linestyle='--')
    ax1.set_title('Amplitudes over time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Amplitude')
    ax1.legend()

    # Plot wavelength
    ax2 = fig.add_subplot(gs[2, :2])
    ax2.plot(frames, data['wavelengths'], label='Wavelength')
    ax2.plot(frames, smoothed_data['smoothed_wavelengths'], label='Smoothed Wavelength', linestyle='--')
    ax2.set_title('Wavelength over time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Wavelength')
    ax2.legend()

    # Plot worm length
    ax3 = fig.add_subplot(gs[3, :2])
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
    ax9 = fig.add_subplot(gs[4, :2])
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

    # Plot head bends
    ax11 = fig.add_subplot(gs[1, :2])
    ax11.plot(times, data['head_bends'], alpha=0.5, label='Raw Head Bend', color='lightblue')
    ax11.plot(times, smoothed_data['smoothed_head_bends'], label='Smoothed Head Bend', color='darkblue')
    
    # Add peaks and troughs to the plot
    head_bend_analysis = results['head_bend_analysis']
    ax11.scatter([times[i] for i in head_bend_analysis['peaks']], 
                 [smoothed_data['smoothed_head_bends'][i] for i in head_bend_analysis['peaks']], 
                 color='red', s=50, label='Peaks', zorder=5)
    ax11.scatter([times[i] for i in head_bend_analysis['troughs']], 
                 [smoothed_data['smoothed_head_bends'][i] for i in head_bend_analysis['troughs']], 
                 color='green', s=50, label='Troughs', zorder=5)

    ax11.set_ylabel('Head Bend Angle (degrees)')
    ax11.set_xlabel('Time (seconds)')
    ax11.set_title('Worm Head Bends Over Time')
    ax11.legend()
    ax11.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax12 = fig.add_subplot(gs[4, 2:])
    head_bend_analysis = results['head_bend_analysis']
    ax12.plot(head_bend_analysis['freqs'][1:len(head_bend_analysis['freqs'])//2], 
              np.abs(head_bend_analysis['fft'][1:len(head_bend_analysis['fft'])//2]))
    ax12.set_title('Frequency Spectrum of Head Bends')
    ax12.set_xlabel('Frequency (Hz)')
    ax12.set_ylabel('Magnitude')
    ax12.grid(True)

    # Add text information in the top right corner
    ax_text = fig.add_subplot(gs[0, 3:])
    ax_text.axis('off')
    
    head_bend_analysis = results['head_bend_analysis']
    info_text = (
        f"Head Bend Analysis:\n\n"
        f"Number of peaks: {head_bend_analysis['num_peaks']}\n"
        f"Number of troughs: {head_bend_analysis['num_troughs']}\n"
        f"Average peak depth: {head_bend_analysis['avg_peak_depth']:.2f} degrees\n"
        f"Average trough depth: {head_bend_analysis['avg_trough_depth']:.2f} degrees\n"
        f"Maximum peak depth: {head_bend_analysis['max_peak_depth']:.2f} degrees\n"
        f"Maximum trough depth: {head_bend_analysis['max_trough_depth']:.2f} degrees\n"
        f"Average bending frequency: {head_bend_analysis['avg_bend_frequency']:.2f} Hz\n"
        f"Dominant frequency from FFT: {head_bend_analysis['dominant_freq']:.2f} Hz"
    )
    
    ax_text.text(0, 1, info_text, verticalalignment='top', fontsize=15, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('integrated_worm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
 
 
# Analyze the video
cropped_analysis = analyze_video(video_segments)

fframe_analysis = analyze_video(hd_video_segments)

with open('wormshape_hdresults.pkl', 'wb') as file:
    pickle.dump(fframe_analysis, file)


