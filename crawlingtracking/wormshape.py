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
import os

with open('propagation_fixedcrop.pkl', 'rb') as file:
    video_segments = pickle.load(file)

with open('hd_video_segments.pkl', 'rb') as file:
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

# Visualize the results
visualize_worm_analysis(fframe_analysis)


def visualize_head_localization(analysis_results, frame_num, figure_name):
    """
    Visualize the head localization for a specific frame.
    
    Parameters:
    - analysis_results: The output dictionary from analyze_video function
    - frame_num: The frame number to visualize
    - figure_name: Name of the file to save the figure
    """
    # Extract necessary data
    #segmentation_dict = analysis_results['segmentation_dict']
    smooth_points = analysis_results['smooth_points'][frame_num]
    head_positions = analysis_results['head_positions']
    
    # Flip x, y coordinates
    smooth_points = smooth_points[:, ::-1]
    
    # Get the segmentation mask for the specified frame
    mask = analysis_results['masks'][frame_num]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the segmentation mask
    ax.imshow(mask, cmap='gray', alpha=0.5)
    
    # Plot the skeleton
    ax.plot(smooth_points[:, 1], smooth_points[:, 0], 'b-', linewidth=2, alpha=0.7)
    
    # Highlight the head position
    head_index = head_positions[frame_num]
    #head_point = smooth_points[0] if head_index == 0 else smooth_points[-1]
    head_point=head_index
    ax.plot(head_point[0], head_point[1], 'ro', markersize=1, label='Head')
    
    # Set title and labels
    ax.set_title(f'Frame {frame_num}: Head Localization')
    ax.set_xlabel('Y coordinate')
    ax.set_ylabel('X coordinate')
    
    # Add legend
    ax.legend()
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure saved as {figure_name}")



visualize_head_localization(fframe_analysis, frame_num=26, figure_name='head_localization_frame.png')


data = np.array(fframe_analysis["head_positions"])
x = data[:, 0]
y = data[:, 1]

# Create the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(x, y, alpha=0.5)
plt.title('Head Positions')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Invert y-axis to match image coordinates (0,0 at top-left)
plt.gca().invert_yaxis()

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()
plt.savefig('head_positions_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

###Stuff###


""" def analyze_video(segmentation_dict, fps=10, window_size=5, overlap=2.5):
    results = []
    curvature_time_series = []
    
    # First pass to get endpoints
    for frame_num, frame_data in segmentation_dict.items():
        print(f"Processing frame {frame_num}")
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)
        
        frame_results = analyze_shape(skeleton, frame_num, None)
        results.append(frame_results)
        curvature_time_series.append(np.mean(frame_results['curvature']))

    # Identify head
    endpoints = track_endpoints(results)
    head_index = identify_head(endpoints)

    # Second pass with head information
    results = []
    for frame_num, frame_data in segmentation_dict.items():
        print(f"Processing frame {frame_num}")
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)
        
        frame_results = analyze_shape(skeleton, frame_num, head_index)
        results.append(frame_results)
        curvature_time_series.append(np.mean(frame_results['curvature']))

    
    # Temporal frequency analysis
    curvature_1d = np.array(curvature_time_series)
    curvature_1d = (curvature_1d - np.mean(curvature_1d)) / np.std(curvature_1d)
    
    nperseg = int(window_size * fps)
    noverlap = int(overlap * fps)
    
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
    frame_numbers = np.arange(len(results))
    interpolated_freqs = np.interp(frame_numbers / fps, time_points, dominant_freqs)
    
    # Add temporal frequency to results and apply smoothing
    smoothed_results = []
    for i, result in enumerate(results):
        smoothed_result = result.copy()
        smoothed_result['dominant_temporal_freq'] = interpolated_freqs[i]
        for metric in ['max_amplitude', 'avg_amplitude', 'wavelength', 'worm_length', 'wave_number', 'normalized_wavelength']:
            if metric in result:
                smoothed_result[f'smoothed_{metric}'] = result[metric]
        smoothed_results.append(smoothed_result)
    
    # Apply smoothing after collecting all results
    for metric in ['max_amplitude', 'avg_amplitude', 'wavelength', 'worm_length', 'wave_number', 'normalized_wavelength']:
        data = [r[metric] for r in results if metric in r]
        if data:
            smoothed_data = smooth_metric(data)
            for i, result in enumerate(smoothed_results):
                if i < len(smoothed_data):
                    result[f'smoothed_{metric}'] = smoothed_data[i]

    # Add smoothing for head bends
    head_bends = [r['head_bend'] for r in results]
    smoothed_head_bends = smooth_metric(head_bends)
    for i, result in enumerate(smoothed_results):
        result['smoothed_head_bend'] = smoothed_head_bends[i]
    
    return smoothed_results, frame_numbers, interpolated_freqs """



def analyze_head_bends(smoothed_results):
    head_bends = [r['smoothed_head_bend'] for r in smoothed_results]
    
    # Find peaks (deep bends)
    peaks, _ = find_peaks(np.abs(head_bends), height=np.std(head_bends))
    
    # Calculate metrics
    num_bends = len(peaks)
    avg_bend_depth = np.mean(np.abs(np.array(head_bends)[peaks]))
    max_bend_depth = np.max(np.abs(np.array(head_bends)[peaks]))
    
    return {
        'num_bends': num_bends,
        'avg_bend_depth': avg_bend_depth,
        'max_bend_depth': max_bend_depth
    }

# After running analyze_video
head_bend_analysis = analyze_head_bends(video_analysis)


def visualize_head_bends(results, smoothed_results, fps=10):
    # Extract frame numbers and bend angles
    frames = [r['frame'] for r in results]
    times = [f / fps for f in frames]  # Convert frames to seconds
    raw_bends = [r['head_bend'] for r in results]
    smoothed_bends = [r['smoothed_head_bend'] for r in smoothed_results]

    # Find peaks and troughs
    peak_threshold = np.std(smoothed_bends)/3
    peaks, _ = find_peaks(smoothed_bends, prominence=peak_threshold, distance=8)
    troughs, _ = find_peaks(-np.array(smoothed_bends), prominence=peak_threshold, distance=8)

    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1])  # Main plot and residual plot

    # Main plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, raw_bends, alpha=0.5, label='Raw data', color='lightblue')
    ax1.plot(times, smoothed_bends, label='Smoothed data', color='darkblue')
    
    # Add peaks and troughs to the plot
    ax1.scatter([times[i] for i in peaks], [smoothed_bends[i] for i in peaks], 
                color='red', s=50, label='Peaks', zorder=5)
    ax1.scatter([times[i] for i in troughs], [smoothed_bends[i] for i in troughs], 
                color='green', s=50, label='Troughs', zorder=5)

    ax1.set_ylabel('Head Bend Angle (degrees)')
    ax1.set_title('Worm Head Bends Over Time')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Annotate some peaks and troughs
    for i, peak in enumerate(peaks[:5]):  # Annotate first 5 peaks
        ax1.annotate(f'P{i+1}', (times[peak], smoothed_bends[peak]), 
                     xytext=(5, 5), textcoords='offset points')
    for i, trough in enumerate(troughs[:5]):  # Annotate first 5 troughs
        ax1.annotate(f'T{i+1}', (times[trough], smoothed_bends[trough]), 
                     xytext=(5, -15), textcoords='offset points')

    # Residual plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    residuals = np.array(raw_bends) - np.array(smoothed_bends)
    ax2.plot(times, residuals, color='gray', alpha=0.7)
    ax2.fill_between(times, residuals, 0, color='gray', alpha=0.3)
    ax2.set_ylabel('Residuals')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    plt.savefig('headbends.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional analysis
    analyze_head_bend_patterns(times, raw_bends, smoothed_bends, peaks, troughs)


def analyze_head_bend_patterns(times, raw_bends, smoothed_bends, peaks, troughs):
    # Calculate metrics
    num_peaks = len(peaks)
    num_troughs = len(troughs)
    avg_peak_depth = np.mean(np.array(smoothed_bends)[peaks])
    avg_trough_depth = np.mean(np.array(smoothed_bends)[troughs])
    max_peak_depth = np.max(np.array(smoothed_bends)[peaks])
    max_trough_depth = np.min(np.array(smoothed_bends)[troughs])
    
    # Calculate bending frequency
    all_extrema = sorted(np.concatenate([peaks, troughs]))
    if len(all_extrema) > 1:
        extrema_times = np.array(times)[all_extrema]
        bend_intervals = np.diff(extrema_times)
        avg_bend_frequency = 1 / np.mean(bend_intervals)
    else:
        avg_bend_frequency = 0

    print(f"Number of peaks: {num_peaks}")
    print(f"Number of troughs: {num_troughs}")
    print(f"Average peak depth: {avg_peak_depth:.2f} degrees")
    print(f"Average trough depth: {avg_trough_depth:.2f} degrees")
    print(f"Maximum peak depth: {max_peak_depth:.2f} degrees")
    print(f"Maximum trough depth: {max_trough_depth:.2f} degrees")
    print(f"Average bending frequency: {avg_bend_frequency:.2f} Hz")

    # Perform FFT on smoothed data
    fft = np.fft.fft(smoothed_bends)
    freqs = np.fft.fftfreq(len(smoothed_bends), times[1] - times[0])
    dominant_freq = freqs[np.argmax(np.abs(fft[1:]) + 1)]
    
    print(f"Dominant frequency from FFT: {dominant_freq:.2f} Hz")

    # Plot FFT
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[1:len(freqs)//2], np.abs(fft[1:len(fft)//2]))
    plt.title('Frequency Spectrum of Head Bends')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
    plt.savefig('headpatterns.png', dpi=300, bbox_inches='tight')
    plt.close()





smoothed_results, frame_numbers, interpolated_freqs = analyze_video(video_segments)
visualize_head_bends(smoothed_results, smoothed_results)

visualize_worm_metrics(video_analysis, frame_numbers, interpolated_freqs, f, psd, start_frame=0, end_frame=None, sample_rate=1, fps=10)



def visualize_frame_analysis(video_analysis, frame_number):
    # Find the data for the specified frame
    frame_data = next((data for data in video_analysis if data['frame'] == frame_number), None)
    
    if frame_data is None:
        print(f"No data found for frame {frame_number}")
        return

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Frame {frame_number}: Curvature Analysis', fontsize=16)

    # Custom colormap: blue (low curvature) to red (high curvature)
    colors = ['blue', 'cyan', 'yellow', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # Plot snake shape with curvature color
    ax = axs[0]
    scatter = ax.scatter(frame_data['smooth_points'][:, 0], frame_data['smooth_points'][:, 1], 
                         c=frame_data['curvature'], cmap=cmap)
    ax.set_title('Snake Shape with Curvature')
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    fig.colorbar(scatter, ax=ax)

    # Add text annotations for straightness and dominant frequency
    ax.text(0.05, 0.95, f"Straightness: {frame_data['straightness']:.4f}", 
            transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.90, f"Dominant Frequency: {frame_data['dominant_freq']:.4f}", 
            transform=ax.transAxes, verticalalignment='top')

    # Plot curvature along the snake's length
    ax = axs[1]
    ax.plot(frame_data['curvature'])
    ax.set_title('Curvature Along Snake Length')
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Curvature')

    plt.tight_layout()
    plt.savefig(f'frame_{frame_number}_analysis.png')
    plt.close()

visualize_frame_analysis(video_analysis, 0)
frames_to_visualize = [0, 250, 299, 599]  # Example frame numbers
for frame in frames_to_visualize:
    visualize_frame_analysis(video_analysis, frame)

##Check stuff
video_segments[41][1][0]
mask = video_segments[100][1][0]
cleaned_mask = clean_mask(mask)
skeleton = get_skeleton(cleaned_mask)
skeleton
image_size = (94, 94)  # Assuming a grid of 80x80
image = np.zeros(image_size, dtype=np.uint8)
# Set the coordinates in the array to white
for coord in smooth_points_int:
    image[coord[0], coord[1]] = 255
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.axis('off')  # Turn off the axis labels
plt.savefig('smooth.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Visualization
def visualize_frame(frame_data, original_image=None, mask=None):
    fig = plt.figure(figsize=(15, 10))
    
    # Original image with skeleton overlay
    ax1 = fig.add_subplot(221)
    if original_image is not None:
        ax1.imshow(original_image, cmap='gray')
    ax1.plot(frame_data['smooth_points'][:, 1], frame_data['smooth_points'][:, 0], 'r-')
    ax1.set_title(f"Frame {frame_data['frame']}")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")

    # Mask
    ax2 = fig.add_subplot(222)
    if mask is not None:
        ax2.imshow(mask, cmap='binary')
    ax2.set_title("Segmentation Mask")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")

    # Curvature along the body
    ax3 = fig.add_subplot(223)
    im = ax3.imshow(frame_data['curvature'].reshape(-1, 1), aspect='auto', cmap='viridis')
    ax3.set_title("Curvature along the body")
    ax3.set_xlabel("Position along the body")
    ax3.set_ylabel("Curvature")
    plt.colorbar(im, ax=ax3)

    # Metrics
    ax4 = fig.add_subplot(224)
    metrics = [
        f"Straightness: {frame_data['straightness']:.3f}",
        f"Dominant frequency: {frame_data['dominant_freq']:.3f}",
        f"Max curvature: {np.max(frame_data['curvature']):.3f}",
        f"Mean curvature: {np.mean(frame_data['curvature']):.3f}"
    ]
    ax4.axis('off')
    ax4.text(0.5, 0.5, '\n'.join(metrics), ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'frame_{frame_data["frame"]}_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

# Visualize a specific frame (e.g., frame 100)
frame_num = 250
mask = clean_mask(video_segments[frame_num][1][0])
visualize_frame(video_analysis[frame_num], mask=mask)

# Plot shape metrics over time
frames = [data['frame'] for data in video_analysis]
straightness = [data['straightness'] for data in video_analysis]
dominant_freq = [data['dominant_freq'] for data in video_analysis]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(frames, straightness)
ax1.set_title("Straightness over time")
ax1.set_xlabel("Frame")
ax1.set_ylabel("Straightness")

ax2.plot(frames, dominant_freq)
ax2.set_title("Dominant frequency over time")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
plt.savefig('ftime.png', bbox_inches='tight', dpi=300)
plt.close()

