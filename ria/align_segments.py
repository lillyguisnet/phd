import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import distance_transform_edt
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from scipy import ndimage
from scipy.signal import medfilt


def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        nb_frames = 0
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            nb_frames += 1

    
    print(f"{nb_frames} frames loaded from {filename}")
    return cleaned_segments

filename = '/home/lilly/phd/ria/data_analyzed/cleaned_segments/ria-MMH99_10s_20190813_03_crop_riasegmentation_cleanedsegments.h5'

cleaned_segments = load_cleaned_segments_from_h5(filename)


def crop_around_segment(video_segments, segment_id, input_folder, output_folder, padding=1):
    """
    Crops frames and masks around a specific segment with padding, using the maximum extent of the segment across all frames.
    
    Parameters:
        video_segments (dict): Dictionary containing segmentation masks for each frame
        segment_id (int): ID of the segment to crop around
        input_folder (str): Path to the folder containing original frames
        output_folder (str): Path to save cropped frames
        padding (int): Number of pixels to pad around the segment
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find the maximum extent of the segment across all frames
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')
    
    for frame_idx, frame_data in video_segments.items():
        if segment_id in frame_data:
            mask = frame_data[segment_id][0]
            if mask.sum() > 0:  # Only consider non-empty masks
                y_coords, x_coords = np.where(mask)
                min_x = min(min_x, x_coords.min())
                max_x = max(max_x, x_coords.max())
                min_y = min(min_y, y_coords.min())
                max_y = max(max_y, y_coords.max())
    
    # Add padding
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = max_x + padding
    max_y = max_y + padding
    
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Dictionary to store cropped masks
    cropped_segments = {}
    
    # Process each frame
    for frame_file in tqdm(frame_files, desc=f"Cropping frames and masks around segment {segment_id}"):
        frame_idx = int(os.path.splitext(frame_file)[0])
        
        # Read and crop the frame
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        # Ensure crop coordinates are within frame boundaries
        final_max_y = min(max_y, frame.shape[0])
        final_max_x = min(max_x, frame.shape[1])
        
        # Crop the frame
        cropped_frame = frame[int(min_y):int(final_max_y), int(min_x):int(final_max_x)]
        
        # Save the cropped frame
        output_path = os.path.join(output_folder, f"crop_{frame_file}")
        cv2.imwrite(output_path, cropped_frame)
        
        # Crop the masks for this frame
        if frame_idx in video_segments:
            frame_masks = {}
            for obj_id, mask in video_segments[frame_idx].items():
                cropped_mask = mask[:, int(min_y):int(final_max_y), int(min_x):int(final_max_x)]
                frame_masks[obj_id] = cropped_mask
            cropped_segments[frame_idx] = frame_masks
    
    crop_info = {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'width': max_x - min_x,
        'height': max_y - min_y,
        'cropped_segments': cropped_segments
    }
    
    return crop_info


# Create temporary directory for crops
temp_dir = os.path.join("/home/lilly/phd/ria/tmp_seg", "temp_crops")
input_folder = "/home/lilly/phd/ria/data_foranalysis/riacrop/AG-MMH99_10s_20190306_02_crop"
segment_id = 4

# Crop around segment ID 1 (or whatever segment ID you want to track)
crop_info = crop_around_segment(cleaned_segments, 
                              segment_id=segment_id,
                              input_folder=input_folder,
                              output_folder=temp_dir)

print(f"Crop dimensions: {crop_info['width']}x{crop_info['height']} pixels")
#Get cropped masks
cropped_masks_loop = crop_info['cropped_segments']


# Load reference image (first frame) and corresponding mask
reference_image = cv2.imread(os.path.join(temp_dir, 'crop_000000.jpg'), cv2.IMREAD_GRAYSCALE)

# Load all frames into a list and get corresponding masks from cleaned_segments
num_frames = 600  # Total number of frames
images = []
masks = []
for i in range(num_frames):
    # Load frame
    img = cv2.imread(os.path.join(temp_dir, f'crop_{i:06d}.jpg'), cv2.IMREAD_GRAYSCALE)
    images.append(img)
    
    # Get mask for this frame from cleaned_segments
    frame_masks = cropped_masks_loop[i]
    segment_mask = frame_masks[segment_id][0]  # Get mask for chosen segment
    masks.append(segment_mask)


def align_images(reference_image, images):
    """
    Aligns a list of images to a reference image using affine transformation.

    Parameters:
        reference_image (np.ndarray): The reference grayscale image.
        images (list of np.ndarray): List of grayscale images to align.

    Returns:
        aligned_images (list of np.ndarray): List of aligned images.
        warp_matrices (list of np.ndarray): List of warp matrices used for alignment.
    """
    # Ensure the reference image is in float32
    ref_gray = reference_image.astype(np.float32)
    aligned_images = []
    warp_matrices = []

    for idx, image in enumerate(images):
        print(f"Aligning image {idx+1}/{len(images)}")
        # Convert image to float32
        img_gray = image.astype(np.float32)

        # Define the motion model (affine transformation)
        warp_mode = cv2.MOTION_AFFINE

        # Initialize the warp matrix to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

        try:
            # Run the ECC algorithm
            cc, warp_matrix = cv2.findTransformECC(
                ref_gray,
                img_gray,
                warp_matrix,
                warp_mode,
                criteria,
                inputMask=None,
                gaussFiltSize=5
            )
            # Warp the image to align with the reference
            aligned_image = cv2.warpAffine(
                image,
                warp_matrix,
                (reference_image.shape[1], reference_image.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        except cv2.error as e:
            print(f"Alignment failed for image {idx+1}: {e}")
            # If alignment fails, use the original image
            aligned_image = image.copy()

        aligned_images.append(aligned_image)
        warp_matrices.append(warp_matrix)

    return aligned_images, warp_matrices


def align_masks(masks, warp_matrices, reference_shape):
    """
    Applies the warp matrices to the corresponding masks.

    Parameters:
        masks (list of np.ndarray): List of boolean masks.
        warp_matrices (list of np.ndarray): Corresponding warp matrices.
        reference_shape (tuple): Shape of the reference image (height, width).

    Returns:
        aligned_masks (list of np.ndarray): List of aligned masks.
    """
    aligned_masks = []
    for idx, (mask, warp_matrix) in enumerate(zip(masks, warp_matrices)):
        print(f"Aligning mask {idx+1}/{len(masks)}")
        # Convert mask to float32
        mask_float = mask.astype(np.float32)
        # Warp the mask using nearest neighbor interpolation
        aligned_mask = cv2.warpAffine(
            mask_float,
            warp_matrix,
            (reference_shape[1], reference_shape[0]),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
        )
        # Convert back to boolean
        aligned_mask = aligned_mask > 0.5
        aligned_masks.append(aligned_mask)

    return aligned_masks

# Align images using affine transformation
aligned_images_loop, warp_matrices_loop = align_images(reference_image, images)

# Align masks
aligned_masks_loop = align_masks(masks, warp_matrices_loop, reference_image.shape)


def align_images_robust(reference_image, images, window_size=3):
    """
    Aligns images using spot enhancement and progressive alignment with frame recovery.
    """
    def enhance_spots(image, threshold_percentile=95):
        """Enhance bright spots in the image"""
        img = image.astype(np.float32)
        thresh = np.percentile(img, threshold_percentile)
        spots_mask = img > thresh
        distance = ndimage.distance_transform_edt(~spots_mask)
        distance = np.exp(-distance / 5.0)
        enhanced = img.copy()
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        result = (enhanced * (distance * 0.5 + 0.5)).astype(np.float32)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result

    def get_spot_mask(image, threshold_percentile=95, min_area=3):
        """
        Enhanced spot detection with local thresholding
        """
        # Global threshold
        global_thresh = np.percentile(image, threshold_percentile)
        binary = (image > global_thresh).astype(np.uint8)

        # Apply local threshold as well
        block_size = 15
        local_thresh = cv2.adaptiveThreshold(
            cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -2
        )

        # Combine global and local thresholds
        combined = cv2.bitwise_and(binary, local_thresh)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined)

        # Filter small components
        mask = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255

        return mask

    def try_alignment(ref_img, target_img, mask=None, initial_warp=None):
        """Attempt alignment with multiple parameter sets"""
        if initial_warp is None:
            initial_warp = np.eye(2, 3, dtype=np.float32)
            
        params_list = [
            {'win_size': 51, 'max_level': 2},
            {'win_size': 31, 'max_level': 1},
            {'win_size': 21, 'max_level': 0}
        ]
        
        for params in params_list:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    ref_img,
                    target_img,
                    None,
                    0.5,
                    params['max_level'],
                    params['win_size'],
                    3,
                    5,
                    1.1,
                    0
                )
                
                h, w = ref_img.shape
                y, x = np.mgrid[0:h, 0:w].reshape(2, -1)
                flow_reshape = flow.reshape(-1, 2).T
                
                src_pts = np.vstack((x, y))
                dst_pts = src_pts + flow_reshape
                
                flow_magnitude = np.sqrt(np.sum(flow_reshape**2, axis=0))
                valid_flow = flow_magnitude > np.percentile(flow_magnitude, 50)
                
                if np.sum(valid_flow) < 10:
                    continue
                
                warp_matrix = cv2.estimateAffinePartial2D(
                    src_pts[:, valid_flow].T.reshape(-1, 1, 2),
                    dst_pts[:, valid_flow].T.reshape(-1, 1, 2),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0
                )[0]
                
                if warp_matrix is not None and validate_warp_matrix(warp_matrix):
                    return warp_matrix
                    
            except Exception:
                continue
                
        return None

    def validate_warp_matrix(matrix, max_scale=1.15, max_translation=25, min_scale=0.85):
        """
        Validates the warp matrix with additional checks for non-uniform scaling
        """
        try:
            # Get the 2x2 transformation part
            A = matrix[:2, :2]

            # Compute SVD to analyze the transformation
            U, s, Vh = np.linalg.svd(A)

            # Check singular values (scales along principal axes)
            if s[0] / s[1] > 1.2 or s[1] / s[0] < 0.8:  # Max 20% difference in scaling
                return False

            # Calculate scaling factors
            scale_x = np.sqrt(matrix[0,0]**2 + matrix[0,1]**2)
            scale_y = np.sqrt(matrix[1,0]**2 + matrix[1,1]**2)

            # Check for reasonable scaling
            if (scale_x > max_scale or scale_y > max_scale or 
                scale_x < min_scale or scale_y < min_scale):
                return False

            # Check for reasonable translation
            if (abs(matrix[0,2]) > max_translation or 
                abs(matrix[1,2]) > max_translation):
                return False

            # Check for NaN or Inf values
            if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                return False

            return True
        except:
            return False


    def validate_alignment(ref_img, aligned_img, original_img, min_overlap_ratio=0.5):
        """
        Validate alignment with corner region analysis
        """
        h, w = ref_img.shape

        # Define corner regions (each covering 25% of the image dimensions)
        corner_size_h = h // 4
        corner_size_w = w // 4
        corners = [
            (slice(0, corner_size_h), slice(0, corner_size_w)),  # Top-left
            (slice(0, corner_size_h), slice(w-corner_size_w, w)),  # Top-right
            (slice(h-corner_size_h, h), slice(0, corner_size_w)),  # Bottom-left
            (slice(h-corner_size_h, h), slice(w-corner_size_w, w))  # Bottom-right
        ]

        # Check each corner region
        for corner in corners:
            ref_corner = ref_img[corner]
            aligned_corner = aligned_img[corner]

            # Get spots in corner regions
            ref_spots = get_spot_mask(ref_corner)
            aligned_spots = get_spot_mask(aligned_corner)

            # Skip corners with no spots in reference
            if np.sum(ref_spots) < 100:  # Minimum number of pixels to consider
                continue

            # Calculate overlap in corner
            overlap = cv2.bitwise_and(ref_spots, aligned_spots)
            corner_overlap_ratio = np.sum(overlap > 0) / max(np.sum(ref_spots > 0), 1)

            # Require stricter overlap in corners
            if corner_overlap_ratio < 0.4:  # 40% overlap required in corners
                return False

            # Check intensity distribution in corner
            ref_intensities = ref_corner[ref_spots > 0]
            aligned_intensities = aligned_corner[aligned_spots > 0]

            if len(ref_intensities) > 0 and len(aligned_intensities) > 0:
                # Compare mean intensities in corner
                ref_mean = np.mean(ref_intensities)
                aligned_mean = np.mean(aligned_intensities)
                if ref_mean > 0:
                    intensity_ratio = aligned_mean / ref_mean
                    if intensity_ratio < 0.7 or intensity_ratio > 1.3:
                        return False

        # Global checks
        ref_spots_global = get_spot_mask(ref_img)
        aligned_spots_global = get_spot_mask(aligned_img)

        # Calculate global overlap
        overlap_global = cv2.bitwise_and(ref_spots_global, aligned_spots_global)
        overlap_ratio = np.sum(overlap_global > 0) / max(np.sum(ref_spots_global > 0), 1)

        if overlap_ratio < min_overlap_ratio:
            return False

        return True

    def interpolate_warp_matrix(failed_idx, warp_matrices, frame_indices):
        """Interpolate transformation matrix from neighboring successful frames"""
        # Find nearest successful frames before and after
        prev_idx = None
        next_idx = None
        
        for i in range(failed_idx - 1, -1, -1):
            if i in frame_indices:
                prev_idx = i
                break
                
        for i in range(failed_idx + 1, len(warp_matrices)):
            if i in frame_indices:
                next_idx = i
                break
        
        # If we can't find both prev and next, use the nearest one we found
        if prev_idx is None and next_idx is not None:
            return warp_matrices[next_idx]
        elif next_idx is None and prev_idx is not None:
            return warp_matrices[prev_idx]
        elif prev_idx is None and next_idx is None:
            return np.eye(2, 3, dtype=np.float32)
        
        # Linear interpolation between prev and next
        prev_matrix = warp_matrices[prev_idx]
        next_matrix = warp_matrices[next_idx]
        
        # Calculate interpolation weight based on position
        weight = (failed_idx - prev_idx) / (next_idx - prev_idx)
        
        # Interpolate the matrices
        interpolated = prev_matrix * (1 - weight) + next_matrix * weight
        
        return interpolated

    # Initialize tracking variables
    aligned_images = [images[0]]
    warp_matrices = [np.eye(2, 3, dtype=np.float32)]
    last_good_warp = np.eye(2, 3, dtype=np.float32)
    alignment_stats = {
        'successful': 0,
        'failed': 0,
        'interpolated': 0,
        'successful_indices': set([0])  # Include first frame
    }
    
    # First pass: Try to align all frames
    progress_bar = tqdm(total=len(images)-1, desc="Aligning images")
    
    for idx in range(1, len(images)):
        success = False
        current_image = enhance_spots(images[idx])
        reference = enhance_spots(aligned_images[-1])
        
        warp_matrix = try_alignment(reference, current_image, initial_warp=last_good_warp)
        
        if warp_matrix is not None:
            aligned_image = cv2.warpAffine(
                images[idx],
                warp_matrix,
                (reference_image.shape[1], reference_image.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            
            if validate_alignment(reference, aligned_image, images[idx]):
                success = True
                aligned_images.append(aligned_image)
                warp_matrices.append(warp_matrix)
                last_good_warp = warp_matrix.copy()
                alignment_stats['successful'] += 1
                alignment_stats['successful_indices'].add(idx)
        
        if not success:
            # Temporarily add identity matrix as placeholder
            aligned_images.append(images[idx])
            warp_matrices.append(np.eye(2, 3, dtype=np.float32))
            alignment_stats['failed'] += 1
            
        progress_bar.update(1)
    
    # Second pass: Interpolate failed frames
    print("\nRecovering failed frames...")
    for idx in range(1, len(images)):
        if idx not in alignment_stats['successful_indices']:
            # Interpolate transformation matrix
            interpolated_matrix = interpolate_warp_matrix(
                idx, 
                warp_matrices, 
                alignment_stats['successful_indices']
            )
            
            # Apply interpolated transformation
            aligned_image = cv2.warpAffine(
                images[idx],
                interpolated_matrix,
                (reference_image.shape[1], reference_image.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            
            # Update arrays with interpolated result
            aligned_images[idx] = aligned_image
            warp_matrices[idx] = interpolated_matrix
            alignment_stats['interpolated'] += 1
    
    progress_bar.close()
    
    # Calculate summary
    processed_frames = len(images) - 1
    summary = {
        'total_frames': len(images),
        'successful_alignments': alignment_stats['successful'],
        'failed_alignments': alignment_stats['failed'],
        'interpolated_frames': alignment_stats['interpolated'],
        'success_rate': (alignment_stats['successful'] / processed_frames) * 100
    }
    
    print("\nAlignment Summary:")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Successfully aligned: {summary['successful_alignments']} ({summary['success_rate']:.1f}%)")
    print(f"Failed alignments: {summary['failed_alignments']}")
    print(f"Interpolated frames: {summary['interpolated_frames']}")
    
    return aligned_images, warp_matrices, summary


def align_masks_robust(masks, warp_matrices, reference_shape, max_scale=1.2, max_translation=30):
    """
    Aligns binary segmentation masks using pre-computed warp matrices.
    """
    def is_near_identity(matrix, tolerance=1e-10):
        """Check if matrix is close to identity matrix"""
        return np.allclose(matrix, np.eye(2, 3), atol=tolerance)

    def validate_warp_matrix(matrix):
        """Validates the warp matrix with strict constraints"""
        try:
            # Extract scale components
            scale_x = np.sqrt(matrix[0,0]**2 + matrix[0,1]**2)
            scale_y = np.sqrt(matrix[1,0]**2 + matrix[1,1]**2)
            
            # Check scaling
            if scale_x > max_scale or scale_y > max_scale or scale_x < 1/max_scale or scale_y < 1/max_scale:
                return False
                
            # Check translation
            if abs(matrix[0,2]) > max_translation or abs(matrix[1,2]) > max_translation:
                return False
            
            # Check for invalid values
            if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                return False
            
            return True
        except:
            return False

    def check_mask_quality(original_mask, aligned_mask, min_area_ratio=0.7):
        """Enhanced mask quality check with fixed scalar comparison"""
        try:
            # Convert masks to binary (0 or 1)
            original_mask_bin = (original_mask > 0).astype(np.uint8)
            aligned_mask_bin = (aligned_mask > 0).astype(np.uint8)
            
            # Calculate areas
            original_area = float(np.sum(original_mask_bin))
            if original_area == 0:
                return True, "empty_mask"
                
            aligned_area = float(np.sum(aligned_mask_bin))
            area_ratio = aligned_area / original_area
            
            # Check area ratio
            if area_ratio < min_area_ratio:
                return False, f"area_ratio_{area_ratio:.3f}"
                
            # Connected components analysis
            # Note: second return value is already a scalar integer
            _, orig_num = cv2.connectedComponents(original_mask_bin)
            _, align_num = cv2.connectedComponents(aligned_mask_bin)

            # Get maximum values from the arrays
            orig_num_max = np.max(orig_num)
            align_num_max = np.max(align_num)
            
            # Compare number of components (they're already scalars)
            if align_num_max > (orig_num_max * 1.5):
                return False, f"fragmented_{align_num}_{orig_num}"
            
            return True, "valid"
            
        except Exception as e:
            return False, f"quality_check_error: {str(e)}"

    aligned_masks = []
    mask_stats = {
        'successful': 0,
        'warnings': 0,
        'warning_indices': [],
        'warning_reasons': [],
        'warning_details': {},
        'interpolated': 0
    }
    
    # Add first mask unchanged
    aligned_masks.append(masks[0].astype(bool))
    mask_stats['successful'] += 1
    
    progress_bar = tqdm(total=len(masks)-1, desc="Aligning masks")
    
    # Process remaining masks
    for idx in range(1, len(masks)):
        try:
            current_mask = masks[idx]
            warp_matrix = warp_matrices[idx]
            warning_messages = []
            
            # Ensure mask is binary
            current_mask_bin = (current_mask > 0)
            
            # Check if transformation was interpolated
            is_interpolated = False
            if is_near_identity(warp_matrix):
                warning_messages.append('identity_transform')
            elif not validate_warp_matrix(warp_matrix):
                warning_messages.append('invalid_transform')
            else:
                # Check if this was an interpolated frame by looking at matrix values
                if idx > 1 and idx < len(warp_matrices) - 1:
                    prev_diff = np.abs(warp_matrix - warp_matrices[idx - 1]).max()
                    next_diff = np.abs(warp_matrix - warp_matrices[idx + 1]).max()
                    if prev_diff < 0.1 and next_diff < 0.1:  # Small difference from neighbors
                        is_interpolated = True
                        mask_stats['interpolated'] += 1
            
            # Apply the transformation using nearest neighbor interpolation
            aligned_mask = cv2.warpAffine(
                current_mask_bin.astype(np.uint8),
                warp_matrix,
                (reference_shape[1], reference_shape[0]),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Convert back to boolean
            aligned_mask = aligned_mask > 0
            
            # Check mask quality
            is_valid, quality_message = check_mask_quality(current_mask_bin, aligned_mask)
            if not is_valid:
                warning_messages.append(quality_message)
            
            # Store the aligned mask
            aligned_masks.append(aligned_mask)
            
            # Update statistics
            if warning_messages:
                mask_stats['warnings'] += 1
                mask_stats['warning_indices'].append(idx)
                mask_stats['warning_reasons'].extend(warning_messages)
                mask_stats['warning_details'][idx] = {
                    'messages': warning_messages,
                    'interpolated': is_interpolated
                }
            else:
                mask_stats['successful'] += 1
            
        except Exception as e:
            warning_messages = [f'error: {str(e)}']
            aligned_masks.append(current_mask)  # Use original mask on error
            mask_stats['warnings'] += 1
            mask_stats['warning_indices'].append(idx)
            mask_stats['warning_reasons'].extend(warning_messages)
            mask_stats['warning_details'][idx] = {
                'messages': warning_messages,
                'interpolated': False
            }
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate summary
    processed_masks = len(masks) - 1
    mask_summary = {
        'total_masks': len(masks),
        'processed_masks': processed_masks,
        'successful_alignments': mask_stats['successful'],
        'masks_with_warnings': mask_stats['warnings'],
        'interpolated_masks': mask_stats['interpolated'],
        'success_rate': (mask_stats['successful'] / len(masks)) * 100,
        'warning_frame_indices': mask_stats['warning_indices'],
        'warning_reasons': mask_stats['warning_reasons'],
        'warning_details': mask_stats['warning_details']
    }
    
    # Print summary
    print("\nMask Alignment Summary:")
    print(f"Total masks: {mask_summary['total_masks']}")
    print(f"Masks processed: {mask_summary['processed_masks']} (excluding first reference mask)")
    print(f"Successfully aligned: {mask_summary['successful_alignments']} ({mask_summary['success_rate']:.1f}%)")
    print(f"Masks with warnings: {mask_summary['masks_with_warnings']}")
    print(f"Interpolated masks: {mask_summary['interpolated_masks']}")
    
    if mask_summary['masks_with_warnings'] > 0:
        reason_counts = {}
        for reason in mask_stats['warning_reasons']:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print("\nWarning reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} masks")
        
        if len(mask_stats['warning_indices']) <= 10:
            print(f"\nFrames with warnings: {mask_stats['warning_indices']}")
        else:
            print(f"\nFirst 10 frames with warnings: {mask_stats['warning_indices'][:10]}...")
    
    return aligned_masks, mask_summary


#loop
raligned_images_loop, rwarp_matrices_loop, summary = align_images_robust(
    reference_image, 
    images, 
    window_size=3
)

raligned_masks_loop, rmask_summary = align_masks_robust(
    masks,                  # Your binary masks
    rwarp_matrices_loop,      # From successful image alignment
    reference_image.shape  # Original image dimensions
)

#nrD
aligned_images_nrd, warp_matrices_nrd, summary = align_images_robust(
    reference_image, 
    images, 
    window_size=5, 
    max_tries=10, 
    use_multiscale=True
)

aligned_masks_nrd, mask_summary = align_masks_robust(
    masks,                  # Your binary masks
    warp_matrices_nrd,      # From successful image alignment
    reference_image.shape,  # Original image dimensions
    max_scale=1.5,         # Adjust if needed
    max_translation=50      # Adjust if needed
)

#nrV
aligned_images_nrv, warp_matrices_nrv, summary = align_images_robust(
    reference_image, 
    images, 
    window_size=5, 
    max_tries=10, 
    use_multiscale=True
)

aligned_masks_nrv, mask_summary = align_masks_robust(
    masks,                  # Your binary masks
    warp_matrices_nrv,      # From successful image alignment
    reference_image.shape,  # Original image dimensions
    max_scale=1.5,         # Adjust if needed
    max_translation=50      # Adjust if needed
)

# Save frame 23 from aligned images
cv2.imwrite('/home/lilly/phd/ria/overlay_tst/aligned_frame23.png', aligned_images_nrd[23])



# Save images with masks overlaid
for idx, (img, mask) in enumerate(zip(aligned_images_loop, aligned_masks_loop)):
    # Convert grayscale to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Create a red overlay for the mask
    mask_overlay = np.zeros_like(img_rgb)
    mask_overlay[mask] = [0, 0, 255]  # Red color
    
    # Blend image and mask
    alpha = 0.8  # Transparency of the mask overlay
    overlaid_img = cv2.addWeighted(img_rgb, 1, mask_overlay, alpha, 0)
    
    # Save overlaid image
    cv2.imwrite(f'/home/lilly/phd/ria/overlay_tst/aligned_frame{idx}_with_mask.png', overlaid_img)


# Create video from aligned images with masks
output_path = '/home/lilly/phd/ria/overlay_tst/aligned_frames_video_loop.mp4'
height, width = aligned_images_loop[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Write frames to video
for img, mask in zip(aligned_images_loop, aligned_masks_loop):
    # Convert grayscale to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Create a red overlay for the mask
    mask_overlay = np.zeros_like(img_rgb)
    mask_overlay[mask] = [0, 0, 255]  # Red color
    
    # Blend image and mask
    alpha = 0.98  # Transparency of the mask overlay
    overlaid_img = cv2.addWeighted(img_rgb, 1, mask_overlay, alpha, 0)
    
    out.write(overlaid_img)

# Release video writer
out.release()


###Make a consensus mask
def create_average_mask(aligned_masks: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Creates an average mask from a list of aligned boolean masks.
    
    Parameters:
        aligned_masks: List of boolean numpy arrays representing aligned masks
        threshold: Frequency threshold for including a pixel in the final mask (0.0 to 1.0)
    
    Returns:
        Binary mask where pixels are True if they appear in at least threshold% of frames
    """
    # Convert boolean masks to numeric and sum
    mask_sum = np.zeros_like(aligned_masks[0], dtype=float)
    for mask in aligned_masks:
        mask_sum += mask.astype(float)
    
    # Calculate average (frequency of each pixel being True)
    mask_avg = mask_sum / len(aligned_masks)
    
    # Create binary mask using threshold
    final_mask = mask_avg >= threshold
    
    return final_mask

average_mask_loop = create_average_mask(aligned_masks_loop, threshold=0.80)
average_mask_nrd = create_average_mask(aligned_masks_nrd, threshold=0.10)
average_mask_nrv = create_average_mask(aligned_masks_nrv, threshold=0.20)
# Save average mask as PNG
cv2.imwrite('/home/lilly/phd/ria/overlay_tst/average_mask_nrd.png', average_mask_nrd.astype(np.uint8) * 255)


def visualize_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """
    Creates a visualization of the mask overlaid on the image.
    
    Parameters:
        image: Grayscale image
        mask: Binary mask
        alpha: Transparency of the overlay (0.0 to 1.0)
    
    Returns:
        RGB image with colored mask overlay
    """
    # Create RGB image from grayscale
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create colored mask overlay (using red)
    overlay = np.zeros_like(rgb_image)
    overlay[mask] = [0, 0, 255]  # Red color for mask
    
    # Blend images
    output = cv2.addWeighted(rgb_image, 1, overlay, alpha, 0)
    
    return output


first_frame_overlay = visualize_mask_overlay(aligned_images_nrv[450], average_mask_nrv)
cv2.imwrite('/home/lilly/phd/ria/overlay_tst/first_frame_overlay.png', first_frame_overlay)



# Create video with average mask overlay
output_path_avg = '/home/lilly/phd/ria/overlay_tst/aligned_frames_with_average_mask_loop.mp4'
height, width = aligned_images_loop[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out_avg = cv2.VideoWriter(output_path_avg, fourcc, fps, (width, height))

# Write frames to video using average mask
for img in aligned_images_loop:
    # Create overlay using the average mask
    overlaid_img = visualize_mask_overlay(img, average_mask_loop, alpha=0.98)
    out_avg.write(overlaid_img)

# Release video writer
out_avg.release()


#Get mean and std of brightness values from masked regions using average mask
def extract_brightness_values(aligned_images: List[np.ndarray], 
                            average_mask: np.ndarray) -> pd.DataFrame:
    """
    Extracts mean and standard deviation of brightness values from masked regions.
    
    Parameters:
        aligned_images: List of grayscale images (numpy arrays)
        average_mask: Binary mask to apply to all images
    
    Returns:
        DataFrame containing frame number, mean and std brightness values for each frame
    """
    data = []
    
    for frame_idx, image in enumerate(aligned_images):
        # Extract pixels within mask
        masked_pixels = image[average_mask]
        
        # Calculate statistics
        mean_val = np.mean(masked_pixels)
        std_val = np.std(masked_pixels)
        
        data.append({
            'frame': frame_idx,
            'mean_brightness': mean_val,
            'std_brightness': std_val
        })
    
    return pd.DataFrame(data)



brightness_df = extract_brightness_values(aligned_images_loop, average_mask_loop)
brightness_nrd = extract_brightness_values(aligned_images_nrd, average_mask_nrd)
brightness_nrv = extract_brightness_values(aligned_images_nrv, average_mask_nrv)


def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

normalized_brightness = normalize(brightness_df['mean_brightness'])
normalized_brightness_nrd = normalize(brightness_nrd['mean_brightness'])
normalized_brightness_nrv = normalize(brightness_nrv['mean_brightness'])

fiji_data = pd.read_excel('/home/lilly/phd/ria/MMH99_10s_20190306_02.xlsx')
# Shift Frame values down by 1 to start at 0
fiji_data['Frame'] = fiji_data['Frame'] - 1

# Normalize the columns nrD, nrV and loop
columns_to_normalize = ['nrD', 'nrV', 'loop']
for col in columns_to_normalize:
    norm_col_name = f"{col}_normalized"
    fiji_data[norm_col_name] = normalize(fiji_data[col])

# Calculate and normalize the difference between loop and nrV
fiji_data['loop_minus_nrv'] = fiji_data['loop'] - fiji_data['nrV']
fiji_data['loop_minus_nrv_norm'] = normalize(fiji_data['loop_minus_nrv'])  # Changed name to avoid error

nonaligned_nrd = normalize(df_wide_bg_corrected['2'])
nonaligned_nrv = normalize(df_wide_bg_corrected['3'])
nonaligned_loop = normalize(df_wide_bg_corrected['4'])

nonaligned_loop_distance = normalize(df_wide_bg_corrected_distance[4])

#Smoothing
def median_filter_with_edges(df, column_name, kernel_size=11, edge_method='mirror', plot=True):
    """
    Apply median filter with proper edge handling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column_name : str
        Name of the column to smooth
    kernel_size : int
        Size of the median filter kernel (must be odd)
    edge_method : str
        Method to handle edges: 'mirror', 'nearest', or 'auto'
    plot : bool
        Whether to plot the results
        
    Returns:
    --------
    pandas.Series
        Smoothed data series
    """
    data = df[column_name].values
    pad_size = kernel_size // 2
    
    if edge_method == 'mirror':
        # Mirror the edges
        padded_data = np.pad(data, pad_size, mode='reflect')
    elif edge_method == 'nearest':
        # Repeat the edge values
        padded_data = np.pad(data, pad_size, mode='edge')
    elif edge_method == 'auto':
        # Use smaller kernel sizes at the edges
        result = np.copy(data)
        
        # Apply progressively larger filters near the edges
        for i in range(pad_size):
            current_kernel = 2 * i + 3  # Make sure kernel is odd
            
            # Handle left edge
            result[i] = np.median(data[0:current_kernel])
            
            # Handle right edge
            result[-(i+1)] = np.median(data[-current_kernel:])
        
        # Apply full kernel to the middle
        result[pad_size:-pad_size] = medfilt(data, kernel_size)[pad_size:-pad_size]
        
        smoothed_data = result
    else:
        raise ValueError(f"Unknown edge_method: {edge_method}")
    
    # Apply median filter (for 'mirror' and 'nearest' methods)
    if edge_method in ['mirror', 'nearest']:
        smoothed_data = medfilt(padded_data, kernel_size)[pad_size:-pad_size]
    
    if plot:
        plt.figure(figsize=(15, 8))
        
        # Create zoom windows for start and end
        num_edge_points = min(50, len(data) // 4)
        
        # Main plot
        plt.subplot(2, 2, (1, 2))
        plt.plot(df.index, data, 'b.', label='Original', alpha=0.5)
        plt.plot(df.index, smoothed_data, 'r-', 
                label=f'Median Filter ({edge_method} edges)', linewidth=2)
        plt.title(f'Full Time Series - {edge_method} edge handling')
        plt.legend()
        plt.grid(True)
        
        # Start zoom
        plt.subplot(2, 2, 3)
        plt.plot(df.index[:num_edge_points], data[:num_edge_points], 
                'b.', label='Original', alpha=0.5)
        plt.plot(df.index[:num_edge_points], smoothed_data[:num_edge_points], 
                'r-', label='Smoothed', linewidth=2)
        plt.title('Start of Series (Zoomed)')
        plt.grid(True)
        
        # End zoom
        plt.subplot(2, 2, 4)
        plt.plot(df.index[-num_edge_points:], data[-num_edge_points:], 
                'b.', label='Original', alpha=0.5)
        plt.plot(df.index[-num_edge_points:], smoothed_data[-num_edge_points:], 
                'r-', label='Smoothed', linewidth=2)
        plt.title('End of Series (Zoomed)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return pd.Series(smoothed_data, index=df.index, name=f'{column_name}_smoothed')

# Example usage with different edge handling methods:
"""
# Create sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
data = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.2, 1000)
# Add some edge noise
data[0] += 3
data[-1] += -3
df = pd.DataFrame({'value': data}, index=dates)

# Apply different edge handling methods
smoothed_mirror = median_filter_with_edges(df, 'value', kernel_size=11, edge_method='mirror')
smoothed_nearest = median_filter_with_edges(df, 'value', kernel_size=11, edge_method='nearest')
smoothed_auto = median_filter_with_edges(df, 'value', kernel_size=11, edge_method='auto')
"""

def compare_edge_methods(df, column_name, kernel_size=11):
    """
    Compare different edge handling methods
    """
    methods = ['mirror', 'nearest', 'auto']
    results = {}
    
    plt.figure(figsize=(15, 10))
    
    # Full series plot
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df[column_name], 'k.', label='Original', alpha=0.3)
    
    for method in methods:
        smoothed = median_filter_with_edges(df, column_name, 
                                          kernel_size=kernel_size,
                                          edge_method=method,
                                          plot=False)
        results[method] = smoothed
        plt.plot(df.index, smoothed, '-', label=f'{method}', linewidth=2)
    
    plt.title('Comparison of Edge Handling Methods - Full Series')
    plt.legend()
    plt.grid(True)
    
    # Zoom to edges
    plt.subplot(2, 1, 2)
    num_edge_points = min(20, len(df) // 10)
    
    # Plot start and end regions
    for i, idx_slice in enumerate([slice(0, num_edge_points), 
                                 slice(-num_edge_points, None)]):
        if i == 0:
            plt.plot(df.index[idx_slice], df[column_name].iloc[idx_slice], 
                    'k.', label='Original', alpha=0.3)
        else:
            plt.plot(df.index[idx_slice], df[column_name].iloc[idx_slice], 
                    'k.', alpha=0.3)
            
        for method in methods:
            if i == 0:
                plt.plot(df.index[idx_slice], results[method].iloc[idx_slice], 
                        '-', label=f'{method}', linewidth=2)
            else:
                plt.plot(df.index[idx_slice], results[method].iloc[idx_slice], 
                        '-', linewidth=2)
    
    plt.title('Edge Behavior (Zoomed to First and Last Points)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results


smoothed__alignedloop_auto = median_filter_with_edges(brightness_df, 'mean_brightness', kernel_size=7, edge_method='auto', plot=False)
smoothed__nonalignedloop_auto = median_filter_with_edges(df_wide_bg_corrected, '4', kernel_size=5, edge_method='auto', plot=False)
smoothed__nonalignednrv_auto = median_filter_with_edges(df_wide_bg_corrected, '3', kernel_size=5, edge_method='auto', plot=False)
smoothed__nonalignednrd_auto = median_filter_with_edges(df_wide_bg_corrected, '2', kernel_size=5, edge_method='auto', plot=False)
smoothed__nonalignedloopdistance_auto = median_filter_with_edges(df_wide_bg_corrected_distance, 4, kernel_size=5, edge_method='auto', plot=False)

# Create plot of normalized brightness over frames with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(40, 24))

# Define colors for each segment type
loop_color = 'blue'
nrv_color = 'red' 
nrd_color = 'green'

# Top subplot - loop lines only
#ax1.plot(brightness_df['frame'].to_numpy(), normalized_brightness.to_numpy(), 
#         color=loop_color, linestyle='-', linewidth=2, label='Aligned Loop')
ax1.plot(brightness_df['frame'].to_numpy(), normalize(smoothed__alignedloop_auto).to_numpy(),
         color=loop_color, linestyle='--', linewidth=2, label='Smoothed Aligned Loop')
ax1.plot(range(len(smoothed__nonalignedloop_auto)), normalize(smoothed__nonalignedloop_auto).to_numpy(),
         color='orange', linestyle='--', linewidth=2, label='Smoothed Nonaligned Loop')
ax1.plot(range(len(smoothed__nonalignednrv_auto)), normalize(smoothed__nonalignednrv_auto).to_numpy(),
         color='red', linestyle='--', linewidth=2, label='Smoothed Nonaligned nrV')
ax1.plot(fiji_data['Frame'].to_numpy(), fiji_data['loop_normalized'].to_numpy(),
         linewidth=2, label='Fiji Loop')

#ax1.plot(fiji_data['Frame'].to_numpy(), fiji_data['loop_minus_nrv_norm'].to_numpy(),
#         linewidth=2, label='Fiji Loop-nrV')
#ax1.plot(range(len(nonaligned_loop)), nonaligned_loop.to_numpy(),
#         color='orange', linestyle='-', linewidth=2, label='Nonaligned Loop')
ax1.plot(range(len(smoothed__nonalignedloopdistance_auto)), normalize(smoothed__nonalignedloopdistance_auto).to_numpy(), linestyle='-', linewidth=2, label='Smoothed Nonaligned Loop Distance')
#ax1.plot(range(len(nonaligned_nrv)), nonaligned_nrv.to_numpy(),
#         linewidth=2, label='Nonaligned nrV')
ax1.set_xlabel('Frame Number', fontsize=14)
ax1.set_ylabel('Normalized Brightness', fontsize=14)
ax1.set_title('Normalized Brightness Over Frames (Loop)', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks(np.arange(0, len(nonaligned_loop), 10))
ax1.tick_params(axis='both', labelsize=12)
ax1.legend(fontsize=12)
ax1.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Middle subplot - nrV lines
#ax2.plot(brightness_nrv['frame'].to_numpy(), normalized_brightness_nrv.to_numpy(), 
#         linewidth=2, label='Aligned nrV')
ax2.plot(fiji_data['Frame'].to_numpy(), fiji_data['nrV_normalized'].to_numpy(),
         linewidth=2, label='Fiji nrV')
ax2.plot(range(len(nonaligned_nrv)), nonaligned_nrv.to_numpy(),
         color=nrv_color, linestyle='-', linewidth=2, label='Nonaligned nrV')
ax2.plot(range(len(smoothed__nonalignednrv_auto)), normalize(smoothed__nonalignednrv_auto).to_numpy(),
         color='red', linestyle='--', linewidth=2, label='Smoothed Nonaligned nrV')
ax2.set_xlabel('Frame Number', fontsize=14)
ax2.set_ylabel('Normalized Brightness', fontsize=14)
ax2.set_title('Normalized Brightness Over Frames (nrV)', fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks(np.arange(0, len(nonaligned_nrv), 10))
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(fontsize=12)
ax2.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Bottom subplot - nrD lines
#ax3.plot(brightness_nrd['frame'].to_numpy(), normalized_brightness_nrd.to_numpy(), 
#         linewidth=2, label='Aligned nrD')
ax3.plot(fiji_data['Frame'].to_numpy(), fiji_data['nrD_normalized'].to_numpy(),
         linewidth=2, label='Fiji nrD')
ax3.plot(range(len(nonaligned_nrd)), nonaligned_nrd.to_numpy(),
         color=nrd_color, linestyle='-', linewidth=2, label='Nonaligned nrD')
ax3.plot(range(len(smoothed__nonalignednrd_auto)), normalize(smoothed__nonalignednrd_auto).to_numpy(),
         color='green', linestyle='--', linewidth=2, label='Smoothed Nonaligned nrD')
ax3.set_xlabel('Frame Number', fontsize=14)
ax3.set_ylabel('Normalized Brightness', fontsize=14)
ax3.set_title('Normalized Brightness Over Frames (nrD)', fontsize=18)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_xticks(np.arange(0, len(nonaligned_nrd), 10))
ax3.tick_params(axis='both', labelsize=12)
ax3.legend(fontsize=12)
ax3.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax3.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax3.axvline(x=500, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('normalized_brightness_over_frames_ria9902.png', dpi=300, bbox_inches='tight')
plt.close()







def create_mask_video(image_dir, masks_dict, output_path, fps=10, alpha=0.99):
    """
    Create a video with mask overlays from a directory of images and a dictionary of masks.
    
    Args:
        image_dir (str): Directory containing the input images
        masks_dict (dict): Dictionary where keys are frame indices and values are
                          dictionaries of mask_id: mask pairs for that frame
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        alpha (float): Transparency of the mask overlay (0-1)
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

    def overlay_masks(image, frame_masks, mask_colors, alpha):
        """Helper function to overlay masks on an image"""
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
        
        # Combine with original image
        return cv2.addWeighted(image, 1, overlay, alpha, 0)

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
    all_mask_ids = set()
    for masks in masks_dict.values():
        all_mask_ids.update(masks.keys())
    mask_colors = {mask_id: COLORS[i % len(COLORS)] 
                  for i, mask_id in enumerate(all_mask_ids)}

    # Process each frame
    for frame_idx, image_file in enumerate(image_files):
        try:
            # Read image
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply masks if available for this frame
            if frame_idx in masks_dict:
                frame = overlay_masks(frame, masks_dict[frame_idx], 
                                   mask_colors, alpha)

            # Write frame
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue

    # Clean up
    out.release()
    print(f"Video saved to {output_path}")

# Example usage:
"""
image_dir = "/home/lilly/phd/ria/data_foranalysis/riacrop/AG-MMH99_10s_20190306_02_crop"
masks_dict = cleaned_segments
output_path = "cleaned_segments_video.mp4"

create_mask_video(image_dir, masks_dict, output_path, fps=10, alpha=1)
"""