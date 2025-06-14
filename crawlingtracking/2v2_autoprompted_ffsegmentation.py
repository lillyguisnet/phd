import os
import sys
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from scipy.ndimage import binary_dilation
import tqdm
sys.path.append("/home/lilly/phd/segment-anything-2")
from sam2.build_sam import build_sam2_video_predictor
import h5py
import random
import concurrent.futures
import multiprocessing

# Configure CUDA settings
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Initialize predictor
sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


def _process_frame_chunk(args):
    """Helper function for multiprocessing pool to process a chunk of frames."""
    frames_chunk, start_h5_idx, object_ids, segments_chunk, mask_shape = args
    chunk_masks_by_obj = {obj_id: [] for obj_id in object_ids}
    for frame_idx in frames_chunk:
        objects_in_frame = segments_chunk.get(frame_idx, {})
        for obj_id in object_ids:
            mask = objects_in_frame.get(obj_id, np.zeros(mask_shape, dtype=bool))
            chunk_masks_by_obj[obj_id].append(mask)
    return start_h5_idx, chunk_masks_by_obj


def _process_frame_for_video(args):
    """Helper function to process a single frame for video creation."""
    frame_idx, frame_name, video_dir, masks_for_frame, colors, alpha, new_size = args
    
    image_path = os.path.join(video_dir, frame_name)
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image if needed
        if new_size != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        if masks_for_frame:
            # Create a blank overlay
            overlay = np.zeros_like(image)
            
            # Process each mask
            for mask_id, mask in masks_for_frame.items():
                if mask_id is None: 
                    continue
                # Convert mask to binary numpy array if it's not already
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                if mask.dtype != bool:
                    mask = mask > 0.5
                
                # Ensure mask is 2D
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                # Resize the mask to match the image dimensions
                mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Get the color for this mask
                color = colors.get(mask_id)
                if color is None:
                    continue

                # Create a colored mask
                colored_mask = np.zeros_like(image)
                colored_mask[mask_resized == 1] = color
                
                # Add the colored mask to the overlay
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
            
            # Overlay the masks on the image
            overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
        else:
             # If no segmentation for this frame, use the original image (already resized)
            overlaid_image = image

        # Return in BGR format for cv2.VideoWriter
        return cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error processing frame {frame_idx}: {str(e)}")
        # If there's an error, write the original frame
        original_frame = cv2.imread(image_path)
        if original_frame is not None:
            if new_size != (original_frame.shape[1], original_frame.shape[0]):
                original_frame = cv2.resize(original_frame, new_size, interpolation=cv2.INTER_AREA)
            return original_frame
        
        print(f"Could not read original frame {frame_idx}, returning blank frame.")
        return np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)


def show_mask(mask, ax, obj_id=None, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else \
            np.array([*plt.get_cmap("tab10")(0 if obj_id is None else obj_id)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=26):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def add_prompt_frames_to_video(video_dir, prompt_dir):
    existing_frames = [f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    last_frame_num = max(int(os.path.splitext(f)[0]) for f in existing_frames)

    prompt_frames = sorted([f for f in os.listdir(prompt_dir) if f.lower().endswith(('.jpg', '.jpeg'))],
                           key=lambda x: int(os.path.splitext(x)[0]))  # Sort by frame number

    frame_mapping = {}
    for i, prompt_frame in enumerate(prompt_frames, start=1):
        new_frame_num = last_frame_num + i
        new_frame_name = f"{new_frame_num:06d}.jpg"
        shutil.copy(os.path.join(prompt_dir, prompt_frame), os.path.join(video_dir, new_frame_name))
        frame_mapping[new_frame_num] = int(os.path.splitext(prompt_frame)[0])  # Store original frame number

    final_frame_count = last_frame_num + len(prompt_frames) + 1
    print(f"Added {len(prompt_frames)} prompt frames to the video directory. There are now {final_frame_count} frames in the video directory.")
    return frame_mapping

def remove_prompt_frames_from_video(video_dir, frame_mapping):
    """
    Remove the added prompt frames from the video directory.
    
    :param video_dir: Directory containing the video frames
    :param frame_mapping: Dictionary mapping frame numbers to original prompt frame names
    """
    for frame_num in frame_mapping.keys():
        frame_name = f"{frame_num:06d}.jpg"
        frame_path = os.path.join(video_dir, frame_name)
        if os.path.exists(frame_path):
            os.remove(frame_path)
    
    print(f"Removed {len(frame_mapping)} prompt frames from the video directory.")

def add_prompts(inference_state, frame_idx, obj_id, points, labels):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels
    )
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, f"{frame_idx:06d}.jpg")))
    show_points(points, labels, plt.gca())
    
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    
    plt.savefig(f"prompt_frame.png")
    plt.close()
    
def check_overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    overlap_pixels = np.sum(intersection)
    iou = overlap_pixels / np.sum(union) if np.sum(union) > 0 else 0
    return overlap_pixels > 0, iou, overlap_pixels

def check_distance(mask1, mask2, max_distance=10):
    # First check if either mask is empty
    if mask1.sum() == 0 or mask2.sum() == 0:
        return True  # Return True to indicate masks are not distant (they're empty)
    
    # If neither mask is empty, proceed with distance check
    dilated_mask1 = binary_dilation(mask1, iterations=max_distance)
    dilated_mask2 = binary_dilation(mask2, iterations=max_distance)
    
    # Check if the dilated masks overlap
    return np.any(np.logical_and(dilated_mask1, dilated_mask2))

def analyze_masks(video_segments):
    results = {'empty': {}, 'high': {}, 'overlapping': {}, 'distant': {}}
    max_counts = {'empty': 0, 'high': 0, 'overlapping': 0, 'distant': 0}
    max_frames = {'empty': None, 'high': None, 'overlapping': None, 'distant': None}

    for frame, mask_dict in video_segments.items():
        mask_ids = [mask_id for mask_id in mask_dict.keys() if mask_id is not None]
        
        # Track empty masks first
        empty_masks = set()  # Keep track of which masks are empty
        for mask_id in mask_ids:
            mask = mask_dict[mask_id]
            if mask is not None:
                mask_sum = mask.sum()
                if mask_sum == 0:
                    results['empty'].setdefault(frame, []).append(mask_id)
                    empty_masks.add(mask_id)
                elif mask_sum >= 40000:
                    results['high'].setdefault(frame, []).append(mask_id)

        # Process overlaps
        for i in range(len(mask_ids)):
            mask_id = mask_ids[i]
            if mask_id in empty_masks:  # Skip empty masks for overlap checking
                continue
                
            mask = mask_dict[mask_id]
            if mask is not None:
                # Check for overlaps with other masks in the same frame
                for j in range(i + 1, len(mask_ids)):
                    other_mask_id = mask_ids[j]
                    if other_mask_id in empty_masks:  # Skip empty masks
                        continue
                        
                    other_mask = mask_dict[other_mask_id]
                    if other_mask is not None:
                        is_overlapping, iou, overlap_pixels = check_overlap(mask, other_mask)
                        if is_overlapping:
                            results['overlapping'].setdefault(frame, []).append((mask_id, other_mask_id, iou, overlap_pixels))
        
        # Check distance between object 3 and 4, but only if neither is empty
        if (3 in mask_dict and 4 in mask_dict and 
            mask_dict[3] is not None and mask_dict[4] is not None and 
            3 not in empty_masks and 4 not in empty_masks):  # Added empty mask check
            if not check_distance(mask_dict[3], mask_dict[4]):
                results['distant'].setdefault(frame, []).append((3, 4))

        # Update max counts and frames
        for category in ['empty', 'high', 'overlapping', 'distant']:
            if frame in results[category]:
                count = len(results[category][frame])
                if count > max_counts[category]:
                    max_counts[category] = count
                    max_frames[category] = frame

    return results, max_counts, max_frames

def collect_results(result_dict, condition, max_count, max_frame):
    detailed_output = []
    summary_output = []
    
    if result_dict:
        detailed_output.append(f"!!! Frames with masks {condition}:")
        for frame, data in result_dict.items():
            if condition == "overlapping":
                overlap_info = [f"{a}-{b} ({iou:.2%}, {pixels} pixels)" for a, b, iou, pixels in data]
                detailed_output.append(f"  Frame {frame}: Overlapping Mask ID pairs {', '.join(overlap_info)}")
            elif condition == "distant":
                detailed_output.append(f"  Frame {frame}: Distant Mask ID pairs {data}")
            else:
                detailed_output.append(f"  Frame {frame}: Mask IDs {data}")
        if max_count > 0:
            summary_output.append(f"Latest frame with highest number of {condition} masks: {max_frame} (Count: {max_count})")
    else:
        summary_output.append(f"Yay! No masks {condition} found!")
    
    return detailed_output, summary_output

def analyze_and_print_results(video_segments):
    # Perform the analysis
    analysis_results, max_counts, max_frames = analyze_masks(video_segments)

    all_detailed_outputs = []
    all_summary_outputs = []
    problematic_frame_counts = {
        'empty': 0,
        'high': 0,
        'overlapping': 0,
        'distant': 0
    }

    # Get total number of frames
    total_frames = len(video_segments)

    # Collect results for each category
    for category in ['empty', 'high', 'overlapping', 'distant']:
        detailed, summary = collect_results(analysis_results[category], category, max_counts[category], max_frames[category])
        all_detailed_outputs.extend(detailed)
        all_summary_outputs.extend(summary)
        problematic_frame_counts[category] = len(analysis_results[category])

    # Print all the detailed outputs first
    for line in all_detailed_outputs:
        print(line)

    # Then print all the summary outputs
    for line in all_summary_outputs:
        print(line)

    # Print the number of problematic frames for each category
    print("\nNumber of problematic frames:")
    for category, count in problematic_frame_counts.items():
        percentage = (count / total_frames) * 100
        print(f"Frames with {category} masks: {count} out of {total_frames} ({percentage:.2f}%)")

    # Calculate and print the total number of unique problematic frames
    unique_problematic_frames = set()
    for category, frames in analysis_results.items():
        unique_problematic_frames.update(frames.keys())
    
    unique_problematic_count = len(unique_problematic_frames)
    unique_problematic_percentage = (unique_problematic_count / total_frames) * 100
    print(f"\nTotal number of unique problematic frames: {unique_problematic_count} out of {total_frames} ({unique_problematic_percentage:.2f}%)")

    # Additional statistics
    if video_segments:
        objects_per_frame = [len([obj for obj in segments.keys() if obj is not None]) for segments in video_segments.values()]
        if objects_per_frame:
            avg_objects_per_frame = sum(objects_per_frame) / len(objects_per_frame)
            print(f"\nAverage number of objects per frame: {avg_objects_per_frame:.2f}")
            print(f"Minimum objects in a frame: {min(objects_per_frame)}")
            print(f"Maximum objects in a frame: {max(objects_per_frame)}")
        else:
            print("\nNo objects found to analyze.")
    else:
        print("\nNo segments to analyze for statistics.")

def create_mask_overlay_video(video_dir, frame_names, video_segments, output_video_path, fps=10, alpha=0.99, num_workers=None, scale_factor=1.0):
    # Predefined list of visually distinct colors
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

    # Prepare the video writer
    frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    if frame is None:
        raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
    height, width, _ = frame.shape
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_size = (new_width, new_height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, new_size)

    # Assign colors to each unique mask ID
    all_mask_ids = set()
    for masks in video_segments.values():
        all_mask_ids.update(masks.keys())
    colors = {}
    for i, mask_id in enumerate(all_mask_ids):
        if mask_id is not None:
            colors[mask_id] = COLORS[i % len(COLORS)]
            
    # Setup for parallel processing
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    tasks = []
    for frame_idx, frame_name in enumerate(frame_names):
        masks_for_frame = video_segments.get(frame_idx, {})
        tasks.append((frame_idx, frame_name, video_dir, masks_for_frame, colors, alpha, new_size))

    # Process frames in parallel and write to video
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm.tqdm(total=len(tasks), desc="Creating overlay video") as pbar:
            for result_frame in pool.imap(_process_frame_for_video, tasks):
                if result_frame is not None:
                    out.write(result_frame)
                pbar.update(1)

    # Release the video writer
    out.release()

    print(f"Video saved to {output_video_path}")

def overlay_predictions_on_frame(video_dir, frame_idx, video_segments, alpha=0.99):
    # Predefined list of visually distinct colors
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

    # Load the image
    image_path = os.path.join(video_dir, f"{frame_idx:06d}.jpg")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a blank overlay
    overlay = np.zeros_like(image)

    # Check if we have predictions for this frame
    if frame_idx in video_segments:
        masks = video_segments[frame_idx]

        # Assign colors to each unique mask ID
        colors = {mask_id: COLORS[i % len(COLORS)] for i, mask_id in enumerate(masks.keys())}

        # Process each mask
        for mask_id, mask in masks.items():
            # Convert mask to binary numpy array if it's not already
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            if mask.dtype != bool:
                mask = mask > 0.5

            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()

            # Resize the mask to match the image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Get the color for this mask
            color = colors[mask_id]

            # Create a colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = color

            # Add the colored mask to the overlay
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)

        # Overlay the masks on the image
        overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
    else:
        print(f"No predictions found for frame {frame_idx}")
        overlaid_image = image

    # Save the overlaid image
    cv2.imwrite("frame_prediction_overlay.png", cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
    print(f"Overlaid image saved to frame_prediction_overlay.png")

def check_prompt_data(frame_idx, prompt_data, video_dir, inference_state, frame_mapping):
    new_frame_number = next((new for new, original in frame_mapping.items() if original == frame_idx), None)
    if new_frame_number is None:
        raise ValueError(f"No mapping found for original frame number {frame_idx}")
    prompts_for_frame = {}
    for obj_id, obj_data in prompt_data[str(frame_idx)].items():
        print(f"Processing frame {new_frame_number}, object {obj_id}")
        points = obj_data["points"]
        labels = obj_data["labels"]
        prompts_for_frame[int(obj_id)] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=new_frame_number,
            obj_id=int(obj_id),
            points=points,
            labels=labels
        )
        
        plt.figure(figsize=(12, 8))
        plt.title(f"New frame {new_frame_number}, prompt frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, f"{new_frame_number:06d}.jpg")))
        show_points(points, labels, plt.gca())
        for i, out_obj_id in enumerate(out_obj_ids):
            show_points(points, labels, plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)   
        plt.savefig(f"prompt_frame_data_check.png")
        print(f"Prompt frame data check saved to prompt_frame_data_check.png")
        plt.close()      
        time.sleep(0.02)  # Optimal delay between iterations for concurrency

    return prompts_for_frame

def analyze_prompt_frames_immediate(video_dir, frame_mapping, prompt_data, inference_state, predictor):
    prompt_frame_results = {}

    # Create a tqdm progress bar
    pbar = tqdm.tqdm(frame_mapping.items(), desc="Analyzing prompt frames", unit="frame")

    for new_frame_num, original_frame_num in pbar:
        if str(original_frame_num) in prompt_data:
            pbar.set_postfix({"Original Frame": original_frame_num})
            
            # Get the mask predictions for this frame
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=new_frame_num,
                obj_id=None,  # Set to None to get all object masks
                points=np.empty((0, 2)),  # Empty array as we're not adding new points
                labels=np.empty(0)
            )

            masks = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Analyze masks for this prompt frame
            empty_masks = []
            large_masks = []
            overlapping_masks = []
            
            def calculate_overlap(mask1, mask2):
                intersection = np.logical_and(mask1, mask2)
                union = np.logical_or(mask1, mask2)
                overlap_pixels = np.sum(intersection)
                iou = overlap_pixels / np.sum(union)
                return iou, overlap_pixels
            
            for obj_id, mask in masks.items():
                if obj_id is not None:  # Ensure we're only processing valid object IDs
                    mask_sum = mask.sum()
                    
                    if mask_sum == 0:
                        empty_masks.append(obj_id)
                    elif mask_sum >= 40000:  # Check for masks with 800 pixels or more
                        large_masks.append(obj_id)
                    
                    # Check for significant overlaps with other masks
                    for other_obj_id, other_mask in masks.items():
                        if other_obj_id is not None and obj_id != other_obj_id:
                            overlap, overlap_pixels = calculate_overlap(mask, other_mask)
                            if overlap > 0.01:  # 1% overlap threshold
                                overlapping_masks.append((obj_id, other_obj_id, overlap, overlap_pixels))
            
            prompt_frame_results[new_frame_num] = {
                'original_frame': original_frame_num,
                'all_objects': list(masks.keys()),  # Store all object IDs
                'empty_masks': empty_masks,
                'large_masks': large_masks,
                'overlapping_masks': overlapping_masks
            }
            
            # Visualize the prompt frame results
            plt.figure(figsize=(12, 8))
            plt.title(f"Prompt Frame {new_frame_num} (Original: {original_frame_num})")
            image = Image.open(os.path.join(video_dir, f"{new_frame_num:06d}.jpg"))
            plt.imshow(image)
            
            for obj_id, mask in masks.items():
                if obj_id is not None:
                    show_mask(mask, plt.gca(), obj_id=obj_id, random_color=True)
            
            plt.savefig(f"prompt_frame_analysis_{new_frame_num}.png")
            plt.close()

    return prompt_frame_results

def print_prompt_frame_analysis(prompt_frame_results):
    print("\nPrompt Frame Analysis Summary:")
    
    problematic_frames = []
    frames_without_issues = []
    all_objects = set()

    def safe_sort(iterable):
        return sorted((item for item in iterable if item is not None), key=lambda x: (x is None, x))

    for frame_num, results in prompt_frame_results.items():
        issues = []
        frame_objects = set(obj for obj in results['all_objects'] if obj is not None)

        if results['empty_masks']:
            issues.append(f"Empty masks: {safe_sort(results['empty_masks'])}")
        if results['large_masks']:
            issues.append(f"Large masks (800+ pixels): {safe_sort(results['large_masks'])}")
        if results['overlapping_masks']:
            overlap_info = [f"{a}-{b} ({overlap:.2%}, {pixels} pixels)" for a, b, overlap, pixels in results['overlapping_masks'] if a is not None and b is not None]
            issues.append(f"Overlapping masks: {', '.join(overlap_info)}")
        
        all_objects.update(frame_objects)
        
        if issues:
            problematic_frames.append((frame_num, results['original_frame'], issues, frame_objects))
        else:
            frames_without_issues.append((frame_num, frame_objects))

    if frames_without_issues:
        print("\nFrames without issues:")
        for frame_num, frame_objects in frames_without_issues:
            print(f"  Frame {frame_num}: Objects present: {safe_sort(frame_objects)}")

    if problematic_frames:
        print("Problematic frames:")
        for frame_num, original_frame, issues, frame_objects in problematic_frames:
            print(f"  Frame {frame_num} (Original: {original_frame}):")
            print(f"    Objects present: {safe_sort(frame_objects)}")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("No problematic frames detected.")


    print(f"\nTotal frames analyzed: {len(prompt_frame_results)}")
    print(f"Frames with issues: {len(problematic_frames)}")
    print(f"Frames without issues: {len(frames_without_issues)}")

    print(f"\nTotal unique object IDs detected across all frames: {safe_sort(all_objects)}")
    print(f"Number of unique objects: {len(all_objects)}")

    # Additional statistics
    if video_segments:
        objects_per_frame = [len([obj for obj in segments.keys() if obj is not None]) for segments in video_segments.values()]
        if objects_per_frame:
            avg_objects_per_frame = sum(objects_per_frame) / len(objects_per_frame)
            print(f"\nAverage number of objects per frame: {avg_objects_per_frame:.2f}")
            print(f"Minimum objects in a frame: {min(objects_per_frame)}")
            print(f"Maximum objects in a frame: {max(objects_per_frame)}")
        else:
            print("\nNo objects found to analyze.")
    else:
        print("\nNo segments to analyze for statistics.")

def save_video_segments_to_h5(video_segments, video_dir, output_dir, frame_mapping, num_workers=None):
    """
    Saves video segments to an HDF5 file, using parallel processing to speed up the process.
    """
    last_folder = os.path.basename(os.path.normpath(video_dir))
    filename = f"{last_folder}.h5"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    exclude_frames = set(frame_mapping.keys())
    filtered_video_segments = {
        frame: segments for frame, segments in video_segments.items()
        if frame not in exclude_frames
    }

    if not filtered_video_segments:
        print("No frames to save after filtering.")
        return {}

    all_obj_ids = set()
    for segments in filtered_video_segments.values():
        all_obj_ids.update(segments.keys())

    object_ids = sorted([oid for oid in all_obj_ids if oid is not None])
    if None in all_obj_ids:
        object_ids.append(None)

    # Correctly get a sample mask to determine shape
    sample_mask = next(iter(next(iter(filtered_video_segments.values())).values()))
    mask_shape = sample_mask.shape

    sorted_frame_indices = sorted(filtered_video_segments.keys(), reverse=True)
    num_frames = len(sorted_frame_indices)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_frames'] = num_frames
        f.attrs['object_ids'] = [str(obj_id) if obj_id is not None else 'None' for obj_id in object_ids]

        for obj_id in object_ids:
            obj_id_str = str(obj_id) if obj_id is not None else 'None'
            f.create_dataset(
                f'masks/{obj_id_str}',
                shape=(num_frames,) + mask_shape,
                dtype=bool,
                compression="gzip"
            )

        chunk_size = int(np.ceil(num_frames / num_workers))
        if chunk_size == 0:
            return filtered_video_segments
            
        tasks = []
        for i in range(0, num_frames, chunk_size):
            frames_chunk = sorted_frame_indices[i:i + chunk_size]
            segments_chunk = {frame_idx: filtered_video_segments[frame_idx] for frame_idx in frames_chunk}
            tasks.append((frames_chunk, i, object_ids, segments_chunk, mask_shape))

        # Use a context manager for the pool
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use tqdm to show progress
            results = list(tqdm.tqdm(pool.imap(_process_frame_chunk, tasks), total=len(tasks), desc="Processing and writing chunks"))

            for start_h5_idx, chunk_masks_by_obj in results:
                for obj_id, masks in chunk_masks_by_obj.items():
                    obj_id_str = str(obj_id) if obj_id is not None else 'None'
                    end_h5_idx = start_h5_idx + len(masks)
                    if masks:
                        f[f'masks/{obj_id_str}'][start_h5_idx:end_h5_idx] = np.array(masks)
    
    print(f"Saved filtered video segments to: {output_path}")
    print(f"Number of frames saved: {len(filtered_video_segments)}")
    print(f"Number of frames excluded: {len(exclude_frames)}")
    print(f"Mask dimensions: {mask_shape}")
    return filtered_video_segments

def get_random_unprocessed_video(crop_videos_dir, segmented_videos_dir):
    all_videos = [d for d in os.listdir(crop_videos_dir) if os.path.isdir(os.path.join(crop_videos_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(segmented_videos_dir, video + ".h5"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(crop_videos_dir, random.choice(unprocessed_videos))





##Get random video to process
videos_dir = '/home/lilly/phd/crawlingtracking/data_foranalysis/food'
segmented_videos_dir = '/home/lilly/phd/crawlingtracking/final_data/fullframe_segmentations'
video_dir = get_random_unprocessed_video(videos_dir, segmented_videos_dir)
print(f"Processing video: {video_dir}")



#region [add prompt frames to video]
prompt_dir = "/home/lilly/phd/crawlingtracking/prompt_frames"
prompt_data_path = "/home/lilly/phd/crawlingtracking/prompt_data.json"

# Add prompt frames to the video directory
frame_mapping = add_prompt_frames_to_video(video_dir, prompt_dir)

# Get all frame names from the video directory
frame_names = sorted([p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))],
                     key=lambda p: int(os.path.splitext(p)[0]))

#predictor.reset_state(inference_state)
inference_state = predictor.init_state(video_path=video_dir)

# Load prompt data from JSON file
with open(prompt_data_path, 'r') as f:
    prompt_data = json.load(f)

# Convert loaded data to appropriate numpy arrays
for frame_num in prompt_data:
    for obj_id in prompt_data[frame_num]:
        prompt_data[frame_num][obj_id]['points'] = np.array(prompt_data[frame_num][obj_id]['points'], dtype=np.float32)
        prompt_data[frame_num][obj_id]['labels'] = np.array(prompt_data[frame_num][obj_id]['labels'], dtype=np.int32)


# Add prompts for each frame
for new_frame_num, original_frame_num in frame_mapping.items():
    if str(original_frame_num) in prompt_data:
        for obj_id, obj_data in prompt_data[str(original_frame_num)].items():
            print(f"Processing frame {new_frame_num} (original {original_frame_num}), object {obj_id}")
            add_prompts(inference_state, new_frame_num, int(obj_id), obj_data["points"], obj_data["labels"])
            time.sleep(0.02)  # Optimal delay between iterations for concurrency
    print(f"Completed frame {new_frame_num}")


prompt_frame_results = analyze_prompt_frames_immediate(video_dir, frame_mapping, prompt_data, inference_state, predictor)
print_prompt_frame_analysis(prompt_frame_results)

#prompts_for_frame = check_prompt_data(2, prompt_data, video_dir, inference_state, frame_mapping)

#endregion


### Propagate in video
video_segments = {}
last_frame_idx = max(frame_mapping.keys())
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=last_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# Check results
analyze_and_print_results(video_segments)

#overlay_predictions_on_frame(video_dir, 506, video_segments, alpha=0.99)

#Make video with masks
create_mask_overlay_video(
    video_dir,
    frame_names,
    video_segments,
    output_video_path="crop_promptframe_tst_food.mp4",
    fps=10,
    alpha=1.0,
    num_workers=multiprocessing.cpu_count(),
    scale_factor=0.5
)

# Remove prompt frames from the video directory
remove_prompt_frames_from_video(video_dir, frame_mapping)

output_dir = segmented_videos_dir
filtered_video_segments = save_video_segments_to_h5(video_segments, video_dir, output_dir, frame_mapping)







####Adjust prompts####
prompt_data["1"]


new_prompts = {}
new_prompt_frame = 599  #frame index
#worm
ann_obj_id = 2  #object id
points = np.array([[750, 1590]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
new_prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=new_prompt_frame,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {new_prompt_frame}")
plt.imshow(Image.open(os.path.join(video_dir, f"{new_prompt_frame:06d}.jpg")))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*new_prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()




def add_new_prompt(frame_number, video_dir, prompt_dir, prompt_data_file, prompts):
    """
    Add a new prompt image and its associated data based on a frame number from the video directory.
    
    :param frame_number: Number of the frame to be used as a prompt
    :param video_dir: Directory containing the video frames
    :param prompt_dir: Directory where prompt images are stored
    :param prompt_data_file: Path to the JSON file containing prompt data
    :param prompts: Dictionary containing prompt data in the format {obj_id: (points, labels)}
    """
    # Ensure directories exist
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Find the correct frame in the video directory
    frame_name = f"{frame_number:06d}.jpg"  # Assuming 6-digit zero-padded frame numbers
    source_frame_path = os.path.join(video_dir, frame_name)
    
    if not os.path.exists(source_frame_path):
        raise FileNotFoundError(f"Frame {frame_name} not found in {video_dir}")

    # Load existing prompt data
    if os.path.exists(prompt_data_file) and os.path.getsize(prompt_data_file) > 0:
        with open(prompt_data_file, 'r') as f:
            existing_prompts = json.load(f)
    else:
        existing_prompts = {}
    
    # Determine the new prompt number
    existing_numbers = [int(num) for num in existing_prompts.keys()]
    new_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    # Copy the frame to the prompt directory with the new number
    new_image_name = f"{new_number}.jpg"
    new_image_path = os.path.join(prompt_dir, new_image_name)
    shutil.copy(source_frame_path, new_image_path)
    
    # Transform the prompts data into the required format for JSON
    new_prompt_data = {}
    for obj_id, (points, labels) in prompts.items():
        new_prompt_data[str(obj_id)] = {
            "points": points.tolist(),
            "labels": labels.tolist()
        }
    
    # Add the new prompt data to the existing prompts dictionary
    existing_prompts[str(new_number)] = new_prompt_data
    
    # Save the updated prompt data back to the JSON file
    with open(prompt_data_file, 'w') as f:
        json.dump(existing_prompts, f, indent=2)
    
    print(f"Added new prompt image {new_image_name} for frame {frame_number} and updated prompt data.")


add_new_prompt(new_prompt_frame, video_dir, prompt_dir, prompt_data_path, new_prompts)



def modify_prompt(frame_number, frame_mapping, prompt_data_file, new_prompts):
    """
    Modify existing prompt data or add new data to an existing prompt in the prompt data file,
    using the frame number from the video directory and the existing frame mapping.
    
    :param frame_number: Number of the frame in the video directory to be modified
    :param frame_mapping: Dictionary mapping new frame numbers to original prompt frame numbers
    :param prompt_data_file: Path to the JSON file containing prompt data
    :param new_prompts: Dictionary containing new or updated prompt data in the format {obj_id: (points, labels)}
    """
    # Load existing prompt data
    if os.path.exists(prompt_data_file) and os.path.getsize(prompt_data_file) > 0:
        with open(prompt_data_file, 'r') as f:
            existing_prompts = json.load(f)
    else:
        raise FileNotFoundError(f"Prompt data file not found or is empty: {prompt_data_file}")
    
    # Find the original prompt frame number using the frame_mapping
    original_frame_number = None
    for new_frame, original_frame in frame_mapping.items():
        if new_frame == frame_number:
            original_frame_number = original_frame
            break
    
    if original_frame_number is None:
        raise ValueError(f"No mapping found for frame number {frame_number}")
    
    # Convert the original frame number to string for JSON key
    prompt_number = str(original_frame_number)
    
    # Update or add new prompt data
    if prompt_number not in existing_prompts:
        existing_prompts[prompt_number] = {}
    
    for obj_id, (points, labels) in new_prompts.items():
        existing_prompts[prompt_number][str(obj_id)] = {
            "points": points.tolist(),
            "labels": labels.tolist()
        }
    
    # Save the updated prompt data back to the JSON file
    with open(prompt_data_file, 'w') as f:
        json.dump(existing_prompts, f, indent=2)
    
    print(f"Updated prompt data for video frame {frame_number} (original prompt frame {original_frame_number}).")


frame_to_modify = new_prompt_frame
updated_prompts = new_prompts
modify_prompt(frame_to_modify, frame_mapping, prompt_data_path, updated_prompts)


