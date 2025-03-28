import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
from torchvision import datasets, models, transforms
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import skimage
from skimage.morphology import medial_axis, skeletonize
from skimage.measure import label
import scipy
from scipy.ndimage import convolve
import glob
import os
import pickle
import random
from skimage.measure import label
from scipy.ndimage import convolve
import pandas as pd
from sklearn.metrics import mean_squared_error
import shutil
sys.path.append("/home/lilly/phd/segment-anything-2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2 = build_sam2(model_cfg, checkpoint, device ='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

classifdevice = torch.device("cuda:0")
classif_weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
worm_noworm_classif_model = torchvision.models.vit_h_14(weights=classif_weights)
num_ftrs = worm_noworm_classif_model.heads.head.in_features
worm_noworm_classif_model.heads.head = nn.Linear(num_ftrs, 2)
worm_noworm_classif_model = worm_noworm_classif_model.to(classifdevice)
worm_noworm_classif_model.load_state_dict(torch.load('/home/lilly/phd/dev/sambody/worm_noworm_classifier_vith_perfect_weights.pth', map_location=classifdevice))

worm_noworm_classif_model.eval()
class_names = ["notworm", "worm_any"]
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.85,
    stability_score_offset=0.85
)

def is_on_edge(x, y, w, h, img_width, img_height):
    # Check left edge
    if x <= 0:
        return True
    # Check top edge
    if y <= 0:
        return True
    # Check right edge
    if (x + w) >= img_width - 1:  # -1 because of zero-based indexing
        return True
    # Check bottom edge
    if (y + h) >= img_height - 1:  # -1 because of zero-based indexing
        return True
    return False

def get_valid_imaging_area(image, threshold=5, margin=5, max_iterations=100):
    """
    Find the actual microscope field of view in the image.
    
    Args:
        image: RGB image array
        threshold: pixel value threshold to consider as "black"
        margin: number of pixels to shrink the valid area by for safety
        max_iterations: maximum number of iterations for erosion to prevent infinite loops
    
    Returns:
        mask: Binary mask of the valid imaging area
        success: Boolean indicating if the operation was successful
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Use Otsu's method to find optimal threshold
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No valid imaging area found")
        return np.ones_like(gray, dtype=bool), False
    
    # Find the largest contour that's not the entire image
    valid_contours = [cnt for cnt in contours 
                     if 0.1 < cv2.contourArea(cnt) / (gray.shape[0] * gray.shape[1]) < 0.99]
    
    if not valid_contours:
        print("Warning: No valid contours found within acceptable size range")
        return np.ones_like(gray, dtype=bool), False
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Create mask of valid area
    valid_area_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(valid_area_mask, [largest_contour], -1, 255, -1)
    
    # Erode the mask by margin pixels with iteration limit
    if margin > 0:
        kernel = np.ones((3, 3), np.uint8)  # Using smaller kernel for more controlled erosion
        eroded_mask = valid_area_mask.copy()
        for _ in range(min(margin, max_iterations)):
            temp_mask = cv2.erode(eroded_mask, kernel)
            # Check if erosion would eliminate the mask entirely
            if np.sum(temp_mask) < 1000:  # Minimum area threshold
                break
            eroded_mask = temp_mask
        valid_area_mask = eroded_mask
    
    return valid_area_mask > 0, True

def get_nonedge_masks(img_path):
    # Read and convert image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Generate masks first (since we'll need them in either case)
    masks2 = mask_generator_2.generate(image)

    # Try to get valid imaging area
    valid_area, success = get_valid_imaging_area(image)
    
    # Initialize list for non-edge masks
    nonedge_masks = []

    if success:
        # Use valid area method
        for mask in masks2:
            segmentation = mask['segmentation']
            if np.all(segmentation * valid_area == segmentation):
                nonedge_masks.append(segmentation)
    else:
        # Fall back to simple edge detection
        print(f"Falling back to simple edge detection for {img_path}")
        for mask in masks2:
            segmentation = mask['segmentation']
            coords = np.where(segmentation)
            y1, x1 = np.min(coords[0]), np.min(coords[1])
            y2, x2 = np.max(coords[0]), np.max(coords[1])
            h, w = (y2 - y1 + 1), (x2 - x1 + 1)
            
            if not is_on_edge(x1, y1, w, h, img_width, img_height):
                nonedge_masks.append(segmentation)

    return image, img_height, img_width, nonedge_masks



def save_mask_cutouts(image, nonedge_masks, output_dir='/home/lilly/phd/laeya/temp_cutouts'):
    """
    Save cutouts of the masks from the image to the specified directory.
    Refreshes the output directory each time.
    
    Args:
        image: RGB image array
        nonedge_masks: List of binary mask arrays
        output_dir: Directory to save the cutouts
    """
    # Refresh temp directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(nonedge_masks)} non-edge cutouts to")
    
    for i, mask in enumerate(nonedge_masks):
        # Get bounding box coordinates of the mask
        coords = np.where(mask)
        y1, x1 = np.min(coords[0]), np.min(coords[1])
        y2, x2 = np.max(coords[0]), np.max(coords[1])
        
        # Create a 3D mask by repeating the 2D mask for each color channel
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # Apply mask to original image
        cutout = image * mask_3d
        
        # Crop to bounding box
        cutout = cutout[y1:y2+1, x1:x2+1]
        
        # Save the cutout as jpg
        cutout_path = os.path.join(output_dir, f'{i}.jpg')
        cv2.imwrite(cutout_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))




##Classify cutouts
def classify_cutouts(nonedge_masks, cutouts_dir='/home/lilly/phd/laeya/temp_cutouts'):
    """
    Classify each cutout image as either 'worm' or 'not worm' using the pre-trained classifier.
    
    Args:
        nonedge_masks: List of masks to classify
        cutouts_dir: Directory containing the cutout images
    
    Returns:
        list: Classifications for each mask ('notworm' or 'worm_any')
    """
    classifications = []
    for i in range(len(nonedge_masks)):
        cutout_path = os.path.join(cutouts_dir, f'{i}.jpg')
        imgg = Image.open(cutout_path)
        imgg = data_transforms['val'](imgg)
        imgg = imgg.unsqueeze(0)
        imgg = imgg.to(classifdevice)
        
        outputs = worm_noworm_classif_model(imgg)
        _, preds = torch.max(outputs, 1)
        classifications.append(class_names[preds])
    
    return classifications





##Check for overlapping worm masks/Number of distinct worm regions
def merge_and_clean_worm_masks(classifications, nonedge_masks, output_dir='/home/lilly/phd/laeya/temp_cutouts', overlap_threshold=0.95, min_area=25):
    """
    Merge overlapping worm masks and clean the results by removing small regions and keeping only the largest connected component.
    Also checks for and removes masks with holes.
    
    Args:
        classifications (list): List of classifications ('worm_any' or 'notworm') for each mask
        nonedge_masks (list): List of binary masks
        output_dir (str): Directory to save mask images
        overlap_threshold (float): Threshold for merging overlapping masks (default: 0.95)
        min_area (int): Minimum area in pixels for a mask to be kept (default: 25)
    
    Returns:
        list: Cleaned and merged worm masks
        int: Number of distinct worm regions
    """
    worm_masks = []
    for i, classification in enumerate(classifications):
        if classification == "worm_any":
            worm_masks.append(nonedge_masks[i])

    if worm_masks:
        # Initialize list to track which masks have been merged
        merged_masks = []
        final_masks = []
        
        # Compare each mask with every other mask
        for i in range(len(worm_masks)):
            if i in merged_masks:
                continue
                
            current_mask = worm_masks[i]
            current_area = np.sum(current_mask)
            merged = False
            
            for j in range(i + 1, len(worm_masks)):
                if j in merged_masks:
                    continue
                    
                other_mask = worm_masks[j]
                # Calculate overlap
                overlap = np.sum(current_mask & other_mask)
                overlap_ratio = overlap / min(current_area, np.sum(other_mask))
                
                # If overlap is more than threshold, merge the masks
                if overlap_ratio > overlap_threshold:
                    current_mask = current_mask | other_mask
                    current_area = np.sum(current_mask)
                    merged_masks.append(j)
                    merged = True
            
            final_masks.append(current_mask)
        
        # Clean final masks - remove regions smaller than min_area pixels and handle discontinuous segments
        worm_masks = []
        for i, mask in enumerate(final_masks):
            if np.sum(mask) >= min_area:
                # Check for holes using contour hierarchy
                contours, hierarchy = cv2.findContours((mask * 255).astype(np.uint8), 
                                                     cv2.RETR_TREE, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                
                has_holes = False
                if hierarchy is not None:
                    hierarchy = hierarchy[0]  # Get the first dimension
                    for h in hierarchy:
                        if h[3] >= 0:  # If has parent, it's a hole
                            has_holes = True
                            print(f"Skipping mask {i} due to holes in the mask")
                            break
                
                if not has_holes:
                    # Find connected components in the mask
                    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                    
                    if num_labels > 2:  # More than one segment (label 0 is background)
                        # Get sizes of each segment
                        unique_labels, label_counts = np.unique(labels[labels != 0], return_counts=True)
                        # Keep only the largest segment
                        largest_label = unique_labels[np.argmax(label_counts)]
                        mask = (labels == largest_label).astype(np.uint8)
                    
                    worm_masks.append(mask)
        
        num_distinct_worms = len(worm_masks)
        print(f"Number of distinct worm regions: {num_distinct_worms}")
    else:
        num_distinct_worms = 0

    return worm_masks, num_distinct_worms


def filter_worms(allworms_metrics, threshold):
    filtered_metrics = []
    for worm in allworms_metrics:
        if worm['area'] > threshold * np.mean([worm['area'] for worm in allworms_metrics]):
            filtered_metrics.append(worm)
    return filtered_metrics

##Get worm metrics
def extract_worm_metrics(worm_masks, img_path, img_height, img_width, threshold=0.75):
    """
    Extract metrics for each worm mask including area, perimeter, medial axis measurements, etc.
    
    Args:
        worm_masks (list): List of binary masks for each detected worm
        img_path (str): Path to the original image
        img_height (int): Height of the original image
        img_width (int): Width of the original image
        
    Returns:
        list: List of dictionaries containing metrics for each worm
    """
    # Get image ID (filename without extension)
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    
    allworms_metrics = []
    for i, npmask in enumerate(worm_masks):
        print(f"Processing worm {i}")

        # Get largest blob
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((npmask * 255).astype(np.uint8), connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component_mask = (labels == largest_label).astype(np.uint8)

        # Get area
        area = np.sum(largest_component_mask)

        # Get contour
        contours, hierarchy = cv2.findContours((largest_component_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        # Get perimeter
        perimeter = cv2.arcLength(contours[0], True)

        # Get medial axis and distance transform
        medial_axis, distance = skimage.morphology.medial_axis(largest_component_mask > 0, return_distance=True)   
        structuring_element = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
        end_points = np.where(neighbours == 11, 1, 0)
        branch_points = np.where(neighbours > 12, 1, 0)
        labeled_branches = label(branch_points, connectivity=2)
        branch_indices = np.argwhere(labeled_branches > 0)
        end_indices = np.argwhere(end_points > 0)
        indices = np.concatenate((branch_indices, end_indices), axis=0)
        
        # Find longest path through medial axis
        paths = []
        for start in range(len(indices)):
            for end in range(len(indices)):
                startid = tuple(indices[start])
                endid = tuple(indices[end])
                route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
                length = len(route)
                paths.append([startid, endid, length, route, weight])
        
        longest_length = max(paths, key=lambda x: x[2])
        pruned_mediala = np.zeros((img_height, img_width), dtype=np.uint8)
        for coord in range(len(longest_length[3])):
            pruned_mediala[longest_length[3][coord]] = 1
            
        # Get measurements along medial axis
        medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
        medialaxis_length_list = 0 + np.arange(0, len(medial_axis_distances_sorted))
        pruned_medialaxis_length = np.sum(pruned_mediala)
        mean_wormwidth = np.mean(medial_axis_distances_sorted)
        mid_length = medial_axis_distances_sorted[int(len(medial_axis_distances_sorted)/2)]

        worm_metrics = {
            "img_id": img_id, 
            "worm_id": i,
            "area": area, 
            "perimeter": perimeter, 
            "medial_axis_distances_sorted": medial_axis_distances_sorted, 
            "medialaxis_length_list": np.ndarray.tolist(medialaxis_length_list), 
            "pruned_medialaxis_length": pruned_medialaxis_length, 
            "mean_wormwidth": mean_wormwidth, 
            "mid_length_width": mid_length,
            "mask": largest_component_mask
        }
        allworms_metrics.append(worm_metrics)
    
    return filter_worms(allworms_metrics, threshold = threshold)




##Save worms
def save_worms(allworms_metrics, original_image=None, cutouts_dir='/home/lilly/phd/laeya/final_cutouts', 
               metrics_dir='/home/lilly/phd/laeya/final_metrics'):
    """
    Filter worms by area and save the filtered cutouts and metrics.
    
    Args:
        allworms_metrics (list): List of dictionaries containing worm metrics
        original_image (numpy.ndarray): Original RGB image for overlay
        cutouts_dir (str): Directory to save the cutouts
        metrics_dir (str): Directory to save the filtered metrics pickle files
    
    Returns:
        list: Filtered worm metrics
    """
    if not allworms_metrics:
        print("No worm metrics provided")
        return []
          
    # Get image ID from first worm metric
    img_id = allworms_metrics[0]["img_id"]
    
    # Save cutouts of filtered worms
    for i, worm in enumerate(allworms_metrics):
        cutout_path = os.path.join(cutouts_dir, f'{img_id}_worm_{i}.png')
        cutout_name = f'{img_id}_worm_{i}'  # The name that will appear on the image
        
        if original_image is not None:
            # Create a color overlay mask (green with 40% transparency)
            overlay = original_image.copy()
            # Apply green color where mask is True
            overlay[worm["mask"] > 0] = [0, 255, 0]  # Green color
            
            # Blend the overlay with the original image
            alpha = 0.4  # Back to original 40% transparency
            blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
            
            # Add text directly to the blended image (no transparency)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blended, cutout_name, (10, 30), font, 1, (255, 255, 255), 2)
            
            # Save the final image
            cv2.imwrite(cutout_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        else:
            # Fall back to saving just the mask if original image not provided
            cv2.imwrite(cutout_path, (worm["mask"] * 255).astype(np.uint8))
    
    # Save metrics as pickle using img_id as filename
    metrics_path = os.path.join(metrics_dir, f'{img_id}.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(allworms_metrics, f)
    print(f"Saved filtered metrics to {metrics_path}")
    
    return allworms_metrics



###Run analysis
def process_folder(input_folder, temp_cutouts_dir='/home/lilly/phd/laeya/temp_cutouts', 
                  final_cutouts_dir='/home/lilly/phd/laeya/final_cutouts',
                  metrics_dir='/home/lilly/phd/laeya/final_metrics',
                  noworms_file='/home/lilly/phd/laeya/noworms_images.csv'):
    """
    Process all images in a folder through the complete worm analysis pipeline.
    
    Args:
        input_folder (str): Path to folder containing input images
        temp_cutouts_dir (str): Directory for temporary mask cutouts
        final_cutouts_dir (str): Directory for final worm cutouts
        metrics_dir (str): Directory for saving worm metrics
        noworms_file (str): Path to CSV file for saving images with no worms
    """
    
    # Create/check noworms CSV file
    if not os.path.exists(noworms_file):
        with open(noworms_file, 'w') as f:
            f.write('image_path\n')

    # Create/check final cutouts directory
    if not os.path.exists(final_cutouts_dir):
        os.makedirs(final_cutouts_dir)
    
    # Create/check metrics directory
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    
    # Process each image
    for img_path in image_files:
        print(f"\nProcessing {img_path}")
        try:
            # Extract masks
            image, img_height, img_width, nonedge_masks = get_nonedge_masks(img_path)
            if len(nonedge_masks) == 0:
                print(f"No valid masks found in {img_path}")
                print(f"Number of worms in image: 0")
                with open(noworms_file, 'a') as f:
                    f.write(f'{img_path}\n')
                continue
                
            # Save mask cutouts
            save_mask_cutouts(image, nonedge_masks, temp_cutouts_dir)
            
            # Classify cutouts
            classifications = classify_cutouts(nonedge_masks, temp_cutouts_dir)
            
            # Merge and clean worm masks
            worm_masks, num_distinct_worms = merge_and_clean_worm_masks(
                classifications, nonedge_masks, temp_cutouts_dir)
            if num_distinct_worms == 0:
                print(f"No worms detected in {img_path}")
                print(f"Number of worms in image: 0")
                with open(noworms_file, 'a') as f:
                    f.write(f'{img_path}\n')
                continue
                
            # Extract metrics
            worm_metrics = extract_worm_metrics(worm_masks, img_path, img_height, img_width)
            
            # Print number of worms kept after all filtering
            print(f"Final number of worms in image: {len(worm_metrics)}")
            
            save_worms(worm_metrics, original_image=image, cutouts_dir=final_cutouts_dir, metrics_dir=metrics_dir)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            print(f"Number of worms in image: 0")
            continue
    
    print(f"\nAnalysis complete. Processed {len(image_files)} images.")
    print(f"Results saved to {metrics_dir}")



input_folder = '/home/lilly/phd/laeya/data/data_jpg/L1'

#D1 DONE
#D3 DONE
#D5 DONE
#D8 DONE
#L1 DONE
#L2 DONE
#L3 DONE
#L4 DONE

process_folder(input_folder, temp_cutouts_dir='/home/lilly/phd/laeya/temp_cutouts', 
                  final_cutouts_dir='/home/lilly/phd/laeya/final_cutouts',
                  metrics_dir='/home/lilly/phd/laeya/final_metrics',
                  noworms_file='/home/lilly/phd/laeya/noworms_images.csv')







# Check for duplicate rows in the noworms_images.csv file and save a deduplicated version
def check_and_remove_duplicates_in_noworms_file(noworms_file):
    try:
        # Check if file exists
        if not os.path.exists(noworms_file):
            print(f"No worms file not found: {noworms_file}")
            return
        
        # Read the file
        with open(noworms_file, 'r') as f:
            lines = f.readlines()
        
        # Keep track of header
        header = None
        if lines and lines[0].strip().lower() == 'image_path':
            header = lines[0]
            lines = lines[1:]
        
        # Check for duplicates and keep only unique entries
        seen = set()
        unique_lines = []
        duplicates = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped in seen:
                duplicates.append(line_stripped)
            else:
                seen.add(line_stripped)
                unique_lines.append(line)
        
        # Print results
        if duplicates:
            print(f"Found {len(duplicates)} duplicate entries in {noworms_file}:")
            for dup in duplicates:
                print(f"  - {dup}")
                
            # Save deduplicated file
            deduplicated_file = noworms_file.replace('.csv', '_deduplicated.csv')
            with open(deduplicated_file, 'w') as f:
                if header:
                    f.write(header)
                f.writelines(unique_lines)
            print(f"Saved deduplicated file to {deduplicated_file}")
        else:
            print(f"No duplicates found in {noworms_file}")
            
    except Exception as e:
        print(f"Error checking duplicates in {noworms_file}: {str(e)}")

# Run the duplicate check and deduplication
noworms_file = '/home/lilly/phd/laeya/noworms_images.csv'
check_and_remove_duplicates_in_noworms_file(noworms_file)




























###Test crap
##Get real worm mask

""" 
def get_real_worm(img_masks, thewormmask, imgw, imgh):  

    #Merge all found worm masks
    sum_masks = np.sum([img_masks[segs]['segmentation'] for segs in thewormmask], axis = 0)
    sum_masks[sum_masks != 0] = 1
    #Get external contours of blobs
    contours, hierarchy = cv2.findContours((sum_masks * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #If countours == 1, masks were worm submasks and were merged. Take the merged mask as worm.
    if len(contours) == 1:
        zeroimg = np.zeros((imgh, imgw), dtype=np.uint8)
        final_worm = cv2.drawContours(zeroimg, [contours[0]], -1, color=1, thickness=cv2.FILLED)

        return final_worm        
    
    #If countours > 1, take best fit quadratic mask as worm.
    else:
        #Get the worm blob with best quadratic fit
        quadfit_mse = []
        for i, contour in enumerate(contours):
            print(i, len(contour))
            if len(contour) < 200:
                quadfit_mse.append(1000000)
            else:
                #Draw the blob
                zeroimg = np.zeros((imgh, imgw), dtype=np.uint8)
                blobarr = cv2.drawContours(zeroimg, [contour], -1, color=1, thickness=cv2.FILLED)
                #Get blob medial axis
                medial_axis, distance = skimage.morphology.medial_axis(blobarr > 0, return_distance=True)
                print(np.sum(blobarr))
                print(np.sum(medial_axis))
                if np.sum(medial_axis) < 200:
                    quadfit_mse.append(1000000)
                else:
                    structuring_element = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
                    neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
                    end_points = np.where(neighbours == 11, 1, 0)
                    branch_points = np.where(neighbours > 12, 1, 0)
                    labeled_branches = label(branch_points, connectivity=2)
                    branch_indices = np.argwhere(labeled_branches > 0)
                    end_indices = np.argwhere(end_points > 0)
                    indices = np.concatenate((branch_indices, end_indices), axis=0)
                    paths = []
                    for start in range(len(indices)):
                        for end in range(len(indices)):
                            startid = tuple(indices[start])
                            endid = tuple(indices[end])
                            route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
                            length = len(route)
                            paths.append([startid, endid, length, route, weight])
                    longest_length = max(paths, key=lambda x: x[2])
                    pruned_mediala = np.zeros((imgh, imgw), dtype=np.uint8)
                    for coord in range(len(longest_length[3])):
                        pruned_mediala[longest_length[3][coord]] = 1
                    #Get distances along the medial axis
                    medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
                    #Medial axis length (length of worm)
                    medialaxis_length = 0 + np.arange(0, len(medial_axis_distances_sorted))
                    #Quadratic fit
                    coefs = np.polyfit(medialaxis_length, medial_axis_distances_sorted, 2)
                    p = np.poly1d(coefs)
                    y_pred = p(medialaxis_length)
                    mse = mean_squared_error(medial_axis_distances_sorted, y_pred)
                    quadfit_mse.append(mse)

        #Draw final worm as blob with the best quadratic fit
        wormcontour = contours[np.argmin(quadfit_mse)]
        zeroimg = np.zeros((imgh, imgw), dtype=np.uint8)
        final_worm = cv2.drawContours(zeroimg, [wormcontour], -1, color=1, thickness=cv2.FILLED)

        return final_worm

 """

allimages_finalwormmask = []
manyworms = []
zeroworm = []

#/home/maxime/prg/phd/dev/sambody/final_worm_masks/fm_c_p1_02_d1.png

""" png = 184
original_image_path = img_path
sam_segmentation = img_masks[3]['segmentation'] """

for png in range(len(allsam)):
    print(png)
    img_path = allsam[png]['img_id']
    if img_path == "/home/maxime/prg/phd/dev/data/c_p1_02_d1.png":
        continue
    print(img_path)
    img_masks = allsam[png]['masks']
    imgh, imgw, _ = cv2.imread(img_path).shape

    thewormmask = []

    for maskpred in range(len(img_masks)):

        ###Remove border masks
        x, y, w, h = bbox = img_masks[maskpred]['bbox']
        if is_on_edge(*bbox, imgw, imgh) == True:
            continue

        ###Class masks
        #if isworm_mask(img_path, img_masks[maskpred]['segmentation']) == False:
            #continue

        if isworm_mask(img_path, img_masks[maskpred]['segmentation']) == True:
            thewormmask.append(maskpred)

    if len(thewormmask) > 1:
        real_worm = get_real_worm(img_masks, thewormmask, imgw, imgh)
        manyworms.append(img_path)
        print("Many: " + str(len(thewormmask)))
    
    if len(thewormmask) == 0:
        #Fix it
        zeroworm.append(img_path)
        print("No mask:" + str(len(img_masks)))
        continue  

    if len(thewormmask) == 1:
        real_worm = img_masks[thewormmask[0]]['segmentation'] 
    
    allimages_finalwormmask.append({"img_id": img_path, "final_wormmask": real_worm})


with open("/home/maxime/prg/phd/dev/sambody/final_worm_mask.pkl", "wb") as file:
    pickle.dump(allimages_finalwormmask, file)


for dict_ in allimages_finalwormmask:
    if dict_.get('img_id') == img_path:
        wmask_01 = dict_.get('final_wormmask')









###Test crap

max_dicts = 0
for item in allimages_finalwormmask:
    num_dicts = len(item["final_wormmask"])
    print(num_dicts)
    if num_dicts > max_dicts:
        max_dicts = num_dicts

#43 Many 8 noworm
#37 Many 6 noworm
    

medial_axis, distance = skimage.morphology.medial_axis(img_masks[3]['segmentation'] > 0, return_distance=True)

# Function to compute the medial axis and find branch and end points
    # Define a structuring element for the convolution operation to find neighbours
structuring_element = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)
    # Convolve skeleton with the structuring element to count neighbours
neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
    # End points have one neighbour
end_points = np.where(neighbours == 11, 1, 0)
    # Branch points have more than two neighbours
branch_points = np.where(neighbours > 12, 1, 0)

labeled_branches = label(branch_points, connectivity=2)
# Extract the indices of branch and end points
branch_indices = np.argwhere(labeled_branches > 0)
end_indices = np.argwhere(end_points > 0)
# Include branch points as potential start/end points for routes
indices = np.concatenate((branch_indices, end_indices), axis=0)

paths = []
for start in range(len(indices)):
    for end in range(len(indices)):
        startid = tuple(indices[start])
        endid = tuple(indices[end])
        route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
        length = len(route)
        paths.append([startid, endid, length, route, weight])


longest_length = max(paths, key=lambda x: x[2])

pruned_mediala = np.zeros(img_masks[3]['segmentation'].shape)

for coord in range(len(longest_length[3])):
    pruned_mediala[longest_length[3][coord]] = 1 

medial_axis_distances = distance[np.array(pruned_mediala, dtype=bool)]
medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
wormlength = 0 + np.arange(0, len(medial_axis_distances_sorted))

masks1 = medial_axis_distances_sorted
masks2 = medial_axis_distances_sorted
masks3 = medial_axis_distances_sorted

#Peaks - Peaks too small
peaks, properties = scipy.signal.find_peaks(masks3, prominence=35)

#Total variation - Not different enough
sum(abs(y - x) for x, y in zip(masks3[:-1], masks3[1:]))

#Autocorrelation - Not different enough
pd.Series(masks3).autocorr(lag=1)

#Quadratic fit
def fit_quadratic_and_calculate_mse(x, y):
    # Fit the quadratic polynomial
    coefs = np.polyfit(x, y, 2)
    p = np.poly1d(coefs)
    # Predict y values using the polynomial
    y_pred = p(x)
    # Calculate MSE
    mse = mean_squared_error(y, y_pred)
    return mse, p

mse1, p1 = fit_quadratic_and_calculate_mse(0 + np.arange(0, len(masks1)), masks1)
mse2, p2 = fit_quadratic_and_calculate_mse(0 + np.arange(0, len(masks2)), masks2)
mse3, p3 = fit_quadratic_and_calculate_mse(0 + np.arange(0, len(masks3)), masks3)


im1 = img_masks[0]['segmentation']
im2 = img_masks[1]['segmentation']
im3 = img_masks[2]['segmentation']

im1 = img_masks[thewormmask[0]]['segmentation']
im2 = img_masks[thewormmask[1]]['segmentation']
im3 = img_masks[thewormmask[2]]['segmentation']

img_array = np.uint8(real_worm) * 70 
img = Image.fromarray(img_array)
img.save('real_worm.png')

img_array = np.uint8(im2) * 100 
img = Image.fromarray(img_array)
img.save('im2.png')

img_array = np.uint8(im3) * 100 
img = Image.fromarray(img_array)
img.save('im3.png')
