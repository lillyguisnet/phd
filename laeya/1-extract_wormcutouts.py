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
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2 = build_sam2(model_cfg, checkpoint, device ='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)


##Extract masks with SAM2
img = '/home/lilly/phd/laeya/data/data_jpg/D1/D1_rep2_15.jpg'

mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.85,
    stability_score_offset=0.85
)

image = cv2.imread(img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks2 = mask_generator_2.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.savefig("tst.png")
plt.close()
len(masks2)


## Filter out border masks
def is_on_edge(x, y, w, h, img_width, img_height):
    # Check left edge
    if x == 0:
        return True
    # Check top edge
    if y == 0:
        return True
    # Check right edge
    if x + w == img_width:
        return True
    # Check bottom edge
    if y + h == img_height:
        return True
    return False


nonedge_masks = []
img_height, img_width = image.shape[:2]

for mask in masks2:
    # Get the bounding box coordinates from the segmentation mask
    segmentation = mask['segmentation']
    coords = np.where(segmentation)
    y1, x1 = np.min(coords[0]), np.min(coords[1])
    y2, x2 = np.max(coords[0]), np.max(coords[1])
    h, w = y2 - y1, x2 - x1
    
    # Only keep masks that are not touching the edges
    if not is_on_edge(x1, y1, w, h, img_width, img_height):
        nonedge_masks.append(segmentation)
len(nonedge_masks)


##Save cutouts of nonedge masks
if os.path.exists('/home/lilly/phd/laeya/temp_cutouts'):
    shutil.rmtree('/home/lilly/phd/laeya/temp_cutouts')
os.makedirs('/home/lilly/phd/laeya/temp_cutouts', exist_ok=True)

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
    cutout_path = f'/home/lilly/phd/laeya/temp_cutouts/{i}.jpg'
    cv2.imwrite(cutout_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))



##Classify cutouts
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

classifications = []
for i in range(len(nonedge_masks)):
    cutout_path = f'/home/lilly/phd/laeya/temp_cutouts/{i}.jpg'
    imgg = Image.open(cutout_path)
    imgg = data_transforms['val'](imgg)
    imgg = imgg.unsqueeze(0)
    imgg = imgg.to(classifdevice)
    
    outputs = worm_noworm_classif_model(imgg)
    _, preds = torch.max(outputs, 1)
    classifications.append(class_names[preds])



##Check for overlapping worm masks/Number of distinct worm regions
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
            
            # If overlap is more than 95%, merge the masks
            if overlap_ratio > 0.95:
                current_mask = current_mask | other_mask
                current_area = np.sum(current_mask)
                merged_masks.append(j)
                merged = True
        
        final_masks.append(current_mask)
    
    # Clean final masks - remove regions smaller than 25 pixels and handle discontinuous segments
    worm_masks = []
    for i, mask in enumerate(final_masks):
        if np.sum(mask) >= 25:
            # Find connected components in the mask
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            
            if num_labels > 2:  # More than one segment (label 0 is background)
                # Get sizes of each segment
                unique_labels, label_counts = np.unique(labels[labels != 0], return_counts=True)
                # Keep only the largest segment
                largest_label = unique_labels[np.argmax(label_counts)]
                mask = (labels == largest_label).astype(np.uint8)
            
            worm_masks.append(mask)
            # Save each filtered mask as PNG
            cv2.imwrite(f'/home/lilly/phd/laeya/temp_cutouts/region_{i}.png', (mask * 255).astype(np.uint8))
    
    num_distinct_worms = len(worm_masks)
    print(f"Number of distinct worm regions: {num_distinct_worms}")
else:
    print("No worms detected")




##Get worm metrics
allworms_metrics = []
for i, npmask in enumerate(worm_masks):
    print(f"Processing worm {i}")
    img_id = img #<--------------------------------------------
    imgh, imgw = npmask.shape

    #Get largest blob
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((npmask * 255).astype(np.uint8), connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component_mask = (labels == largest_label).astype(np.uint8)

    ##Get area
    area = np.sum(largest_component_mask)

    #Get contour
    contours, hierarchy = cv2.findContours((largest_component_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    ##Get perimeter
    perimeter = cv2.arcLength(contours[0], True)

    medial_axis, distance = skimage.morphology.medial_axis(largest_component_mask > 0, return_distance=True)   
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
    ##Get distances along the medial axis
    medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
    medialaxis_length_list = 0 + np.arange(0, len(medial_axis_distances_sorted))
    ##Get medial axis length (length of worm)
    pruned_medialaxis_length = np.sum(pruned_mediala)
    ##Get mean width of worm
    mean_wormwidth = np.mean(medial_axis_distances_sorted)
    ##Get mid length width of worm
    mid_length = medial_axis_distances_sorted[int(len(medial_axis_distances_sorted)/2)]

    worm_metrics = {"img_id": img_id, 
                    "worm_id": i,
                    "area": area, 
                    "perimeter": perimeter, 
                    "medial_axis_distances_sorted": medial_axis_distances_sorted, 
                    "medialaxis_length_list": np.ndarray.tolist(medialaxis_length_list), 
                    "pruned_medialaxis_length": pruned_medialaxis_length, 
                    "mean_wormwidth": mean_wormwidth, 
                    "mid_length_width": mid_length,
                    "mask": largest_component_mask}
    allworms_metrics.append(worm_metrics)



##Filter out cut worms
# Calculate average area across all worms
avg_area = np.mean([worm["area"] for worm in allworms_metrics])

# Filter out worms with area less than 75% of average
allworms_metrics_filtersmall = [worm for worm in allworms_metrics if worm["area"] >= 0.75 * avg_area]
len(allworms_metrics_filtersmall)


##Save cutouts of filtered worms
for i, worm in enumerate(allworms_metrics_filtersmall):
    img_id = worm["img_id"]
    mask = worm["mask"]
    cv2.imwrite(f'/home/lilly/phd/laeya/temp_cutouts/worm_{i}.png', (mask * 255).astype(np.uint8))

#Save metrics































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
