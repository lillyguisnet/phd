import sys
sys.path.append("..")
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
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.ndimage import distance_transform_edt
from numpy.random import default_rng


###Using all masks

#For each image in the sam predictions
    #Save the masks predictions in a list (worm, not worm, background) by checking the cutout classification
    #Check if some of the worm masks overlap, if so, keep only the largest as real worms
    #Save RGB info for each worm mask within the image (use cutout id as worm id)
    #Save other worm metrics for each worm as well
    #Get pixels from background


#Get mask preds from full images
allmasks_pred = "/home/maxime/prg/phd/dev/oro/oro_allmaskspred.pkl"
with open(allmasks_pred, 'rb') as file:
    allmasks_pred = pickle.load(file)

#Cutout classification
cutout_preds = "/home/maxime/prg/phd/dev/oro/oro_cutouts_classification.pkl"
with open(cutout_preds, 'rb') as file:
    cutout_preds = pickle.load(file)


final_worms_metrics = []

for img in allmasks_pred:
    print(img['filename'])

    ##for each mask in the image, check if it is a worm or not
    mask_classifications = []
    mask_count = 0
    for mask_index in range(len(img['masks'])):
        #print(img['masks'][mask_index])
        cutout_id = "c" + str(mask_index) + "_" + img['filename'].split("/")[-1]
        maskpred = [cutout for cutout in cutout_preds if cutout['maskid'] == cutout_id]
        if not maskpred:
            mask_classifications.append("background")        
        else:
            if maskpred[0]['classid'] == "worm":
                mask_classifications.append("worm")
            if maskpred[0]['classid'] == "notworm":
                mask_classifications.append("notworm")


    ##Check if some of the worm masks overlap, if so, keep only the largest as real worms
    worm_masks_with_indices = [(i, img['masks'][i]['segmentation']) for i, classification in enumerate(mask_classifications) if classification == 'worm']

    if not worm_masks_with_indices: #if no worm masks in img, skip to next img
        continue
    else:
        #Get biggest worm mask if worms overlap
        original_indices, worm_masks = zip(*worm_masks_with_indices)
        n_worms = len(worm_masks)
        # Create an adjacency matrix for the graph, where a 1 indicates an overlap between two masks
        adjacency_matrix = np.zeros((n_worms, n_worms))
        # Calculate areas for all worm masks
        areas = np.array([np.sum(mask) for mask in worm_masks])
        # Populate the adjacency matrix by checking overlaps
        for i in range(n_worms):
            for j in range(i + 1, n_worms):
                if np.any(np.logical_and(worm_masks[i], worm_masks[j])):
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1  # The matrix is symmetric
        # Find connected components (i.e., groups of overlapping masks)
        n_components, labels = connected_components(csgraph=csr_matrix(adjacency_matrix), directed=False)
        # For each component, find the largest worm mask
        largest_masks_original_indices = []
        largest_masks_areas = []

        for component_label in set(labels):
            component_mask_indices = [index for index, label in enumerate(labels) if label == component_label]
            component_areas = [areas[index] for index in component_mask_indices]
            largest_index = component_mask_indices[np.argmax(component_areas)]

            # Map back to original index in the classifications list
            original_index = original_indices[largest_index]
            largest_masks_original_indices.append(original_index)
            largest_masks_areas.append(np.max(component_areas))


        ##Save channel info and metrics for each worm mask within the image (use cutout id as worm id)
        for i, wmask_index in enumerate(largest_masks_original_indices):
            #Get worm mask
            worm_mask = img['masks'][wmask_index]['segmentation']
            #Use worm_mask to get pixel value from the orignal image where the mask is true
            #Open img
            img_array = cv2.imread("/home/maxime/prg/phd/dev/oro/original_imgs_fixedname/" + img['filename'].replace(" ", "_").split("/")[-1])
            imgh, imgw, ch = img_array.shape

            #Get largest blob of mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((worm_mask * 255).astype(np.uint8), connectivity=8)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_component_mask = (labels == largest_label)

            ##Get area
            area = np.sum(largest_component_mask)

            #Get contour
            contours, hierarchy = cv2.findContours((largest_component_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

            ##Get perimeter
            perimeter = cv2.arcLength(contours[0], True)

            ##Get medial axis
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

            ##Get background for each worm mask
            #Sum all "background" masks from mask_classifications
            background_mask = np.zeros((imgh, imgw), dtype=np.uint8)
            for maskindex in range(len(img['masks'])):
                if mask_classifications[maskindex] == "background":
                    background_mask = background_mask + img['masks'][maskindex]['segmentation']
            #Select 50 random pixels from background_mask that are at least 100 pixels away from edges and 100 pixels away from non-background locations.
            distances_to_non_background = distance_transform_edt(background_mask)
            rows, cols = background_mask.shape
            edge_distance = 100
            non_background_distance = 100
            distances_to_non_background[:edge_distance, :] = 0  # Top edge
            distances_to_non_background[-edge_distance:, :] = 0  # Bottom edge
            distances_to_non_background[:, :edge_distance] = 0  # Left edge
            distances_to_non_background[:, -edge_distance:] = 0  # Right edge
            valid_pixels = np.argwhere(distances_to_non_background >= non_background_distance)

            rng = default_rng()
            selected_indices = rng.choice(len(valid_pixels), size=50, replace=False)
            selected_pixels = valid_pixels[selected_indices]

            #Get rgb correction values from original image
            selected_pixel_values = img_array[selected_pixels[:, 0], selected_pixels[:, 1]]
            bgcorr_r = np.sum(selected_pixel_values[:, 2])/50
            bgcorr_g = np.sum(selected_pixel_values[:, 1])/50
            bgcorr_b = np.sum(selected_pixel_values[:, 0])/50

            #Get each channel from the original image
            img_r = img_array[:,:,2][largest_component_mask == 1]
            img_g = img_array[:,:,1][largest_component_mask == 1]
            img_b = img_array[:,:,0][largest_component_mask == 1]

            #Normalize the RGB channels against the background and keep worm mask
            red_channel_normbg = (img_r / bgcorr_r) #* largest_component_mask
            green_channel_normbg = (img_g / bgcorr_g) #* largest_component_mask
            blue_channel_normbg = (img_b / bgcorr_b) #* largest_component_mask

            #Ensure the normalized values are within the valid range
            #red_channel_normbg = np.clip(red_channel_normbg, 0, 255)
            #green_channel_normbg = np.clip(green_channel_normbg, 0, 255)
            #blue_channel_normbg = np.clip(blue_channel_normbg, 0, 255)

            #Calculate the maximum and minimum of the normalized RGB channels
            #max_channel = np.maximum(np.maximum(red_channel_normbg, green_channel_normbg), blue_channel_normbg)
            #min_channel = np.minimum(np.minimum(red_channel_normbg, green_channel_normbg), blue_channel_normbg)

            #Avoid division by zero
            #max_channel[max_channel == 0] = 1

            #Calculate the saturation (per pixel)
            #saturation = (max_channel - min_channel) / max_channel

            #Calculate luminance (brightness) using the normalized red channel as a proxy (per pixel)
            luminance = red_channel_normbg #/ 255.0  # Normalize to [0,1]

            #Calculate red dominance (per pixel)
            red_dominance = red_channel_normbg - (green_channel_normbg + blue_channel_normbg) / 2

            #Combine saturation, luminance, and red dominance (per pixel)
            redness_measure = (1/luminance) * red_dominance #saturation *

            #redness_measure_masked = redness_measure[largest_component_mask == 1]

            # Calculate statistics
            mean_redness = np.mean(redness_measure)
            min_redness = np.min(redness_measure)
            max_redness = np.max(redness_measure)
            median_redness = np.median(redness_measure)
            std_dev_redness = np.std(redness_measure)

            #Get worm mask metrics
            worm_mask_metrics = {"maskid": "c" + str(wmask_index) + "_" + img['filename'].split("/")[-1].replace(" ", "_"),	
                                "mean_redness": mean_redness,
                                "min_redness": min_redness,
                                "max_redness": max_redness,
                                "median_redness": median_redness,
                                "std_dev_redness": std_dev_redness,
                                "area": area,
                                "perimeter": perimeter,
                                "medial_axis_distances_sorted": medial_axis_distances_sorted, 
                                "medialaxis_length_list": np.ndarray.tolist(medialaxis_length_list), 
                                "pruned_medialaxis_length": pruned_medialaxis_length, 
                                "mean_wormwidth": mean_wormwidth, 
                                "mid_length_width": mid_length}

            final_worms_metrics.append(worm_mask_metrics)



with open("/home/maxime/prg/phd/dev/oro/allmasks_removeoverlap_normredness_nosaturation_metrics.pkl", "wb") as file:
    pickle.dump(final_worms_metrics, file)



with open("/home/maxime/prg/phd/dev/oro/allmasks_removeoverlap_normredness_nosaturation_metrics.pkl", 'rb') as file:
    allworms_metrics = pickle.load(file)


final_df = pd.DataFrame(allworms_metrics[0])

for i in range(1, len(allworms_metrics)):
    df_i = pd.DataFrame(allworms_metrics[i])
    final_df = pd.concat([final_df, df_i])

final_df.to_csv('/home/maxime/prg/phd/dev/oro/allmasks_removeoverlap_normredness_nosaturation_metrics.csv', index=False)







###Test crap

#Make a list of worm masks
classification = "/home/maxime/prg/phd/dev/oro/oro_cutouts_classification.pkl"
with open(classification, 'rb') as file:
    cutouts_classification = pickle.load(file)

worm_masks = [cutout["maskid"] for cutout in cutouts_classification if cutout['classid'] == "worm"]







path = 'c0_a_20220403T185346 001.png'
#Find mask pred for img path
mask_index = int(path.split("_")[0].removeprefix("c"))
img_name = "/home/maxime/prg/phd/dev/data_oro/data_oro_png/" + path.removeprefix(path.split("_")[0] + "_")
mask_pred = [mask['masks'][mask_index] for mask in allmasks_pred if mask['filename'] == img_name]

cutout_preds = "/home/maxime/prg/phd/dev/oro/oro_cutouts_classification.pkl"
with open(cutout_preds, 'rb') as file:
    cutout_preds = pickle.load(file)

#Check if masks for for the same image are overlapping
def group_masks_by_image(mask_names):
    # Initialize an empty dictionary to hold image names as keys and a list of mask indices as values
    masks_by_image = {}

    # Iterate over each mask name in the list
    for mask_name in mask_names:
        # Split the mask name by underscore to separate the components
        parts = mask_name['filename'].split('_')
        # The mask index is the first element, and the image name comprises the remaining elements
        mask_index = parts[0].removeprefix("c")
        image_name = '_'.join(parts[1:])

        # Check if the image name is already a key in the dictionary
        if image_name in masks_by_image:
            # If the image name exists, append the mask index to its list
            masks_by_image[image_name].append(mask_index)
        else:
            # If the image name does not exist, create a new entry with the mask index in a list
            masks_by_image[image_name] = [mask_index]

    # Return the dictionary of masks grouped by image name
    return masks_by_image

group_masks_by_image(allmasks_pred)





#Get metrics for each worm mask

allworms_metrics = []
for path in worm_masks:
    print(path)
    img_id = path.replace(" ", "_")
    pathtoimg = "/home/maxime/prg/phd/dev/oro/mask_cutouts_fixedname/" + img_id
    imgmask = cv2.imread(pathtoimg) #, cv2.IMREAD_GRAYSCALE
    imgh, imgw, ch = imgmask.shape

    #Make binary array for shape ananylsis
    binary_mask = np.where(np.all(imgmask == 0, axis=-1), 0, 1)

    #Get largest blob
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((binary_mask * 255).astype(np.uint8), connectivity=8)
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
                    "area": area, 
                    "perimeter": perimeter, 
                    "medial_axis_distances_sorted": medial_axis_distances_sorted, 
                    "medialaxis_length_list": np.ndarray.tolist(medialaxis_length_list), 
                    "pruned_medialaxis_length": pruned_medialaxis_length, 
                    "mean_wormwidth": mean_wormwidth, 
                    "mid_length_width": mid_length}
    allworms_metrics.append(worm_metrics)


with open("/home/maxime/prg/phd/dev/oro/allmasks_nofilters_metrics.pkl", "wb") as file:
    pickle.dump(allworms_metrics, file)



with open("/home/maxime/prg/phd/dev/oro/allmasks_nofilters_metrics.pkl", 'rb') as file:
    allworms_metrics = pickle.load(file)


final_df = pd.DataFrame(allworms_metrics[0])

for i in range(1, len(allworms_metrics)):
    df_i = pd.DataFrame(allworms_metrics[i])
    final_df = pd.concat([final_df, df_i])

final_df.to_csv('/home/maxime/prg/phd/dev/oro/allworms_metrics.csv', index=False)



## Test crap 

# Assuming 'data' is your list of dictionaries
data = allworms_metrics[0]

# Convert to DataFrame
df = pd.DataFrame(data)




# assuming df is your DataFrame and 'col1' and 'col2' are the columns you want to explode
df['combined'] = list(zip(df['medial_axis_distances_sorted'], df['medialaxis_length_list']))
exploded_df = df['combined'].explode()

# this will give you a DataFrame with tuples, you can convert it back to separate columns like this:
exploded_df[['medial_axis_distances_sorted', 'medialaxis_length_list']] = pd.DataFrame(df['combined'].tolist(), index=df.index)





# Explode the 'medial_axis_distances_sorted' list into separate rows
df = df.explode('medial_axis_distances_sorted')

# Convert the 1D array to a list and explode it into separate rows
df['medialaxis_length_list'] = df['medialaxis_length_list'].apply(list)
df = df.explode('medialaxis_length_list')

# Reset index if needed
df.reset_index(drop=True, inplace=True)

print(df)

img_array = cv2.drawContours(zeroimg, [contours[0]], -1, 1) *100
img = Image.fromarray(img_array)
img.save('contours.png')
