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


###Get metrics for worm body
    #DONE Length
    #DONE Width
        #DONE Along
        # Average
        # Mid length
    #DONE Area
    #DONE Permimeter


normalfolder = "/home/maxime/prg/phd/dev/sambody/maskproblems_classification_training_wormmaskonly/normal/"

allworms_metrics = []
for path in glob.glob(normalfolder + "*.png"):
    print(path)
    img_id = path.split("/")[-1].removesuffix(".png").removeprefix("fm_")
    imgmask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imgh, imgw = imgmask.shape
    npmask = np.array(imgmask) != 0
    npmask = npmask.astype(np.uint8)
    npmask[npmask != 0] = 1

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
                    "area": area, 
                    "perimeter": perimeter, 
                    "medial_axis_distances_sorted": medial_axis_distances_sorted, 
                    "medialaxis_length_list": np.ndarray.tolist(medialaxis_length_list), 
                    "pruned_medialaxis_length": pruned_medialaxis_length, 
                    "mean_wormwidth": mean_wormwidth, 
                    "mid_length_width": mid_length}
    allworms_metrics.append(worm_metrics)


with open("/home/maxime/prg/phd/dev/sambody/allworms_metrics.pkl", "wb") as file:
    pickle.dump(allworms_metrics, file)



with open("/home/maxime/prg/phd/dev/sambody/allworms_metrics.pkl", 'rb') as file:
    allworms_metrics = pickle.load(file)


final_df = pd.DataFrame(allworms_metrics[0])

for i in range(1, len(allworms_metrics)):
    df_i = pd.DataFrame(allworms_metrics[i])
    final_df = pd.concat([final_df, df_i])

final_df.to_csv('allworms_metrics.csv', index=False)



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
