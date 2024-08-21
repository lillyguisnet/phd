import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import glob
import os
import skimage

sam_checkpoint = "phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)



default_generator = SamAutomaticMaskGenerator(sam)


img = cv2.imread('phd/dev/sambody/tstimgs/body_4.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

masks = default_generator.generate(img)
print(len(masks))
print(masks[0].keys())


wmask = masks[1]['segmentation']
bmask = masks[0]['segmentation']

wmask_01 = wmask.astype(int)
bmask_02 = np.where(bmask, 2, 0)

merged_masks = wmask_01 + bmask_02

nbrows, nbcols = merged_masks.shape

###Take worm blob


# Assume 'binary_image' is a binary image with the green blobs as the foreground.
# Apply connected components analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((wmask_01 * 255).astype(np.uint8), connectivity=8)

# Find the label of the largest component, ignoring the background
# The first entry in the stats array corresponds to the background
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

# Create a mask of the largest component
largest_component_mask = (labels == largest_label).astype(np.uint8)*255

# You can now use 'largest_component_mask' to isolate the largest blob in the original image



###Measure skeleton

from skimage.morphology import medial_axis, skeletonize

# Assume 'largest_component_mask' is the binary mask of the largest blob

# Convert the blob to its skeleton
#skeleton = skeletonize(largest_component_mask > 0)

# Optionally, compute the medial axis (this also gives the distances which we ignore here)
medial_axis, distance = medial_axis(largest_component_mask > 0, return_distance=True)

# Find the endpoints of the skeleton or medial axis. This is application-specific and
# you might need a custom function depending on the shape of your blob.

# Once you have the endpoints, trace the path along the skeleton to measure its length.
# This is a non-trivial task and depends on the topology of your skeleton.
# One approach is to use graph search algorithms if you treat the skeleton as a graph.

# Measure the length of the skeleton (this is a rough measurement in pixels)
length = np.sum(medial_axis)

# If you need the exact path length from one end to the other, it will require additional steps.


###Correct for pinhole worm

# Now, 'distance_transform' contains the distance from each pixel to the nearest boundary
# And 'medial_axis' is a binary image with 'True' on the medial axis

# To get the distances for just the medial axis points:
medial_axis_distances = distance[medial_axis]

mean_wormwidth = np.mean(medial_axis_distances)

#if pinhole worm
final_wormlength = length - mean_wormwidth

final_wormmeanwidth = mean_wormwidth * 2

final_wormarea = wmask = masks[1]['area']

contours, hierarchy = cv2.findContours(largest_component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
final_perimeter = 0
for contour in contours:
    final_perimeter += cv2.arcLength(contour, True)  # True for closed contours (first and last points connected)



#Save array as img
mediala = np.where(medial_axis, 3, 0)
overlay = wmask_01 + bmask_02 + mediala

img_array = np.uint8(labels) * 50 
img = Image.fromarray(img_array)
img.save('labels.png')