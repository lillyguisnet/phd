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
from skimage.morphology import medial_axis, skeletonize
import scipy

sam_checkpoint = "phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
default_generator = SamAutomaticMaskGenerator(sam)


""" 
DONE Non-continuous worm blob
DONE Pinhole worm
- More than 2 masks, but good worm
- More than 2 masks, but touching worm (prob substract)
DONE 2 masks, but merged anomaly (8, 51)
DONE Folded worm (either end width smaller than middle width) (2)
 """



img = cv2.imread('phd/dev/sambody/tstimgs/body_6.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = default_generator.generate(img)
print(len(masks))
print(masks[0].keys())


wmask = masks[1]['segmentation']
bmask = masks[0]['segmentation']
omask = masks[2]['segmentation']
wmask = wmask.astype(int) - omask.astype(int)
wmask_01 = np.where(wmask == 1, 1, 0)
bmask_02 = np.where(bmask, 2, 0)
omask_03 = np.where(omask, 3, 0)

merged_masks = wmask_01 + bmask_02 + omask_03

nbrows, nbcols = merged_masks.shape


###Non-continuous worm blob (works for 2 blobs)

_, binary_image = cv2.threshold((wmask_01 * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

blob1 = contours[0]
blob2 = contours[1]

# Find the closest points between the two blobs
def find_closest_points(contour1, contour2):
    min_distance = np.inf
    points_pair = None

    for point1 in contour1:
        for point2 in contour2:
            distance = np.linalg.norm(point1[0] - point2[0])
            if distance < min_distance:
                min_distance = distance
                points_pair = (point1[0], point2[0])
    
    return points_pair

closest_points = find_closest_points(blob1, blob2)

#Bridge gap between blobs
cv2.line(binary_image, tuple(closest_points[0]), tuple(closest_points[1]), (255), thickness=3)
""" kernel = np.ones((5,5),np.uint8)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel) """
""" 
# Define a structuring element for dilation                ###Dilation affects too much of rest of worm
kernel_size = 3  # This size might need to be adjusted
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
# Dilate the image to merge the blobs
dilated_image = cv2.dilate((wmask_01 * 255).astype(np.uint8), kernel, iterations=20)
# You may want to iterate with dilation and find contours to check if blobs have merged
# and to ensure you're not over-dilating. """

##Find pinhole and number of worm blobs

#num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((wmask_01 * 255).astype(np.uint8), connectivity=8)

contours, hierarchy = cv2.findContours((wmask_01 * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # More than 1 contour with no child/parent == disconnected blobs.
    # If pinhole, contours with child/parent.


###More than 2 masks, but good worm

merged_masks = []
for nb in range(len(masks)):
    mask = masks[nb]['segmentation']
    bimask = np.where(mask, nb, 0)
    merged_masks.append(bimask)
all_masks = sum(merged_masks)

contours, hierarchy = cv2.findContours((wmask_01 * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


###2 masks, but merged anomaly (8, 51)

""" from scipy.interpolate import splprep, splev

_, binary_image = cv2.threshold((wmask_01 * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume we're working with the largest contour
contour = max(contours, key=cv2.contourArea)

# Fit a spline to the contour
contour = contour.squeeze()
tck, u = splprep([contour[:, 0], contour[:, 1]], s=5, k =1, per=True)

# Evaluate the spline and calculate the first and second derivatives
num_points = len(contour)  # The number of points to evaluate on the spline
u_new = np.linspace(u.min(), u.max(), num_points)
dx, dy = splev(u_new, tck, der=1)
d2x, d2y = splev(u_new, tck, der=2)

# Curvature calculation for each point
curvature = np.abs(dx * d2y - dy * d2x) / np.power(dx**2 + dy**2, 3/2)
# The 'curvature' array now holds the curvature values for each point on the contour

# Evaluate the spline to get the (x, y) coordinates
x_new, y_new = splev(u_new, tck)

# Zip the coordinates and curvature values together
curvature_points = list(zip(x_new, y_new, curvature))

# Now, for each (x, y) coordinate, you have a corresponding curvature value.
# Keep in mind that these (x, y) are floating-point numbers, and you may need to round them
# to the nearest integer pixel values for certain applications.
curvature_points = [(int(round(x)), int(round(y)), curv) for x, y, curv in curvature_points]

# Optionally, you can create an image to visualize the curvature on the original contour
curvature_image = cv2.UMat(np.zeros_like((wmask_01 * 255).astype(np.uint8)))
for x, y, curv in curvature_points:
    # Map the curvature value to a color
    # Here we just scale it to a value between 0 and 255
    color = int(min(curv * 200, 255))  # scale_factor to be defined based on your curvature range
    cv2.circle(curvature_image, (x, y), 1, color, -1)

# Display or save the curvature image
cv2.imwrite('curvature_image.png', curvature_image)
cv2.destroyAllWindows() """

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((wmask_01 * 255).astype(np.uint8), connectivity=8)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
largest_component_mask = (labels == largest_label).astype(np.uint8)*255

medial_axis, distance = skimage.morphology.medial_axis(largest_component_mask > 0, return_distance=True)

from skimage.measure import label
from scipy.ndimage import convolve

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

pruned_mediala = np.zeros(wmask.shape)

for coord in range(len(longest_length[3])):
    pruned_mediala[longest_length[3][coord]] = 1 


length = np.sum(pruned_mediala)

medial_axis_distances = distance[np.array(pruned_mediala, dtype=bool)]

mean_wormwidth = np.mean(medial_axis_distances)

wormlength = 0 + np.arange(0, len(medial_axis_distances))

medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
wormlength = 0 + np.arange(0, len(medial_axis_distances_sorted))

#Find local maxima along curve to identify potential deviations of the medial axis
mads6 = medial_axis_distances_sorted

""" plt.plot(wormlength, medial_axis_distances_sorted)
plt.savefig("tst1.png")
plt.close() """

""" with np.printoptions(threshold=np.inf):
    print(wmask_01[0:10]) """

""" y_coordinates = np.argmax(pruned_mediala, axis=0)
# Filter out the zeros which are not part of the curve
y_coordinates = y_coordinates[y_coordinates != 0] """
peaks, properties = scipy.signal.find_peaks(mads6, prominence=35)

peaks6_35p = peaks




###Folded worm (either end width smaller than middle width) (2, 10, 43)

""" num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((wmask_01 * 255).astype(np.uint8), connectivity=8)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
largest_component_mask = (labels == largest_label).astype(np.uint8)*255

medial_axis, distance = skimage.morphology.medial_axis(largest_component_mask > 0, return_distance=True)
 """
#Get pruned ma
length = np.sum(medial_axis)

medial_axis_distances = distance[medial_axis]

mean_wormwidth = np.mean(medial_axis_distances)

midbody_width = medial_axis_distances[int(np.floor(len(medial_axis_distances)/2))]

#Mean of 5% tip of length
tip1_width = np.mean(medial_axis_distances[0:10])
tip2_width = np.mean(medial_axis_distances[-11:-1])
    #if both tips not at least 3/4 width midbody: folding



#Save array as img
mediala = np.where(pruned_mediala, 3, 0)
overlay = wmask_01 + mediala
overlay[:,596+367] = 0

canvas = np.zeros((1500, 1500, 3), dtype=np.uint8)
# Iterate through the list of tuples and mark the pixels in the canvas
for x, y in paths[1][0]:
    canvas[y, x] = [255, 0, 0] 

img_array = np.uint8(overlay) * 100 
img = Image.fromarray(img_array)
img.save('overlay.png')

import matplotlib.pyplot as plt
plt.hist(medial_axis_distances, bins = 20)
plt.savefig("distance51.png")
plt.close()