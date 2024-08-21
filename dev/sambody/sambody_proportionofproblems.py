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
from skimage.measure import label
from scipy.ndimage import convolve
import scipy

sam_checkpoint = "phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
default_generator = SamAutomaticMaskGenerator(sam)

"""
For each image:
    If more than one mask:
        Save to more than one mask
    If contours have child/parent:
        Save to pinhole
    If either end width smaller than middle width:
        Save to folded
    If peaks in pruned medial axis ordered:
        Save to merged anomaly
    Else:
        Save as perfect worm (but maybe merge blobs)
"""

""" img = cv2.imread('phd/dev/sambody/tstimgs/body_2.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = default_generator.generate(img)
##---> Get worm mask
wmask = masks[1]['segmentation']
wmask_01 = np.where(wmask == 1, 1, 0) """


###If more than one mask:
def multimask(masks):
    if len(masks) > 2 or len(masks) == 1:
        multimask = True
    else:
        multimask = False
    return multimask
    #True, save to multimask and next
    #False, continue

###If zero mask:
def zeromask(masks):
    if len(masks) == 1:
        noworm = True
    else:
        noworm = False
    return noworm
    #True, save to no worm and next
    #False, continue



###If contours have child/parent:
def ispinhole(wormmask):
    contours, hierarchy = cv2.findContours((wormmask * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    pinhole = np.any(hierarchy[0][:, -1] != -1)
    return pinhole
    #True, save to pinhole and next
    #False, continue


#Get pruned medial axis
def pruned_medialaxis(wormmask):
    #Take largest blob of mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((wormmask * 255).astype(np.uint8), connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component_mask = (labels == largest_label).astype(np.uint8)*255

    #Find branches and end points of raw medial axis
    medial_axis, distance = skimage.morphology.medial_axis(largest_component_mask > 0, return_distance=True)
    structuring_element = np.array([[1, 1, 1],
                                    [1, 10, 1],
                                    [1, 1, 1]], dtype=np.uint8)
    neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
    end_points = np.where(neighbours == 11, 1, 0)
    branch_points = np.where(neighbours > 12, 1, 0)
    labeled_branches = label(branch_points, connectivity=2)
    branch_indices = np.argwhere(labeled_branches > 0)
    end_indices = np.argwhere(end_points > 0)
    indices = np.concatenate((branch_indices, end_indices), axis=0)

    #Choose longest path between points as pruned medial axis
    paths = []
    for start in range(len(indices)):
        for end in range(len(indices)):
            startid = tuple(indices[start])
            endid = tuple(indices[end])
            route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
            length = len(route)
            paths.append([startid, endid, length, route, weight])
    longest_length = max(paths, key=lambda x: x[2])
    pruned_mediala = np.zeros(wormmask.shape)
    for coord in range(len(longest_length[3])):
        pruned_mediala[longest_length[3][coord]] = 1

    medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]

    return pruned_mediala, distance, medial_axis_distances_sorted


###If either end width smaller than middle width:
def folded(wormmask):

    pruned_mediala, distance, distsort = pruned_medialaxis(wormmask=wormmask)

    length = np.sum(pruned_mediala)
    medial_axis_distances = distance[np.array(pruned_mediala, dtype=bool)]
    mean_wormwidth = np.mean(medial_axis_distances)
    midbody_width = medial_axis_distances[int(np.floor(len(medial_axis_distances)/2))]
    tip_length = int(np.floor(length*0.03))
    
    tip1_width = np.mean(distsort[0:tip_length])
    tip2_width = np.mean(distsort[-tip_length-1:-1])
    
    if tip1_width < midbody_width*0.95 and tip2_width < midbody_width*0.95:
        isfolded = False
    else:
        isfolded = True

    return isfolded
    #True, save to folded and next
    #False, continue


###If peaks in pruned medial axis ordered:
def merged_anomaly(wormmask):
    
    pruned_mediala, distance, medial_axis_distances_sorted = pruned_medialaxis(wormmask=wormmask)
    peaks, properties = scipy.signal.find_peaks(medial_axis_distances_sorted, prominence=35)
    if len(peaks) == 0:
        anomaly = False
    else:
        anomaly = True

    return anomaly
    #True, save to mergedanomaly and next
    #False, continue


imgs_path = "/home/maxime/prg/phd/dev/data/"

#multilist = []
#pinlist = []
#foldlist = []
#mergedlist = []
#normal = []

path = "/home/maxime/prg/phd/dev/data/a_p1_01_d2.png"
pma, d, orderedma = pruned_medialaxis(wmask_01)

for path in glob.glob(imgs_path + "*.png"):
    print(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = default_generator.generate(image)

    if multimask(masks) == True:
        multilist.append(path)
        continue

    wmask = masks[-1]['segmentation']
    wmask_01 = np.where(wmask == 1, 1, 0)

    if ispinhole(wmask_01) == True:
        pinlist.append(path)
        continue

    if folded(wmask_01) == True:
        foldlist.append(path)
        continue    

    if merged_anomaly(wmask_01) == True:
        mergedlist.append(path)
        continue

    else:
        print("i'm here")
        normal.append(path)



#zeroworm = []
#manythings = []

for manymasks in range(len(multilist)):
    print(multilist[manymasks])
    image = cv2.imread(multilist[manymasks])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = default_generator.generate(image)

    if zeromask(masks) == True:
        print("No worm")
        zeroworm.append(multilist[manymasks])
    else:
        manythings.append(multilist[manymasks])


with open("manythings.txt", "w") as file:
    for item in manythings:
        file.write("%s\n" % item)

