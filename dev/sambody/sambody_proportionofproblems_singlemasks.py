import numpy as np
#import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
sys.path.append("..")
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import glob
import os
import skimage
from skimage.morphology import medial_axis, skeletonize
from skimage.measure import label
from scipy.ndimage import convolve
import scipy
import pickle

#Count normal worms
normalfolder = "/home/maxime/prg/phd/dev/sambody/maskproblems_classification_training_wormmaskonly/normal/"

normal_list = []
for path in glob.glob(normalfolder + "*.png"):
    print(path)
    img_ids = path.split("/")[-1].removesuffix(".png").split("_")
    condition = img_ids[1] + img_ids[4].removeprefix("d")
    normal_list.append(condition)

np.unique(normal_list, return_counts=True)



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
    
    if tip1_width < midbody_width*0.97 and tip2_width < midbody_width*0.97:
        isfolded = False
    else:
        isfolded = True

    return isfolded
    #True, save to folded and next
    #False, continue


###If peaks in pruned medial axis ordered:
def merged_anomaly(wormmask):
    
    pruned_mediala, distance, medial_axis_distances_sorted = pruned_medialaxis(wormmask=wormmask)
    peaks, properties = scipy.signal.find_peaks(medial_axis_distances_sorted, prominence=45)
    if len(peaks) == 0:
        anomaly = False
    else:
        anomaly = True

    return anomaly
    #True, save to mergedanomaly and next
    #False, continue



with open("/home/maxime/prg/phd/dev/sambody/final_worm_mask.pkl", "rb") as file:
    allimages_finalwormmask = pickle.load(file)

multilist = []
pinlist = []
foldlist = []
mergedlist = []
normal = []

#path = "/home/maxime/prg/phd/dev/data/d_p4_05a_d6.png"
for dict_ in allimages_finalwormmask:
    if dict_.get('img_id') == path:
        wmask_01 = dict_.get('final_wormmask')
#pma, d, orderedma = pruned_medialaxis(wmask_01)
#png = 20
#wormmask = wmask_01

for png in range(len(allimages_finalwormmask)):
    print(png)
    path = allimages_finalwormmask[png]['img_id']
    print(path)
    wmask_01 = allimages_finalwormmask[png]['final_wormmask']
    imgh, imgw, _ = cv2.imread(path).shape

    #wmask_01 = np.where(wmask == 1, 1, 0)

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


img_array = np.uint8(wmask_01) * 100 
img = Image.fromarray(img_array)
img.save('wmask_01.png')

