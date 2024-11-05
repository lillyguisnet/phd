import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
from typing import Dict, Set, List, Tuple
import itertools


def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            
            # Debug print
            print(f"Loading frame {frame_idx}")
    
    print(f"Cleaned segments loaded from {filename}")
    return cleaned_segments



aligned_segments = "ria/data_analyzed/aligned_segments/AG-MMH99_10s_20190306_02_crop_riasegmentation_alignedsegments.h5"
loaded_segments = load_cleaned_segments_from_h5(aligned_segments)


first_frame = loaded_segments[0]

def get_centroid(mask):
    # Get indices of True values - mask is already 2D
    y_indices, x_indices = np.where(mask[0])
    if len(x_indices) == 0:  # If no True values found
        return None
    
    # Calculate centroid
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    return (centroid_x, centroid_y)

def get_relative_position(first_frame):
    # Get centroids for objects 2 and 4
    centroid2 = get_centroid(first_frame[2])
    centroid4 = get_centroid(first_frame[4])
    
    if centroid2 is None or centroid4 is None:
        return "One or both objects not found in frame"
    
    # Compare x-coordinates of centroids
    if centroid4[0] < centroid2[0]:
        return "left"
    else:
        return "right"
    


position = get_relative_position(first_frame)