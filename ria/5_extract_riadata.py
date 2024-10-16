import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        nb_frames = 0
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            nb_frames += 1

    
    print(f"{nb_frames} frames loaded from {filename}")
    return cleaned_segments

filename = '/home/lilly/phd/ria/data_analyzed/cleaned_segments/AG-MMH122_10s_20190830_04_crop_riasegmentation_cleanedsegments.h5'

cleaned_segments = load_cleaned_segments_from_h5(filename)








