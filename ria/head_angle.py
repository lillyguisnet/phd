import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shutil
from pathlib import Path
import cv2
import h5py

video_dir = '/home/lilly/phd/ria/data_foranalysis/videotojpg/AG-MMH99_10s_20190306_02'
output_filename = '/home/lilly/phd/ria/data_analyzed/head_segmentation/AG-MMH99_10s_20190306_02_headsegmentation.h5'

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

head_segments = load_cleaned_segments_from_h5(output_filename)


