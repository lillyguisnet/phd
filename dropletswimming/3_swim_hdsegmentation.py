import os
import sys
sys.path.append("/home/maxime/prg/phd/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path
import re
from tqdm import tqdm
import pickle
from sam2.build_sam import build_sam2_video_predictor
import shutil
import h5py

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


def calculate_fixed_crop_window(video_segments, original_size, crop_size):
    orig_height, orig_width = original_size
    centers = []
    empty_masks = 0
    total_masks = 0

    for frame_num in sorted(video_segments.keys()):
        mask = next(iter(video_segments[frame_num].values()))
        total_masks += 1
        y_coords, x_coords = np.where(mask[0])
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            center_x = (x_coords.min() + x_coords.max()) // 2
            center_y = (y_coords.min() + y_coords.max()) // 2
            centers.append((center_x, center_y))
        else:
            empty_masks += 1
            centers.append((orig_width // 2, orig_height // 2))

    if empty_masks > 0:
        crop_size = 800
        avg_center_x = sum(center[0] for center in centers) // len(centers)
        avg_center_y = sum(center[1] for center in centers) // len(centers)
        centers = [(avg_center_x, avg_center_y)] * len(centers)

    crop_windows = []
    for center_x, center_y in centers:
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(orig_width, left + crop_size)
        bottom = min(orig_height, top + crop_size)
        
        # Adjust if crop window is out of bounds
        if right == orig_width:
            left = right - crop_size
        if bottom == orig_height:
            top = bottom - crop_size
        
        crop_windows.append((left, top, right, bottom))

    return crop_windows, (crop_size, crop_size), empty_masks, total_masks

def process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate fixed crop windows
    crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(video_segments, original_size, 110)
    
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get crop window for this frame
        left, top, right, bottom = crop_windows[idx]
        
        # Crop the frame
        cropped_frame = frame[top:bottom, left:right]
        
        # Ensure the cropped frame is exactly crop_size x crop_size
        if cropped_frame.shape[:2] != (crop_height, crop_width):
            cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
        
        # Save the cropped frame
        cv2.imwrite(os.path.join(output_folder, frame_file), cropped_frame)
    
    return len(frame_files), (crop_height, crop_width)


#Predict
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def swim_hdsegmentation(video_dir, fframe_segments_file):
    #Get detailed segmentation on fframe coordinates
    with open(fframe_segments_file, 'rb') as file:
        ffvideo_segments = pickle.load(file)
       
    #If the segmentation file is an intermediate crop, use the temp_cropdir images as base video
    if "segmentation_800.pkl" in fframe_segments_file:
        temp_cropdir = '/home/maxime/prg/phd/dropletswimming/temp_cropdir'
        temp_cropdir2 = '/home/maxime/prg/phd/dropletswimming/temp_cropdir2'
        os.makedirs(temp_cropdir2, exist_ok=True)
        # Read one frame to get original dimensions
        frame_files = sorted([f for f in os.listdir(temp_cropdir) if f.endswith('.jpg')])
        first_frame = cv2.imread(os.path.join(temp_cropdir, frame_files[0]))
        original_size = first_frame.shape[:2]
        num_frames, crop_size = process_frames_fixed_crop(temp_cropdir, temp_cropdir2, ffvideo_segments, original_size)
        print(f"Processed {num_frames} intermediate crop frames.")
        print(f"Fixed crop size again: {crop_size[1]}x{crop_size[0]}")
        shutil.rmtree(temp_cropdir)
    #If it's the fframe segmentation, then use original video images    
    else:
        #Create cropped images centered on the fframe segmentation coordinates
        temp_cropdir = '/home/maxime/prg/phd/dropletswimming/temp_cropdir'
        os.makedirs(temp_cropdir, exist_ok=True)
        # Read one frame to get original dimensions
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        first_frame = cv2.imread(os.path.join(video_dir, frame_files[0]))
        original_size = first_frame.shape[:2]

        num_frames, crop_size = process_frames_fixed_crop(video_dir, temp_cropdir, ffvideo_segments, original_size)
        print(f"Processed {num_frames} frames.")
        print(f"Fixed crop size: {crop_size[1]}x{crop_size[0]}")

    #Copy prompt frame to the video directory at last position based on crop size
    if "segmentation_800.pkl" in fframe_segments_file:
        prompt_frame = "/home/maxime/prg/phd/dropletswimming/crop000600.jpg" #Second pass for HD crop size	
        shutil.copy(prompt_frame, os.path.join(temp_cropdir2, "000300.jpg"))
        temp_cropdir=temp_cropdir2         
    elif crop_size[0] == 800: 
        prompt_frame = "/home/maxime/prg/phd/dropletswimming/crop800.jpg" #Intermediate crop size pass
        shutil.copy(prompt_frame, os.path.join(temp_cropdir, "000300.jpg"))
    else:
        prompt_frame = "/home/maxime/prg/phd/dropletswimming/crop000600.jpg" #First time HD crop size
        shutil.copy(prompt_frame, os.path.join(temp_cropdir, "000300.jpg"))    
    

    # scan all the jpg frame names in this directory
    frame_names = [
        p for p in os.listdir(temp_cropdir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=temp_cropdir)

    #Add click on prompt frame
    ann_frame_idx = 300  #frame index 
    ann_obj_id = 1  #object id
    if crop_size[0] == 110:
        points = np.array([[58, 54]], dtype=np.float32) #110x110 crop size
    else:
        points = np.array([[343, 533]], dtype=np.float32) #800x800 crop size [343, 533]
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(temp_cropdir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.savefig("tstclick.png")
    plt.close()

    #Propagate to 'video' in reverse and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    #for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=False):
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    #Remove entry for the prompt frame
    del video_segments[ann_frame_idx]

    #Check if some frames are empty
    empty_frames = []
    low_detection_frames = []
    high_detection_frames = []
    for frame, obj_dict in video_segments.items():
        if all(not mask.any() for mask in obj_dict.values()):
            empty_frames.append(frame)
        elif sum(mask.sum() for mask in obj_dict.values()) <= 200:
            low_detection_frames.append(frame)    
        elif sum(mask.sum() for mask in obj_dict.values()) >= 1000:
            high_detection_frames.append(frame)   
    if empty_frames:
        print(f"!!! Empty frames: {empty_frames}")
    else:
        print("Yay! No empty frames found, yay!")
    if low_detection_frames:
        print(f"!!! Frames with 200 or fewer true elements: {low_detection_frames}")
    else:
        print("Yay! No frames with 200 or fewer true elements found, yay!")
    if high_detection_frames:
        print(f"!!! Frames with 1000 or more true elements: {high_detection_frames}")
    else:
        print("Yay! No frames with 1000 or more true elements found, yay!")


    #Save the results as h5 file 
    if crop_size[0] == 110:
        #Remove the content of the temp_cropdir
        shutil.rmtree(temp_cropdir)
        save_name = "/home/maxime/prg/phd/dropletswimming/data_analyzed/hd_segmentations/" + os.path.basename(video_dir) + "_hdsegmentation.h5"
        with h5py.File(save_name, 'w') as hf:
            # Iterate through the main dictionary
            for key, value in video_segments.items():
                # Create a group for each main key
                group = hf.create_group(str(key))
                # Iterate through the nested dictionary
                for sub_key, array in value.items():
                    # Save each array as a dataset
                    group.create_dataset(str(sub_key), data=array)
        print(f"Saved! {save_name}")
    else:
        os.remove(os.path.join(temp_cropdir, "000300.jpg"))
        save_name = "/home/maxime/prg/phd/dropletswimming/data_analyzed/intermediate_crops/" + os.path.basename(video_dir) + "_segmentation_800.pkl"
        with open(save_name, 'wb') as file:
            pickle.dump(video_segments, file)
        print(f"!!! We're in a pickle!!! {save_name}")
                

    return video_segments, crop_size[0], save_name


###Get hdsegmentation###

def get_swim_hdsegmentation(or_vid):
    #Get the fframe video directory from the original video path
    or_vid_path = Path(or_vid)
    sub_folder_name = or_vid_path.parent.name
    video_name = or_vid_path.stem
    new_folder_name = f"{sub_folder_name}-{video_name}"
    output_dir = Path("/home/maxime/prg/phd/dropletswimming/data_foranalysis/visc05")
    ffvideo_dir = str(output_dir / new_folder_name)

    #Infer fframe_segments_file for the video
    fframe_segments_dir = Path("/home/maxime/prg/phd/dropletswimming/data_analyzed/fframe_segmentations")
    fframe_segments_file = str(fframe_segments_dir / f"{new_folder_name}_fframe_segmentation.pkl")

    video_segments, crop_size, save_name = swim_hdsegmentation(ffvideo_dir, fframe_segments_file)

    if crop_size != 110:
        video_segments, crop_size, save_name = swim_hdsegmentation(ffvideo_dir, save_name)

    return video_segments, crop_size, save_name




or_vid = '/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022122501-0000.avi'

video_segments, crop_size, save_name = get_swim_hdsegmentation(or_vid)





###visc05###
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03222022165538-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03222022172859-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03222022173400-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03252022111734-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03252022112259-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03252022112831-0000.avi'

'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03222022180324-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03222022181147-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03222022182050-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03252022113446-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03252022114209-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03252022114730-0000.avi'

'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03222022195635-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03222022200832-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03222022201347-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03252022115515-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03252022120041-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03252022120644-0000.avi'

'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03222022204030-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03222022204518-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03222022205045-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022121224-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022121746-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022122501-0000.avi'



from PIL import Image

mask = video_segments[289][1][0]

image_array = np.uint8(mask	 * 255)
image = Image.fromarray(image_array)
image.save('tst.png')