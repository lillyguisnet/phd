import os
import sys
sys.path.append("/home/maxime/prg/phd/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path
import shutil
import re
from tqdm import tqdm
import pickle
import h5py
from sam2.build_sam import build_sam2_video_predictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def fullframe_segmentation_withcommonprompt(video_dir):
    sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    prompt_path = "/home/maxime/prg/phd/crawlingtracking/000600.jpg"
    # Extract the filename from the source path
    filename = os.path.basename(prompt_path)
    
    # Create the full destination path
    destination_path = os.path.join(video_dir, filename)
    
    # Copy the image to the destination folder
    shutil.copy2(prompt_path, destination_path)
    
    # Replace this comment with your image processing code
    print(f"Processing video_dir: {video_dir}")

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    ann_frame_idx = 600
    ann_obj_id = 1
    points = np.array([[1317, 1481]], dtype=np.float32) #full frame common prompt
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
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.savefig("promptclick.png")
    plt.close()

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    #Save segmentation dict
    save_dir = "/home/maxime/prg/phd/crawlingtracking/final_data/fullframe_segmentations/"
    save_name = save_dir + os.path.basename(os.path.normpath(video_dir)) + ".h5"

    # Convert the dictionary to a single large numpy array
    frames = sorted(video_segments.keys())
    all_masks = np.array([list(video_segments[frame].values())[0].astype(np.uint8) for frame in frames])

    with h5py.File(save_name, 'w') as f:
        # Create a single large dataset
        f.create_dataset('masks', data=all_masks, compression="gzip", compression_opts=9)

        # Save frame numbers as a separate dataset for reference
        f.create_dataset('frame_numbers', data=np.array(frames))
    
    if not os.path.exists(save_name):
        raise IOError(f"File {save_name} was not created successfully.")
    else:
        print(f"Successfully saved {save_name}")

    # Delete the copied image after processing
    os.remove(destination_path)
    
    print(f"Successfully processed and deleted {filename}")


video_dir = "/home/maxime/prg/phd/crawlingtracking/data_foranalysis/food/food-a-02252022132222-0000"

fullframe_segmentation_withcommonprompt(video_dir)