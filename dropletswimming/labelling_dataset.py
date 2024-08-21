import cv2
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.4]])
        img[m] = color_mask
    ax.imshow(img)

def is_on_edge(x, y, w, h, img_width, img_height):
    # Check left edge
    if x == 0:
        return True
    # Check top edge
    if y == 0:
        return True
    # Check right edge
    if x + w == img_width:
        return True
    # Check bottom edge
    if y + h == img_height:
        return True
    return False


sam_checkpoint = "/home/maxime/prg/phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    #points_per_batch = 128,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.6,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,
)


base_folder = "/home/maxime/prg/phd/dropletswimming/data_original"
output_folder = "/home/maxime/prg/phd/dropletswimming/tolabeldata"
num_frames = 2000
temp_plot_filename = 'temp_plot.png'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Dynamically find all .avi files in subfolders
video_paths = []

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".avi"):
            video_paths.append(os.path.join(root, file))

total_frames = sum(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT) for video in video_paths)

frame_indices = random.sample(range(int(total_frames)), num_frames)
current_frame_index = 0
saved_frames = 0

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for vidf in range(total_video_frames):
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame_index in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(frame)
            hi, wi, _ = frame.shape

            plt.figure(figsize=(20,20))
            plt.imshow(frame)
            show_anns(masks)
            plt.axis('off')
            plt.savefig(temp_plot_filename, bbox_inches='tight', pad_inches=0)
            plt.close()

            for maskpred in range(len(masks)):
                bbox = masks[maskpred]['bbox']
                if is_on_edge(*bbox, hi, wi) == True:
                    continue

                cutout = img*np.expand_dims(masks[maskpred]['segmentation'], axis=-1)
                x, y, w, h = bbox
                cropped_image = cutout[y:y+h, x:x+w]

                plot_image = Image.open(temp_plot_filename)
                image_np_pil = Image.fromarray(cropped_image)

                plot_image_height = plot_image.size[1]
                image_np_pil_height = image_np_pil.size[1]

                if plot_image_height != image_np_pil_height:
                    # Calculate aspect ratio of numpy array image
                    aspect_ratio_np_image = image_np_pil.size[0] / image_np_pil.size[1]
                    new_width = int(aspect_ratio_np_image * plot_image_height)
                    image_np_pil = image_np_pil.resize((new_width, plot_image_height), Image.ANTIALIAS)

                
                # Combine the two images side by side
                total_width = plot_image.size[0] + image_np_pil.size[0]
                max_height = max(plot_image.size[1], image_np_pil.size[1])

                combined_image = Image.new('RGB', (total_width, max_height))
                combined_image.paste(plot_image, (0, 0))
                combined_image.paste(image_np_pil, (plot_image.size[0], 0))


                savename = os.path.join(output_folder, "c" + str(maskpred) + "_f" + str(vidf) + "_" + video_path.split(os.sep)[-2] + "_" + video_path.split(os.sep)[-1].replace(".avi","") + ".png")
                combined_image.save(savename)


                #cv2.imwrite(os.path.join(output_folder, "c" + str(maskpred) + "_f" + str(vidf) + "_" + video_paths[0].split(os.sep)[-2] + "_" + video_paths[0].split(os.sep)[-1].replace(".avi","") + ".png"), frame)
            
            saved_frames += 1

            if saved_frames >= num_frames:
                continue
            
        current_frame_index += 1

    cap.release()

