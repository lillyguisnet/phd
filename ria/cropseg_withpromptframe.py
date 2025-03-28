import os
import sys
sys.path.append("/home/lilly/phd/segment-anything-2")

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import shutil
from pathlib import Path
import cv2

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

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


def show_points(coords, labels, ax, marker_size=26):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


video_dir = "/home/lilly/phd/ria/tst_free/tojpg/original-2"
'/home/maxime/prg/phd/ria/tstvideo/ria-nt03_crop'

#region [add prompt frames]
##Add prompt frames
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

#Add 1st prompt frame as last frame
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/1.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name)) 

#Add 2nd prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/2.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 3rd prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/3.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 4rd prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/4.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 5th prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/5.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 6th prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/6.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 7th prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/7.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 8th prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/8.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 9th prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/9.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 10th prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/10.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#Add 11th prompt frame as last frame
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompt_frame = "/home/maxime/prg/phd/ria/prompt_frames/11.jpg" 
# Find the largest frame number
largest_frame_number = max(int(os.path.splitext(p)[0]) for p in frame_names)
prompt_frame_name = f"{largest_frame_number+1:06d}.jpg"
shutil.copy(prompt_frame, os.path.join(video_dir, prompt_frame_name))

#endregion [add prompt frames]

###Set up the inference state###
# scan all the jpg frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)


#region [add first prompt frame]


###Add prompts to prompt frame###
prompts = {}
#NRD
ann_frame_idx = 701 #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[67, 41]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 701  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[47, 58], [46, 56], [45, 58], [49, 53], [52, 50]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 1, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 701  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[42, 71], [40, 72], [42, 69], [46, 63], [55, 48], [63, 41]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add first prompt frame]


#region [add second prompt frame]

###Add prompts to second prompt frame###
prompts = {}
#NRD
ann_frame_idx = 702  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[65, 40]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 702  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[43, 56], [43, 53], [42, 56], [47, 50], [50, 46], [39, 61], [54, 56]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 1, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 702  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[30, 80], [34, 70], [38, 62], [43, 58], [55, 42], [61, 41]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 2]


#region [add 3rd prompt frame]


###Add prompts to third prompt frame###
prompts = {}
#NRD
ann_frame_idx = 703  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[65, 40]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 703  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[44, 56], [43, 53], [42, 58], [47, 50], 
                   [50, 46], [41, 60], [54, 56], [58, 41], [39, 65], [34, 78], [53, 43]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 1, 
                   0, 0, 0, 0, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 703  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[30, 80], [34, 70], [38, 62], [43, 58], [55, 42], [61, 41]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()



#endregion [add prompt frame 3]


#region [add 4th prompt frame]


###Add prompts to 4th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 704  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[70, 40]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 704  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[39, 61], [41, 59]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 704  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[28, 79], [34, 70], [37, 63],
                   [38, 62]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1,
                   0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()

#endregion [add prompt frame 4]


#region [add 5th prompt frame]


###Add prompts to 5th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 705  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[68, 40]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 705  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[39, 60], [41, 57]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 705  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[28, 79], [34, 70], [36, 62], [32, 73],
                   [37, 61]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 1,
                   0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 5]


#region [add 6th prompt frame]


###Add prompts to 6th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 706  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[52, 19]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 706  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[88, 65]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 706  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[87, 74], [89, 85], [88, 98],
                   [85, 71]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1,
                   0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 6]


#region [add 7th prompt frame]


###Add prompts to 7th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 707  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[100, 41]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 707  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[48, 38], [43, 42], 
                   [38, 44], [20, 62]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1,
                   0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 707  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[21, 58], [36, 46], [11, 76],
                   [70, 65], [38, 63], [50, 43], [40, 44]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1,
                   0, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 7]


#region [add 8th prompt frame]


###Add prompts to 8th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 708  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[69, 46]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 708  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[45, 64], [53, 50],
                   [42, 66]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1,
                   0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 708  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[40, 68], [34, 82], [37, 72],
                   [41, 66]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1,
                   0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 8]


#region [add 9th prompt frame]


###Add prompts to 9th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 709  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[40, 37]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 709  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[59, 46], [67, 55],
                   [69, 57], [65, 58], [68, 58], [72, 63]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1,
                   0, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 709  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[69, 60], [75, 72], 
                   [68, 58], [63, 55], [70, 70]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1,
                   0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 9]


#region [add 10th prompt frame]


###Add prompts to 10th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 710  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[41, 52]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 710  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[61, 64], [67, 72],
                   [68, 74], [67, 75], [75, 79], [74, 78], [73, 77], [72, 76]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1,
                   0, 0, 0, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 710  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[69, 78], [74, 86],
                   [68, 75] ], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 
                   0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 10]



#region [add 11th prompt frame]


###Add prompts to 11th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 711  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[41, 70]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 711  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[62, 81], [66, 89],
                   [68, 92], [69, 91]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1,
                   0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 711  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[69, 93], [74, 101],
                   [68, 90] ], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 
                   0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#endregion [add prompt frame 11]


###Test free
###Add prompts to 11th prompt frame###
prompts = {}
#NRD
ann_frame_idx = 200  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[31, 40], [40, 40]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1, 0], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#NRV
ann_frame_idx = 200  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[42, 21], [40, 40], [48, 19]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()


#LOOP
ann_frame_idx = 803  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[31, 15], [34, 8],
                   [31, 17], [30, 16], [31, 16] ], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 
                   0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()



video_segments = {}  # video_segments contains the per-frame segmentation results
#for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    for i, out_obj_id in enumerate(out_obj_ids)
}

# Save video segments to pickle file
with open('video_segments_free2.pkl', 'wb') as f:
    pickle.dump(video_segments, f)
    

empty_masks = {}
low_detection_masks = {}
high_detection_masks = {}
for frame, mask_dict in video_segments.items():
    for mask_id, mask in mask_dict.items():
        mask_sum = mask.sum()        
        if mask_sum == 0:
            if frame not in empty_masks:
                empty_masks[frame] = []
            empty_masks[frame].append(mask_id)
        elif mask_sum <= 200:
            if frame not in low_detection_masks:
                low_detection_masks[frame] = []
            low_detection_masks[frame].append(mask_id)
        elif mask_sum >= 1000:
            if frame not in high_detection_masks:
                high_detection_masks[frame] = []
            high_detection_masks[frame].append(mask_id)
def print_results(result_dict, condition):
    if result_dict:
        print(f"!!! Frames with masks {condition}:")
        for frame, mask_ids in result_dict.items():
            print(f"  Frame {frame}: Mask IDs {mask_ids}")
    else:
        print(f"Yay! No masks {condition} found, yay!")
print_results(empty_masks, "that are empty")
#print_results(low_detection_masks, "having 200 or fewer true elements")
print_results(high_detection_masks, "having 1000 or more true elements")



# region [add prompt within video]

ann_frame_idx = 380  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[67, 39]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()




ann_frame_idx = 380  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[47, 58]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()



ann_frame_idx = 380  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[43, 72], [41, 73], [43, 70], [46, 63], [55, 48], [63, 41]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()






#Propagate to 'video' in reverse and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    for i, out_obj_id in enumerate(out_obj_ids)
}

# Save video_segments to pickle file
with open('video_segments_free1920.pkl', 'wb') as f:
    pickle.dump(video_segments, f)

    

empty_masks = {}
low_detection_masks = {}
high_detection_masks = {}
for frame, mask_dict in video_segments.items():
    for mask_id, mask in mask_dict.items():
        mask_sum = mask.sum()        
        if mask_sum == 0:
            if frame not in empty_masks:
                empty_masks[frame] = []
            empty_masks[frame].append(mask_id)
        elif mask_sum <= 200:
            if frame not in low_detection_masks:
                low_detection_masks[frame] = []
            low_detection_masks[frame].append(mask_id)
        elif mask_sum >= 5000:
            if frame not in high_detection_masks:
                high_detection_masks[frame] = []
            high_detection_masks[frame].append(mask_id)
def print_results(result_dict, condition):
    if result_dict:
        print(f"!!! Frames with masks {condition}:")
        for frame, mask_ids in result_dict.items():
            print(f"  Frame {frame}: Mask IDs {mask_ids}")
    else:
        print(f"Yay! No masks {condition} found, yay!")
print_results(empty_masks, "that are empty")
#print_results(low_detection_masks, "having 200 or fewer true elements")
print_results(high_detection_masks, "having 5000 or more true elements")

# endregion [add prompt within video]



video_segments[0][1][0]
video_segments[1][1][0][:] = False

#open pickle file
with open('/home/maxime/prg/phd/dropletswimming/data_analyzed/fframe_segmentations/ngm-b-02212022181139-0000_fframe_segmentation.pkl', 'rb') as file:
    video_segments = pickle.load(file)

from PIL import Image

mask = video_segments[0][4][0]

image_array = np.uint8(mask	 * 255)
image = Image.fromarray(image_array)
image.save('tst.png')



###Multimask video overlay###
import cv2
import numpy as np
import os
import torch

output_video_path = "free2.mp4"

# Predefined list of visually distinct colors
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (128, 0, 128),  # Purple
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    (0, 128, 0),    # Dark Green
    (0, 128, 128),  # Teal
    (255, 128, 0),  # Orange
    (255, 0, 128),  # Deep Pink
    (128, 255, 0),  # Lime
    (255, 255, 0),  # Yellow	
    (0, 255, 128)   # Spring Green
]

def overlay_masks_on_image(image_path, masks, colors, alpha=0.99):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a blank overlay
    overlay = np.zeros_like(image)
    
    # Process each mask
    for mask_id, mask in masks.items():
        # Convert mask to binary numpy array if it's not already
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if mask.dtype != bool:
            mask = mask > 0.5
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # Resize the mask to match the image dimensions
        mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Get the color for this mask
        color = colors[mask_id]
        
        # Create a colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask_resized == 1] = color
        
        # Add the colored mask to the overlay
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
    
    # Overlay the masks on the image
    overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
    
    return overlaid_image

def create_overlay_video(output_video_path, video_dir, frame_names, video_segments, reverse=False, fps=10, alpha=0.99):
    """
    Create an overlay video with mask segmentations.
    
    Args:
        output_video_path (str): Path where the output video will be saved
        video_dir (str): Directory containing the frame images
        frame_names (list): List of frame filenames
        video_segments (dict): Dictionary containing mask segments
        reverse (bool): If True, create video in reverse order
        fps (int): Frames per second for output video
        alpha (float): Transparency of the mask overlay (0-1)
    """
    # Prepare the video writer
    frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    if frame is None:
        raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Assign colors to each unique mask ID
    all_mask_ids = set()
    for masks in video_segments.values():
        all_mask_ids.update(masks.keys())
    colors = {mask_id: COLORS[i % len(COLORS)] for i, mask_id in enumerate(all_mask_ids)}

    # Create frame index list (normal or reversed)
    frame_indices = range(len(frame_names))
    if reverse:
        frame_indices = reversed(frame_indices)

    # Process each frame
    for frame_idx in frame_indices:
        print(f"Processing frame {frame_idx}")
        image_path = os.path.join(video_dir, frame_names[frame_idx])
        
        try:
            if frame_idx in video_segments:
                masks = video_segments[frame_idx]
                overlaid_frame = overlay_masks_on_image(image_path, masks, colors, alpha)
            else:
                # If no segmentation for this frame, use the original image
                overlaid_frame = cv2.imread(image_path)
                if overlaid_frame is None:
                    raise ValueError(f"Could not read image from {image_path}")
                overlaid_frame = cv2.cvtColor(overlaid_frame, cv2.COLOR_BGR2RGB)
            
            # Write the frame to the video
            out.write(cv2.cvtColor(overlaid_frame, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            # If there's an error, write the original frame
            original_frame = cv2.imread(image_path)
            if original_frame is not None:
                out.write(original_frame)
            else:
                print(f"Could not read original frame {frame_idx}")

    # Release the video writer
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage:
# Forward video
create_overlay_video("free1.mp4", video_dir, frame_names, video_segments, reverse=False)

# Reverse video
create_overlay_video("free2_reverse.mp4", video_dir, frame_names, video_segments, reverse=True)



