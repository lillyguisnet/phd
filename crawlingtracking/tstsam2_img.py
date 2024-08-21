
import sys
sys.path.append("/home/maxime/prg/phd/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2 = build_sam2(model_cfg, checkpoint, device ='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

img = '/home/maxime/prg/phd/IMG_20200320_114007.jpg'
img2 = '/home/maxime/prg/phd/img2.png'
img3 = '/home/maxime/prg/phd/dev/data/d_p2_01a_d6.png'
img4 = '/home/maxime/prg/phd/dev/data/c_p4_01a_d5.png'
img5 = '/home/maxime/prg/phd/dev/data/b_p3_03_d3.png'
img6 = '/home/maxime/prg/phd/dev/data/b_p2_01_d2.png'
img7 = '/home/maxime/prg/phd/dev/data/a_p4_03a_d6.png'
img8 = '/home/maxime/prg/phd/dev/data/a_p3_03_d1.png'
img9 = '/home/maxime/prg/phd/dev/data/a_p2_03a_d4.png'
img10 = '/home/maxime/prg/phd/dev/data/a_p2_03a_d6.png'
img11 = '/home/maxime/prg/phd/dev/data/a_p1_03_d2.png'
img12 = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/a-02252022132222-0000/000000.png"

""" image = Image.open(img3)
image = np.array(image.convert("RGB")) """

image = cv2.imread(img12)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig("tstsam12.png")
plt.close()
len(masks)



mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.6,
    stability_score_offset=0.6,
    crop_n_layers=1,
    box_nms_thresh=0.6,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    #use_m2m=True,
)

image = cv2.imread(img12)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks2 = mask_generator_2.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.savefig("tst2sam12.png")
plt.close()
len(masks2)

cv2.imwrite("mask.png", image*np.expand_dims(masks2[9]['segmentation'], axis=-1))
np.save('tstvideo_1worm.npy', masks2[9]['segmentation'])



def analyze_image(img_path):
    with Image.open(img_path) as img:
        print(f"Image: {img_path}")
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
        print(f"Size: {img.size}")
        
        # Convert to RGB only if it's not already in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        image_array = np.array(img)
        print(f"Array Shape: {image_array.shape}")
        print(f"Data type: {image_array.dtype}")
        print(f"Min value: {image_array.min()}")
        print(f"Max value: {image_array.max()}")
        print(f"Mean value: {image_array.mean()}")
        print("---")



analyze_image(img)
analyze_image(img2)
analyze_image(img3)