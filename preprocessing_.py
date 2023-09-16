import numpy as np
from PIL import Image
from matplotlib.path import Path as Polygon
from torchvision import transforms
import torch
import json
import cv2

def create_segmentation_masks(segmentation_data, category_names, img_size, mask_upper_body=None):
    mask_other_parts = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    
    for region_key, region_name in category_names.items():
        segmentation_points = np.array(segmentation_data[region_key]['segmentation'][0])
        poly = Polygon(segmentation_points, closed=True)
        
        y, x = np.mgrid[:img_size[0], :img_size[1]]
        points = np.vstack((x.ravel(), y.ravel())).T
        
        if region_name in {'inner_torso', 'inner_rsleeve', 'inner_lsleeve'}:
            mask_upper_body[poly.contains_points(points).reshape(img_size[0], img_size[1])] = 1
        else:
            mask_other_parts[poly.contains_points(points).reshape(img_size[0], img_size[1])] = 1

    if mask_upper_body is None:
        mask_upper_body = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)

    return mask_upper_body, mask_other_parts

def get_pose_keypoints(json_data):
    keypoints = []
    for region in json_data.values():
        if region.get('keypoints'):
            keypoints.extend(region['keypoints'])
    return keypoints

# Load the JSON data for the images

with open('./train/model_pose_data/1008_A001_000.json') as json_file:
    data1 = json.load(json_file)
category_names1 = {k: v['category_name'] for k, v in data1.items() if k.startswith('region')}
img_size1 = (data1['image_size']['height'], data1['image_size']['width'])

with open('./train/model_pose_data/1008_A001_044.json') as json_file2:
    data2 = json.load(json_file2)
category_names2 = {k: v['category_name'] for k, v in data2.items() if k.startswith('region')}
img_size2 = (data2['image_size']['height'], data2['image_size']['width'])

# Load the original images as numpy arrays
img1_np = np.array(Image.open('./train/model/1008_A001_000.jpg'))
img2_np = np.array(Image.open('./train/model/1008_A001_044.jpg'))
img2_np_resized = cv2.resize(img2_np, (img1_np.shape[1], img1_np.shape[0]))

# Create segmentation masks
mask_upper_body1, mask_other_parts1 = create_segmentation_masks(data1, category_names1, img_size1)
mask_upper_body2, mask_other_parts2 = create_segmentation_masks(data2, category_names2, img_size2)

# Apply segmentation masks to get the required images
Ia = img1_np * mask_other_parts1[:, :, None]
# Extend mask_upper_body2 to 3 channels
Ic = img2_np_resized * np.repeat(mask_upper_body2[:, :, None], 3, axis=2)


# Get the pose keypoints and normalize them
Jp = get_pose_keypoints(data1)
Jg = get_pose_keypoints(data2)
Jp = [kp / img_size1[1] for kp in Jp]
Jg = [kp / img_size2[1] for kp in Jg]

# Define the transformations for images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of your model
    transforms.ToTensor(),
])

# Convert numpy images to PIL images
Ia_pil = Image.fromarray(Ia.astype('uint8'))
Ic_pil = Image.fromarray(Ic.astype('uint8'))

# Apply the transformations to the images
Ia = preprocess(Ia_pil)
Ic = preprocess(Ic_pil)

# Create the ctryon input
ctryon = {
    'Ia': Ia,
    'Ic': Ic,
    'Jp': torch.tensor(Jp),
    'Jg': torch.tensor(Jg),
}

# Now ctryon is ready to be used as input to your parallelUNet model
