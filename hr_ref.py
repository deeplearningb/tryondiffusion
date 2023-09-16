from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageDraw
import os.path as osp
import numpy as np
import json


class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height = opt['load_height']
        self.load_width = opt['load_width']
        self.semantic_nc = opt['semantic_nc']
        self.data_path = osp.join(opt['dataset_dir'], opt['dataset_mode'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        img_names = []
        c_names = []
        with open(osp.join(opt['dataset_dir'], opt['dataset_dir']), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)
        self.img_names = img_names
        self.c_names = dict()
        self.c_names['unpaired'] = c_names

    def load_keypoints(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get pose keypoints for the first person detected in the image
        pose_keypoints = np.array(data['people'][0]['pose_keypoints']).reshape(-1, 3)
        return pose_keypoints
    
    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    # Adding the method to the VITONDataset class
    # VITONDataset.get_parse_agnostic = get_parse_agnostic

    # Continuing the VITONDataset class definition

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        
        # apply head and lower body masks
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    # Adding the method to the VITONDataset class
    # VITONDataset.get_img_agnostic = get_img_agnostic

    # Continuing the VITONDataset class definition

    def __len__(self):
        return len(self.img_names)

    # Adding the method to the VITONDataset class
    # VITONDataset.__len__ = VITONDataset___len__

    # Continuing the VITONDataset class definition

    def __getitem__(self, index):
        # Load two person images and their corresponding data
        img_name1 = self.img_names[index]
        img_name2 = self.img_names[(index + 1) % len(self.img_names)]  # Getting the next image in the dataset
        
        # Load and preprocess the first image and its data
        img_path1 = osp.join(self.data_path, 'image', img_name1)
        img1 = Image.open(img_path1).convert('RGB')
        
        # Load and preprocess the second image and its data
        img_path2 = osp.join(self.data_path, 'image', img_name2)
        img2 = Image.open(img_path2).convert('RGB')
        
        # Load parse maps and pose keypoints for both images
        # ...
        
        # Create an agnostic version of the first image where the upper clothing is removed
        # Note: You would need to load and use the parse map of the first image to create the agnostic image
        # For now, I'm using a placeholder parse map and pose data
        parse1_path = osp.join(self.data_path, 'path_to_parse_map', img_name1)
        parse1 = Image.open(parse1_path)  # Load the actual parse map
        pose_data1_path = osp.join(self.data_path, 'path_to_pose_data', img_name1.replace('.jpg', '_keypoints.json'))
        pose_data1 = self.load_keypoints(pose_data1_path)  # Load the actual pose data
        
        agnostic_img1 = self.get_parse_agnostic(parse1, pose_data1)
        
        # Apply the transformations to the images
        parse1_path = osp.join(self.data_path, 'parse_maps', img_name1 + '.png')
        parse2_path = osp.join(self.data_path, 'parse_maps', img_name2 + '.png')
        pose_data1_path = osp.join(self.data_path, 'pose_data', img_name1 + '_keypoints.json')
        pose_data2_path = osp.join(self.data_path, 'pose_data', img_name2 + '_keypoints.json')
        
        # Load the parse maps and pose data for the second image
        parse2 = Image.open(parse2_path)  # Load the actual parse map
        pose_data2 = self.load_keypoints(pose_data2_path)  # Load the actual pose data
        
        # Create an "agnostic" version of the second image
        agnostic_img2 = self.get_parse_agnostic(parse2, pose_data2)
        
        # Return the data
        return img1, img2, agnostic_img1, agnostic_img2


opt = {
"load_height": 256,
"load_width": 192,
"semantic_nc": 20,
"dataset_dir": "./VITON_traindata/train_img/",
"dataset_mode": "train",  # or "test"
"dataset_list": "path/to/dataset_list.txt"
}

dataset = VITONDataset(opt)
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True)

# for batch in data_loader:
#     img, c_name = batch
#     # Do something with the data...