import os
import h5py
import json
import torch
import numpy as np
import torch.nn.functional as F
import random
from PIL import Image, ImageFilter
from torch.utils import data
import pandas as pd

import torchvision.transforms as transforms

# envs_splits = json.load(open('data/envs_splits.json', 'r'))

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])
# file_folder_name = 'base_data'
file_folder_name = 'data_base_with_rotationz_realdepth'
file_folder_gt_name = 'ground_truth'
# file_folder_indices = 'Indices_realdepth'


normalize = transforms.Compose([
    # transforms.ToPILImage(),
    # # Addblur(p=1, blur="Gaussian"),
    # AddSaltPepperNoise(0.05, 1),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


class s2d3d_DatasetLoader_pano(data.Dataset):
    def __init__(self, cfg, split='train'):
        self.split = split

        if split == 'train':
            self.root = cfg['root'] + '/training'
        elif split == 'val':
            self.root = cfg['root'] + '/valid'
        elif split == 'test':
            self.root = cfg['root'] + '/testing'

        # self.ego_downsample = cfg['ego_downsample']
        self.feature_type = cfg['feature_type']

        self.files = os.listdir(os.path.join(self.root, file_folder_name))  # base_data

        #self.files = np.array([x for x in self.files if '_'.join(x.split('_')[:2]) in envs_splits['{}_envs'.format(split)]])  # using numpy format
        self.files = np.array(self.files)
        self.envs = np.array([x.split('.')[0] for x in self.files])  # using numpy format
        # print('files_files_name:', self.envs)

        # -- load semantic map GT
        # h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_semmap.h5'), 'r')
        # self.semmap_GT = np.array(h5file['semantic_maps'])
        # h5file.close()

        # self.semmap_GT_envs = json.load(open(os.path.join(self.root, 'smnet_training_data_semmap.json'), 'r'))
        # self.semmap_GT_indx = {i: self.semmap_GT_envs.index(self.envs[i] + '.h5') for i in range(len(self.files))}
        self.files_gt = os.listdir(os.path.join(self.root, file_folder_gt_name))

        assert len(self.files) == len(self.files_gt)
        assert len(self.files) > 0

        self.available_idx = np.array(list(range(len(self.files))))

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, index):
        env_index = self.available_idx[index]

        file = self.files[env_index]
        env = self.envs[env_index]
        env_index = env.split('_')[1]
        # print('env_index:', env_index)

        for i in self.files_gt:
            if env_index in i:
                gt_file_name = i
        # print('gt_file_name:', gt_file_name)

        # file = 'camera_ecdcae672fbc4f78a1bce56f3642d71d_hallway_2_frame_equirectangular_domain_rgb.h5'
        # file = 'camera_b73912d7377b41a584725523f3a91883_office_21_frame_equirectangular_domain_rgb.h5'

        h5file = h5py.File(os.path.join(self.root, file_folder_name, file), 'r')

        rgb = np.array(h5file['rgb'])
        # depth = np.array(h5file['depth'])

        rotationz = np.array(h5file['rotation_z'])
        rotationz = rotationz[0]

        camera_location = np.array(h5file['camera_location'])

        h5file.close()

        ################################################################################################################
        h5file = h5py.File(os.path.join(self.root, file_folder_gt_name, gt_file_name), 'r')
        semmap_gt = np.array(h5file['map_semantic'])  # 40 classes 40 -> 20 classes
        h5file.close()

        # print('semmap_gt:', semmap_gt.shape, np.unique(semmap_gt))

        # modified
        # h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_maxHIndices_{}'.format(self.split), file), 'r')
        h5file = h5py.File(os.path.join(self.root, 'Indices_realdepth'.format(self.split), file), 'r')

        proj_indices = np.array(h5file['indices'])
        masks_outliers = np.array(h5file['masks_outliers'])
        h5file.close()

        rgb_no_norm = rgb

        rgb_img = rgb.astype(np.float32)
        rgb_img = rgb_img / 255.0
        # print('rgb_shape_in_dataloader:', rgb_img.shape)

        rgb_img = torch.FloatTensor(rgb_img).permute(2, 0, 1)
        rgb_img = normalize(rgb_img)
        # rgb_img = rgb_img.unsqueeze(0)

        rgb = rgb_img
        # print('rgb:', rgb.size(), rgb)


        proj_indices = torch.from_numpy(proj_indices).long()
        masks_outliers = torch.from_numpy(masks_outliers).bool()
        masks_inliers = ~masks_outliers

        ################ semmap_gt input ###############################################################################
        # semmap_index = self.semmap_GT_indx[env_index]
        # semmap = self.semmap_GT[semmap_index]
        # semmap = torch.from_numpy(semmap).long()

        return (rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt, rotationz)


