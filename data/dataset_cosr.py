import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import os
import cv2


def get_npy_paths(root_path):
    npy_paths = []
    selected_files = os.listdir(root_path)
    for file_name in selected_files:
        file_path = os.path.join(root_path, file_name)
        if os.path.isfile(file_path):
            npy_paths.append(file_path)
    return npy_paths

# convert uint to 3-dimensional torch tensor and uniformize to [0, 1]
def uint2tensor3(img):
    # return shape: C x H x W
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    max_val = np.max(img)
    if max_val > 1e-2:
        img = img / max_val  # scale to [0, 1]
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)

class DatasetPlain(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetPlain, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.upscale = opt['scale'] 
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        # get the path of H/L
        self.paths_H = get_npy_paths(opt['dataroot_H'])
        self.paths_L = get_npy_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        # get H/L feature map
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        img_H = np.load(H_path)
        img_L = np.load(L_path)
        img_H = np.nan_to_num(img_H)
        img_L = np.nan_to_num(img_L)
        img_H = img_H.astype(np.float32)
        img_L = img_L.astype(np.float32)
        scale = 1.0

        # if train, get L/H patch pairs
        if self.opt['phase'] == 'train':
            patch_H = img_H
            patch_L = img_L

            # Yunling: Data Scaling
            # Dim of 0-12: Log Scaling
            # Dim of 13-62: PCA, keep
            for n in range(patch_H.shape[-1]):
                slice_H = patch_H[:,:,n]
                slice_L = patch_L[:,:,n]
                if n < 12:
                    slice_H = np.log(slice_H + 1)
                    slice_L = np.log(slice_L + 1)
                    patch_H[:,:,n] = slice_H
                    patch_L[:,:,n] = slice_L

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            img_L, img_H = uint2tensor3(patch_L), uint2tensor3(patch_H)

        else:
            img_L, img_H = uint2tensor3(img_L), uint2tensor3(img_H)

        if 'postdam' in H_path:
            resolution = 0.05
        elif 'JAX' in H_path or 'OMA' in H_path:
            resolution = 0.35
        else:
            resolution = 0.3
        resolution = 1.0
        
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path,'scale': scale, 'resolution': resolution }

    def __len__(self):
        return len(self.paths_H)
