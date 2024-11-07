import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils_image as util


class Dataset(data.Dataset): 
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, root_A, root_B, in_channelsA, in_channelsB):
        super(Dataset, self).__init__()
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = util.get_image_paths(root_A)
        self.paths_B = util.get_image_paths(root_B)
        self.inchannelsA = in_channelsA
        self.inchannelsB = in_channelsB

    def __getitem__(self, index):

        # ------------------------------------
        # get under-exposure image, over-exposure image
        # and norm-exposure image
        # ------------------------------------
        # print('input channels:', self.n_channels)
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        img_A = util.imread_uint(A_path, self.inchannelsA)
        img_B = util.imread_uint(B_path, self.inchannelsB)
        
        if self.inchannelsB==3:
            img_B = util.rgb2ycbcr(img_B,only_y=False)
       
        """
        # --------------------------------
        # get testing image pairs
        # --------------------------------
        """
        img_A = util.uint2single(img_A)
        img_B = util.uint2single(img_B)
        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_A = util.single2tensor3(img_A)
        img_B = util.single2tensor3(img_B)

        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)
