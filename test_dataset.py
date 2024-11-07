from cgi import test
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import time
import sys
import utils_image as util
from test_dataloder import Dataset as D
from torch.utils.data import DataLoader


def test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model):
    print(a_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = D(a_dir, b_dir, in_channelA, in_channelB)
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,drop_last=False, pin_memory=True)
    for i, test_data in enumerate(test_loader):
        imgname = test_data['A_path'][0]
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)
        if in_channelB==3:
            ycbcr=util.tensor2uint(img_b.detach()[0].float().cpu())
            img_b=img_b[:,0,:,:]
            img_b=img_b.unsqueeze(0)
        # inference
        with torch.no_grad():
            output = model(img_a, img_b)
            output = output.detach()[0].float().cpu()
        output = util.tensor2uint(output)
        if in_channelB==3:
            ycbcr[:,:,0] = output
            output = util.ycbcr2rgb(ycbcr)
        save_name = os.path.join(save_dir, os.path.basename(imgname))
        util.imsave(output, save_name)
