from cgi import test
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from test_dataset import test
from metrics import metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path='./checkpoints/FNet.pt'
    model = torch.jit.load(model_path)
    model.eval()
    model = model.to(device)
    root_path=r".\test_img/"
    dataset='TNO'                   
    A_dir='ir'
    B_dir='vi'
    in_channelA=1
    in_channelB=1    
    a_dir = os.path.join(root_path, dataset, A_dir)
    b_dir = os.path.join(root_path, dataset, B_dir)
    save_dir = './results/'+dataset
    os.makedirs(save_dir, exist_ok=True)
    test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model)
    metrics(save_dir,dataset,a_dir,b_dir)
    dataset='RoadScene'                   
    a_dir = os.path.join(root_path, dataset, A_dir)
    b_dir = os.path.join(root_path, dataset, B_dir)
    save_dir = './results/'+dataset
    os.makedirs(save_dir, exist_ok=True)
    test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model)
    metrics(save_dir,dataset,a_dir,b_dir)
    dataset='NIR'                   
    A_dir='NIR'
    B_dir='VIS'
    a_dir = os.path.join(root_path, dataset, A_dir)
    b_dir = os.path.join(root_path, dataset, B_dir)
    save_dir = './results/'+dataset
    os.makedirs(save_dir, exist_ok=True)
    test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model)
    metrics(save_dir,dataset,a_dir,b_dir)
    dataset='MRI_CT'                   
    A_dir='MRI'
    B_dir='CT'
    in_channelB=3
    a_dir = os.path.join(root_path, dataset, A_dir)
    b_dir = os.path.join(root_path, dataset, B_dir)
    save_dir = './results/'+dataset
    os.makedirs(save_dir, exist_ok=True)
    test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model)
    metrics(save_dir,dataset,a_dir,b_dir)
    dataset='MRI_PET'                   
    B_dir='PET'
    a_dir = os.path.join(root_path, dataset, A_dir)
    b_dir = os.path.join(root_path, dataset, B_dir)
    save_dir = './results/'+dataset
    os.makedirs(save_dir, exist_ok=True)
    test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model)
    metrics(save_dir,dataset,a_dir,b_dir)
    dataset='MRI_SPECT'                   
    B_dir='SPECT'
    a_dir = os.path.join(root_path, dataset, A_dir)
    b_dir = os.path.join(root_path, dataset, B_dir)
    save_dir = './results/'+dataset
    os.makedirs(save_dir, exist_ok=True)
    test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model)
    metrics(save_dir,dataset,a_dir,b_dir)
    


if __name__ == '__main__':
    main()
