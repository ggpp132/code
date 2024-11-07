import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from Evaluator import Evaluator
import torch
import torch.nn as nn
from utils_image import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


def metrics(save_dir,dataset,a_dir,b_dir):
    metric_result = np.zeros((4))
    i=0
    for img_name in os.listdir(os.path.join(a_dir)):
        ir = image_read_cv2(os.path.join(a_dir, img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(b_dir, img_name), 'GRAY')
        fi = image_read_cv2(os.path.join(save_dir, img_name), 'GRAY')
        metric_result += np.array([Evaluator.MI(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                    , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(save_dir))
    print("\n"*2+"="*45)
    model_name="FNet    "
    print("The test result of "+dataset+' :')
    print("\t\t MI\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))
            )
    print("="*45)