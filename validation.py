# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:26:12 2022

@author: Administrator
"""


#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import os
import numpy as np
from keras.optimizers import Adam

from unet import Unet
from nets.unet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss)

from utils.dataloader_medical import UnetDataset
from utils.utils_metrics import Iou_score, f_score




if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    unet = Unet().model
    #model = unet.model
  
    #----------------------------------------------------------------------------------------------------------#
    #print(unet.metrics_names)
    
    unet.summary()
    
    VOCdevkit_path = 'Particle_Datasets'
    num_classes =2 
    batch_size = 2
  
    input_shape     = [512, 512]
    cls_weights     = np.ones([num_classes], np.float32)
    loss = CE(cls_weights)
    unet.compile(loss = loss,
                optimizer = Adam(lr=1e-5),
                metrics = ['accuracy', f_score(),Iou_score()])
    
    print(unet.metrics_names)
    
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        test_lines = f.readlines()
        
    steps = len(test_lines)//batch_size
    
    test_dataloader   = UnetDataset(test_lines, input_shape, batch_size, num_classes, False, VOCdevkit_path)
    
    loss,acc, f_score,Iou_score = unet.evaluate_generator(test_dataloader, steps = steps, max_queue_size=10, workers=1, use_multiprocessing=False)
    print(loss,acc, f_score,Iou_score)

      
                

