# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 10:47:50 2022

@author: charl
"""
import sys
import os
import torch as T
import torch.nn.functional as functional
import torch.optim as optim
import torch.nn as nn
from utils import image_loader, imshow
from ConvNets import *
from NeuralST import NST

# model = sys.argv[1]
# img_dims = sys.argv[2]
# content_img = sys.argv[3]
# style_img = sys.argv[4]

model = 'VGG'
img_dims = 512
content_img = 'cmtphoto2.jpg'
style_img = 'shipwrek.jpg'

if __name__ == '__main__':
    wrapnets = {'VGG':VGG_NSTWrapper()}
    
    try:
        st_model = wrapnets[model]
    except KeyError:
        raise KeyError('Style Transfer model not available!')
        
    images = NST(st_model, img_dims, content_img, style_img,
                 start_from_content=True, verbose=True)
    