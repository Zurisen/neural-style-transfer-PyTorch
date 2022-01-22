# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 10:47:50 2022

@author: charl
"""
import sys
import os
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as functional
import torch.optim as optim
import torch.nn as nn
from utils import image_loader, imshow, save_result
from ConvNets import *
from NeuralST import NST

# model = sys.argv[1]
# img_dims = int(sys.argv[2])
# content_img = sys.argv[3]
# style_img = sys.argv[4]

model = 'VGG'
img_dims = 512
content_img = 'bodybuilder.jpg'
style_img = 'the_scream.jpg'

if __name__ == '__main__':
    
    wrapnets = {'VGG':VGG_NSTWrapper()}
    content_img = os.path.join('content_images', content_img)
    style_img = os.path.join('style_images', style_img)
    
    try:
        st_model = wrapnets[model]
    except KeyError:
        raise KeyError('Style Transfer model not available!')
    
    images = NST(st_model, img_dims, content_img, style_img,
                 start_from_content=True, verbose=True,
                 n_iters=500, style_weight=1e5)
    
    save_result(images[-1], model, content_img, style_img)
    