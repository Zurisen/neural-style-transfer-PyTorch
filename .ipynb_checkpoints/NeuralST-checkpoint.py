# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:27:31 2022

@author: The Wizard of Python
"""

from PIL import Image
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as functional
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from utils import image_loader, imshow
from losses import *


def NST(st_model, img_dims, content_img, style_img, start_from_content=True,
        verbose=False, n_iters = 300, style_weight=4e6, content_weight=1,
        content_layers=['conv_10'], 
        style_layers=['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']):
    
    # CUDA device setup if available
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    st_model.eval().to(device)

    # Load images to Torch tensors
    content_img = image_loader(content_img, img_dims).to(device)
    style_img = image_loader(style_img, img_dims).to(device)
    
    # Construct Style and Content loss classes
    content_feature_maps = st_model(content_img)
    content_feature_maps = {key: val.detach() for key, val
                            in content_feature_maps.items()}
    
    style_feature_maps = st_model(style_img)
    style_feature_maps = {key: val.detach() for key, val
                          in style_feature_maps.items()}
    
    content_loss_func = ContentLoss(content_feature_maps)
    style_loss_func = StyleLoss(style_feature_maps)
    
    # Start input image as the content image or not (white noise)
    if start_from_content:
        input_img = content_img.clone().to(device)
    else:
        input_img = T.randn(content_img.data.size(), device=device)
    
    # Use LBFGS optimizer
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    # Store images per 100 iters
    i = [0]
    images = []
    unloader = transforms.ToPILImage()
    while i[0] <= n_iters:
        def closure():
            # Clamping correction
            input_img.data.clamp_(0, 1)
            
            # Reset gradients
            optimizer.zero_grad()
            input_feature_maps = st_model(input_img)
            
            # Feed forward the input feature map through every layer
            # of the network and calculate content and style losses
            style_loss = 0
            content_loss = 0
            for sl in style_layers:
                style_loss += style_loss_func(input_feature_maps,
                                              sl)*style_weight
            for cl in content_layers:
                content_loss += content_loss_func(input_feature_maps,
                                                  cl)*content_weight
            loss = style_loss + content_loss
            
            # Backpropagate
            loss.backward()
            
            # Debugging stats
            print('{}   '.format(i), 'Style Loss: %0.4f' % style_loss,
                  'Content Loss: %0.4f' % content_loss)
            if i[0] % 100 == 0:
                if verbose:
                    plt.figure(figsize=(8, 8))
                    imshow(input_img)
                image = input_img.clone().cpu()
                image = image.squeeze(0)
                image = unloader(image)
                images.append(image)
            
            i[0] += 1
            
            return style_loss + content_loss
        # Perform LBFGS step
        optimizer.step(closure)
        
    # Last correction
    input_img.data.clamp_(0, 1)

    return images
            