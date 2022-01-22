# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 10:11:07 2022

@author: charl
"""

import torch as T
import torch.nn as nn
from utils import InputNormalize
import torchvision.models as models

class VGG_NSTWrapper(nn.Module):
    
    def __init__(self):
        super(VGG_NSTWrapper, self).__init__()
        self.mean = T.tensor([0.485, 0.456, 0.406])
        self.std = T.tensor([0.229, 0.224, 0.225])
        
        self.normalize = InputNormalize(self.mean, self.std)
        self.model = models.vgg19(
            pretrained=True).features # Pretrained Torch VGG model
        
    def forward(self, x):
        layers = {}
        x = self.normalize(x)
        i = 0
        
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(
                    layer.__class__.__name__))
            x = layer(x)
            
            if isinstance(layer, nn.Conv2d):
                layers[name] = x
            if i == 13: # Can be tuned to use more layers
                break
        return layers