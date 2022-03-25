# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 10:09:03 2022

@author: The Wizard of Python
"""
import torch.nn.functional as functional
import torch.nn as nn
from src.utils import gram_matrix


class StyleLoss(nn.Module):
    """ Compute Style Image Loss """
    def __init__(self, feature_maps):
        super(StyleLoss, self).__init__()
        # We detach the target Style (and the target Content) since these
        # are the tensors used as targets
        self.feature_maps = {key: gram_matrix(val).detach() for key, val 
                             in feature_maps.items()}
        
    def forward(self, input, layer_name):
        G = gram_matrix(input[layer_name])
        return functional.mse_loss(G, self.feature_maps[layer_name])
    
    
class ContentLoss(nn.Module):
    def __init__(self, feature_maps):
        super(ContentLoss, self).__init__()
        self.feature_maps = {key: val.detach() for key, val 
                             in feature_maps.items()}
        
    def forward(self, input, layer_name):
        return functional.mse_loss(input[layer_name],
                                   self.feature_maps[layer_name])