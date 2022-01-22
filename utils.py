# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 09:14:40 2022

@author: charl
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as functional
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms


def image_loader(image_name, imsize=480):
    """ Image loader to Torch tensor """
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)[:, :3, :, :]
    return image


def imshow(tensor, title=None):
    """ Display Image using Matplotlib """
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.axis("off")
    plt.ioff()
    plt.show()
    
    plt.pause(0.001) # pause a bit so that plots are updated

def save_result(image, model, content_img, style_img, pathdir='results/'):
    if not os.path.exists(pathdir):
        os.makedirs(pathdir)
    plt.imshow(image)
    plt.axis("off")
    plt.ioff()
    plt.savefig(pathdir+model+'_'+
                content_img[:-4]+'+'+style_img)

def gram_matrix(input):
    """ Compute Gram matrix of input Torch tensor image """
    # batch_size, feature_map, height, width
    a, b, c, d = input.size()
    features = input.view(a*b, c*d)
    
    # Gram product
    G = T.mm(features, features.t())
    # Normalization of the gram matrix by dividing by
    # the number of elements in each faeture map
    return G.div(a*b*c*d)


class InputNormalize(nn.Module):
    """
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    Original source: 
    https://github.com/MadryLab/robustness_lib/blob/master/robustness/helpers.py
    """
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer('new_mean', new_mean)
        self.register_buffer('new_std', new_std)

    def forward(self, x):
        x = T.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized
