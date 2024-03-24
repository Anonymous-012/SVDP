import random

import torch.nn as nn
from mmcv.cnn.bricks import PLUGIN_LAYERS
import torch
import numpy as np
import copy
from scipy.ndimage import zoom
import torch.nn.functional as F


class SparsePrompter_uncertainty(nn.Module):
    def __init__(self, shape=[540, 960], sparse_rate = 0.25):
        super(SparsePrompter_uncertainty, self).__init__()
        self.ratio = sparse_rate
        self.shape_h = shape[0]
        self.shape_w = shape[1]
        self.pnum = int(self.shape_h * self.shape_w * sparse_rate)
        self.patch = nn.Parameter(torch.randn([self.pnum, 3, 1, 1]))
        self.if_mask = False
        self.uncmap =  np.random.choice([0, 1], size=(2 * self.shape_h, 2 * self.shape_w), p=[1- sparse_rate, sparse_rate])
        self.mask_pos_lst = []
    
    def update_uncmap(self, uncmap):
        self.uncmap = uncmap
        
    def downsample_map(self, input_shape):
        zoom_factors = [t/o for t, o in zip(input_shape, self.uncmap.shape)]
        resized_map = zoom(self.uncmap, zoom_factors)
        assert resized_map.shape == input_shape
        return resized_map

    def resize_prompt(self, shape):
        resized_prompt = F.interpolate(self.patch.unsqueeze(0), size=shape, mode='bilinear', align_corners=False)
        return  resized_prompt.squeeze(0)

    def select_position(self, shape):
        uncmap_downsampled = self.downsample_map(shape)
        k = self.pnum 
        topk_indices = uncmap_downsampled.flatten().argsort()[-k:]  # get the indices of the topk elements
        topk_coords = [(i // shape[1], i % shape[1]) for i in topk_indices]  # transform to 2D coordinates
        return topk_coords
    
    def get_masked_prompt(self, shape):
        pos = self.select_position(shape)
        new_prompt = torch.zeros((3, shape[0], shape[1]))  # create a new prompt of zeros
        for i, coord in enumerate(pos):
            new_prompt[:, coord[0], coord[1]] = self.patch[i].squeeze()  # place the i-th prompt at the coord position
        return new_prompt

    def update_mask(self):
        self.prompt_lst = []
        scales = [(270, 480), (270, 480), (405, 720), (405, 720), (540, 960), (540, 960), (675, 1200), (675, 1200), \
                  (810, 1440), (810, 1440), (945, 1680), (945, 1680), (1080, 1920), (1080, 1920)]

        for i, new_shape in enumerate(scales):

            masked_prompt = self.get_masked_prompt(new_shape)

            if i % 2 == 1:
                masked_prompt = torch.flip(masked_prompt, [2])
            
            self.prompt_lst.append(masked_prompt)
        
    def forward(self, x, img_metas, position=None):

        if self.if_mask == False:
            return x
        else:
            if position == None:
                position = (0, 540, 0, 960)
            scale_h = x.shape[2]
            scale_w = x.shape[3]
            scale = (scale_h, scale_w)
            
            scales = [(270, 480), (270, 480), (405, 720), (405, 720), (540, 960), (540, 960), (675, 1024), (675, 1024), \
                    (810, 1024), (810, 1024), (945, 1024), (945, 1024), (1024, 1024), (1024, 1024)]

            index = scales.index(scale)
            
            if img_metas[0]['flip']:
                index = index + 1
            
            prompt = self.prompt_lst[index]
            
            prompt_data = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]]).cuda()
            prompt_data[:,] = prompt[:, position[0]:position[1], position[2]:position[3]]
            return x + prompt_data 

