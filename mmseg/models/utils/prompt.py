import torch.nn as nn
from mmcv.cnn.bricks import PLUGIN_LAYERS
import torch


@PLUGIN_LAYERS.register_module()
class FixedPatchPrompter_image(nn.Module):
    def __init__(self, prompt_size, image_size):
        super(FixedPatchPrompter_image, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([x.shape[0], 3, self.isize[1], self.isize[0]]).cuda()


        prompt[:, :, :self.psize, :self.psize] = self.patch
        return x + prompt

# # for feature level
@PLUGIN_LAYERS.register_module()
class FixedPatchPrompter_feature(nn.Module):
    def __init__(self, prompt_size, image_size):
        super(FixedPatchPrompter_feature, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([2, 2048, self.psize, self.psize])) #2 is batchsize, 2048 is feature dimension

    def forward(self, x):
        tmp = torch.zeros_like(x)
        tmp[:,:, :self.psize, :self.psize] = self.patch
        return x + tmp