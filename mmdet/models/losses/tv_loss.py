import torch
import torch.nn as nn
from torch.autograd import Variable
from ..builder import LOSSES

import os
import numpy as np
import pickle

@LOSSES.register_module()
class TVLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(TVLoss,self).__init__()
        self.loss_weight = loss_weight

    def forward(self,x):
        #print('x.size() in tv loss', x.size()) #[3, 416, 416]
        x = x.unsqueeze(0)
        #print('x.size() in tv loss', x.size())
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.loss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

@LOSSES.register_module()
class NPSLoss(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    """

    def __init__(self, printability_file='30values.txt', patch_size=416, loss_weight=1.0):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_size),requires_grad=False)
        self.loss_weight = loss_weight

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        nps_score = self.loss_weight * nps_score
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, patch_size):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((patch_size, patch_size), red))
            printability_imgs.append(np.full((patch_size, patch_size), green))
            printability_imgs.append(np.full((patch_size, patch_size), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa