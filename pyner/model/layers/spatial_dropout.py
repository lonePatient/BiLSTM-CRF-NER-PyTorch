#encoding:utf-8
import torch.nn as nn
from itertools import repeat
class Spatial_Dropout(nn.Module):
    def __init__(self,drop_prob):

        super(Spatial_Dropout,self).__init__()
        self.drop_prob = drop_prob

    def forward(self,inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self,input):
        return input.new().resize_(input.size(0),*repeat(1, input.dim() - 2),input.size(2))
