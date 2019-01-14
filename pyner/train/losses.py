#encoding:utf-8
import torch
import torch.nn.functional as F

# 多分类
class CrossEntropy(object):
    def __init__(self):
        super(CrossEntropy,self).__init__()
        pass
    def __call__(self, output, target):
        loss = F.cross_entropy(input=output, target=target)
        return loss

# 二分类
# define binary cross entropy loss
# note that the model returns logit to take advantage of the log-sum-exp trick
# for numerical stability in the loss
binary_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')