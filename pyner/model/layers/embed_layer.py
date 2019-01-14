#encoding:utf-8
import numpy as np
import torch
import torch.nn as nn
from .spatial_dropout import Spatial_Dropout
'''
嵌入层
'''
class Embed_Layer(nn.Module):
    def __init__(self,
                 embedding_weight = None,
                 vocab_size = None,
                 embedding_dim = None,
                 training = False,
                 dropout_emb = 0.25):
        super(Embed_Layer,self).__init__()
        self.training = training
        self.encoder = nn.Embedding(vocab_size,embedding_dim)
        self.dropout = nn.Dropout(dropout_emb)
        # self.dropout = Spatial_Dropout(dropout_emb)

        if not self.training:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if embedding_weight is not None:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_weight))
        else:
            self.encoder.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, embedding_dim)))

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, inputs):
        x = self.encoder(inputs)
        # batch_size * seq_len * embed_dim
        x = self.dropout(x)
        return x




