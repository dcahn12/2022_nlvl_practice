import torch
from utils.rnns import (mean_pooling, max_pooling)


class MeanPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()


    def forward(self, batch_hidden_states, lengths, **kwargs):
        return mean_pooling(batch_hidden_states, lengths)


class MaxPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, batch_hidden_states, lengths, **kwargs):
        return max_pooling(batch_hidden_states, lengths)