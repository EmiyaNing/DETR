import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.misc import NestedTensor
from models.resnet_vd import *

class MLP(nn.Layer):
    '''
        Simple multi-layer feed forward network(FFN), also called as Dense Neural Network(DNN).
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList()
        for n, k in zip([input_dim] + h, h + [output_dim]):
            self.layers.sublayers(nn.Linear(n, k))
            

    def forward(self,x):
        '''
            This code seems so coollll!!!!!!!!.
        '''
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
