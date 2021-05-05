import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.misc import NestedTensor
from models.resnet_vd import *
from models.transformer import Transformer

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

class DETR(nn.Layer):
    '''
        This is the DETR module that performs object detection
    '''
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_querys,
                 aux_loss=False):
        super().__init__()
        self.num_querys  = num_querys
        self.transformer = transformer
        hidden_dim       = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)                           #output the classes possibility
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, 3)                                #output the bounding box paramaters(each position output 3 bounding box)
        self.query_embed = nn.Embedding(num_querys, hidden_dim)                             #position embedding
        self.input_proj  = nn.Conv2D(backbone.num_channels, hidden_dim, kernel_size=1)      #input process.
        self.backbone    = backbone
        self.aux_loss    = aux_loss
        

    def forward(self, samples: NestedTensor):
        '''
            Input:
                samples.tensor: batch_size x 3 x H x W
                samples.mask:   batch_size x H x W
            Output:
                'pred_logits': the classification logits. shape=[batch_size x num_queries x (num_classes + 1)]
                'pred_boxes' : The normalized boxes coordinates for all queries(center_x, center_y, height, width).
                'aux_outputs': A list of dictionnaries containing the two above keys for each decoder layer. 
        '''
        pass
