import sys
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
sys.path.append('..')
from utils.misc import NestedTensor
import numpy as np

class PositionEmbeddingSine(nn.Layer):
    '''
        This is a standard version of the position embedding.
        Similar to the Attention is all you need paper.
    '''
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale        

    def forward(self, tensor_list: NestedTensor):
        '''
            In each NestedTensor, mask's batchsize, height, weight should equal to tensor.
            The mask is used to tell the model, which pixel is padding pixel.
            So we define the mask tensor's element must be bool (Force by paddle).
        '''
        x    = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = paddle.logical_not(mask)
        y_embed  = paddle.cumsum(not_mask, axis=1, dtype='float32')
        x_embed  = paddle.cumsum(not_mask, axis=2, dtype='float32')
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(end=self.num_pos_feats, dtype='float32')
        dim_t = self.temperature ** (2 * (paddle.floor(dim_t / 2)) / self.num_pos_feats)

        pos_x = paddle.unsqueeze(x_embed, axis=3) / dim_t
        pos_y = paddle.unsqueeze(y_embed, axis=3) / dim_t
        pos_x = paddle.stack((paddle.sin(pos_x[:, :, :, 0::2]), paddle.cos(pos_x[:, :, :, 0::2])), axis=4)
        pos_y = paddle.stack((paddle.sin(pos_y[:, :, :, 0::2]), paddle.cos(pos_y[:, :, :, 0::2])), axis=4)
        pos_x = paddle.flatten(pos_x,start_axis=3)
        pos_y = paddle.flatten(pos_y,start_axis=3)
        pos   = paddle.concat((pos_y, pos_x), axis=3)
        pos   = paddle.transpose(pos, perm=[0, 3, 1, 2])
        return pos





class PositionEmbeddingLearned(nn.Layer):
    '''
        Those position Embedding class is learnable.

    '''
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(num_embeddings = 100, embedding_dim = num_pos_feats)
        self.col_embed = nn.Embedding(num_embeddings = 100, embedding_dim = num_pos_feats)

    def forward(self, tensor_list: NestedTensor):
        x    = tensor_list.tensors    # the input tensor must be n,c,h,w
        h,w  = x.shape[-2:]           # get each image's h and w
        i    = paddle.arange(end=w)   # get a row tensor, which element is from 0 to w-1
        j    = paddle.arange(end=h)   # get a col tensor, which element is from 0 to h-1
        x_emb= self.col_embed(i)      # embedding the row tensor, demansion w*num_pos_feats
        y_emb= self.row_embed(j)      # embedding the col tensor, demansion h*num_pos_feats
        temp_x = paddle.unsqueeze(x_emb, 0)   # temp_x now should be 1 * w * num_pos_feats
        temp_y = paddle.unsqueeze(y_emb, 1)   # temp_y now should be h * 1 * num_pos_feats
        temp_pos  = paddle.concat([
            paddle.broadcast_to(temp_x, shape=[h, temp_x.shape[1], temp_x.shape[2]]),     # broad to h * w * num_pos_feats
            paddle.broadcast_to(temp_x, shape=[temp_y.shape[0], w, temp_x.shape[2]])      # broad to h * w * num_pos_feats
        ], axis=-1)                                                                       # concat to h * w * (2num_pos_feats)
        temp_pos = paddle.transpose(temp_pos, [2, 0, 1])        # transpose to (2num_pos_feats) * h * w
        temp_pos = paddle.unsqueeze(temp_pos, axis=0)           # insect a demansion to 1 * (2num_pos_feats) * h * w
        pos      = paddle.broadcast_to(temp_pos, [x.shape[0], temp_pos.shape[1], temp_pos.shape[2], temp_pos.shape[3]])
        return pos                                              # Now the pos should be n * (2num_pos_feats) * h * w




if __name__ == '__main__':
    position_embedding = PositionEmbeddingLearned(512)
    data = np.random.randn(4, 3, 64, 64)
    data = paddle.to_tensor(data)
    # there is a test bug, because the mask's element must be boolen type 
    mask_data = np.random.randn(4, 1, 64, 64)
    mask_data = paddle.to_tensor(data)
    nested  = NestedTensor(data, mask_data)
    result = position_embedding(nested)
    print(result.shape)