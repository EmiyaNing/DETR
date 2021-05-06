import paddle
import paddle.nn
import paddle.nn.functional as F
import numpy as np

from models.transformer import *
from models.position_encoding import *
from models.resnet_vd import *
from models.detr import *
from utils.misc import *


def test_transformer_encoder_layer():
    layer = Transformer(128, 4, 4, 4, 512)
    src   = np.random.randn(4, 128, 32, 32)
    src   = paddle.to_tensor(src, dtype='float32')
    mask  = np.random.randint(2, size=[4, 32, 32])
    mask  = np.array(mask, dtype=bool)
    mask  = paddle.to_tensor(mask)
    nest  = NestedTensor(src, mask)
    pos_en= PositionEmbeddingSine()
    pos   = pos_en(nest)
    query = nn.Embedding(64, 128)
    res   = layer(src=src, mask=mask, query_embed=query.weight, pos_embed=pos)
    print(res[0].shape)
    print(res[1].shape)

def test_DETR():
    transformer = Transformer(128, 4, 4, 4, 256)
    backbone    = ResNet18_vd()
    detr        = DETR(backbone, transformer, 12, 256)
    data1       = np.random.randn(3, 64, 64)
    data2       = np.random.randn(3, 64, 64)
    data3       = np.random.randn(3, 64, 64)
    data_list   = [data1, data2, data3]
    result      = detr(data_list)
    print(result['pred_logits'].shape)
    print(result['pred_boxes'].shape)

if __name__ == '__main__':
    #test_transformer_encoder_layer()
    test_DETR()
