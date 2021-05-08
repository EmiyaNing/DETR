import sys
import paddle
import numpy as np
from models.position_encoding import PositionEmbeddingSine,PositionEmbeddingLearned
from utils.misc import *
from utils.box_ops import *
from models.detr import MLP
from models.resnet_vd import *
from models.matcher import *



def test_PositionEmbeddingLearned():
    data = np.random.randn(1, 3, 4, 4).astype('float32')
    mask = [[[True, True, True, True], [True, False, False, True], [True, False, False, True], [True, True, True, True]]]
    data = paddle.to_tensor(data)
    mask = paddle.to_tensor(mask)
    nest = NestedTensor(data, mask)
    position = PositionEmbeddingLearned(256)
    res  = position(nest)
    print(res.shape)

def test_PositionEmbeddingSine():
    data = np.random.randn(1, 3, 4, 4).astype('float32')
    mask = [[[True, True, True, True], [True, False, False, True], [True, False, False, True], [True, True, True, True]]]
    data = paddle.to_tensor(data)
    mask = paddle.to_tensor(mask)
    nest = NestedTensor(data, mask)
    position = PositionEmbeddingSine(256)
    res  = position(nest)
    print(res.shape)

def test_MLP():
    data = np.random.randn(2, 32, 32)
    data = paddle.to_tensor(data)
    mlp  = MLP(32, 32, 32, 4)
    res  = mlp(data)
    print(res.shape)

def test_nested_tensor_from_tensor_list():
    data1 = np.random.randn(3, 32, 32)
    data2 = np.random.randn(3, 48, 48)
    data3 = np.random.randn(3, 64, 64)
    list  = [data1, data2, data3]
    result = nested_tensor_from_tensor_list(list)
    print(result)

def test_backbone():
    resnet = ResNet18_vd()
    data1  = np.random.randn(3,64,64)
    data2  = np.random.randn(3,64,64)
    data   = [paddle.to_tensor(data1), paddle.to_tensor(data2)]
    inputs   = nested_tensor_from_tensor_list(data)
    result,pos = resnet(inputs)
    print(result[-1].mask.shape)  
    print(result[-1].tensors.shape)
    #print(pos)

def test_box_iou():
    boxes1 = paddle.to_tensor(np.random.randn(8, 4))
    boxes2 = paddle.to_tensor(np.random.randn(8, 4))
    res, union = box_iou(boxes1, boxes2)
    print(res)
    print(union)

def test_generalize_box_iou():
    boxes1 = paddle.to_tensor(np.random.randn(8, 4))
    boxes2 = paddle.to_tensor(np.random.randn(8, 4))
    result = generalized_box_iou(boxes1, boxes2)
    print(result)

def test_matcher():
    output_pred = paddle.to_tensor(np.random.randn(4, 256, 13))
    output_bbox = paddle.to_tensor(np.random.randn(4, 256, 4))
    label1      = paddle.to_tensor(np.random.randint(0, 13, size=[64]), dtype='int64')
    label2      = paddle.to_tensor(np.random.randint(0, 13, size=[64]), dtype='int64')
    label3      = paddle.to_tensor(np.random.randint(0, 13, size=[64]), dtype='int64')
    label4      = paddle.to_tensor(np.random.randint(0, 13, size=[64]), dtype='int64')
    '''
        Because the boxes's value is random vale, so the data may be illegal....
        So the cost_giou's compute may have some inf....
    '''
    boxes1      = paddle.to_tensor(np.random.randn(64, 4))
    boxes2      = paddle.to_tensor(np.random.randn(64, 4))
    boxes3      = paddle.to_tensor(np.random.randn(64, 4))
    boxes4      = paddle.to_tensor(np.random.randn(64, 4))

    outputs     = {}
    outputs['pred_logits'] = output_pred
    outputs['pred_boxes']  = output_bbox
    targets     = [{'labels':label1, 'boxes':boxes1}, 
                   {'labels':label2, 'boxes':boxes2},
                   {'labels':label3, 'boxes':boxes3},
                   {'labels':label4, 'boxes':boxes4}]

    matcher     = HungarianMatcher(0.4, 0.3, 0.2)
    result      = matcher(outputs, targets)
    print(result)

if __name__ == '__main__':
    #test_PositionEmbeddingLearned()
    #test_PositionEmbeddingSine()
    #test_MLP()
    #test_nested_tensor_from_tensor_list()
    #test_backbone()
    #test_box_iou()
    #test_generalize_box_iou()
    test_matcher()


