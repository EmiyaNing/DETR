import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Optional, List
from paddle import Tensor


class NestedTensor(object):
    '''
        So this class compose by a tensor list and a mask(tensor)
        It's seems redundance...
    '''
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask    = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _max_by_axis(the_list):
    '''
        type:(List[List[int]]) - > List[int]
        save each sublist's max value to the maxes list...
    '''
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index,item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        # finde the max size image
        shape_list  = []
        for img in tensor_list:
            shape_list.append(list(img.shape))
        max_size    = _max_by_axis(shape_list)
        # add a deminsion to the max_size
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w  = batch_shape
        dtype       = tensor_list[0].dtype
        # create a zeros tensor to save image, a one tensor to save mask(padding information).
        tensor      = paddle.zeros(batch_shape, dtype=dtype)
        mask        = paddle.ones((b,h,w),dtype='int32')
        '''for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img
            m[:img.shape[1], :img.shape[2]] = 0
            print(m)'''
            #m[:img.shape[1], :img.shape[2]] = False
        for index, img in enumerate(tensor_list):
            tensor[index, :img.shape[0], :img.shape[1], :img.shape[2]] = img
            mask[index, :img.shape[1], :img.shape[2]] = 0
    else:
        return ValueError('not supported')
    return NestedTensor(tensor, mask)