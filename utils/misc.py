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