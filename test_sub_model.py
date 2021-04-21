import paddle
import numpy as np
from models.position_encoding import PositionEmbeddingSine,PositionEmbeddingLearned
from utils.misc import NestedTensor

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

if __name__ == '__main__':
    #test_PositionEmbeddingLearned()
    test_PositionEmbeddingSine()

