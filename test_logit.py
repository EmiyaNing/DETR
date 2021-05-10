import paddle
import numpy as np
import paddle.nn.functional as F
from models.matcher import HungarianMatcher

def _get_src_permutation_idx(indices):
    batch_idx = paddle.concat([paddle.full_like(src, i) for i, (src,_) in enumerate(indices)])
    src_idx   = paddle.concat([src for (src, _) in indices])
    return batch_idx, src_idx


def loss_labels(outputs, targets, indices):
    src_logits = outputs['pred_logits']
    idx        = _get_src_permutation_idx(indices)
    target_classes_o = paddle.concat([paddle.index_select(t['labels'], J) for t, (_,J) in zip(targets, indices)])
    target_classes   = paddle.full(src_logits.shape[:2], 12, dtype='int64')
    '''
        use idx as index, idx[0] as row index, idx[1] as col index, find the element.
        As the target_classes_o's order replace the finded element with target_classes_o's element...
        For example:
            data = [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]
            idx  = ([0, 1, 2], [2, 1, 0])
            target = [11, 12, 13]
            data[idx] = target
            data = [[1, 2, 11],
                    [4, 12, 6],
                    [13, 8, 9]]
    '''
    np_target_class = target_classes.numpy()
    np_target_class_o = target_classes_o.numpy()
    idx0 = idx[0].numpy()
    idx1 = idx[1].numpy()
    np_target_class[(idx0, idx1)] = np_target_class_o
    target_classes       = paddle.to_tensor(np_target_class)
    loss_ce = F.cross_entropy(src_logits, target_classes)
    losses  = {'loss_ce': loss_ce}

    return losses
    

    



output_pred = paddle.to_tensor(np.random.randn(4, 256, 13))
output_bbox = paddle.to_tensor(np.random.randn(4, 256, 4))
label1      = paddle.to_tensor(np.random.randint(0, 12, size=[64]), dtype='int64')
label2      = paddle.to_tensor(np.random.randint(0, 12, size=[64]), dtype='int64')
label3      = paddle.to_tensor(np.random.randint(0, 12, size=[64]), dtype='int64')
label4      = paddle.to_tensor(np.random.randint(0, 12, size=[64]), dtype='int64')
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

matcher     = HungarianMatcher(0.4,0.4,0.4)
indices     = matcher(outputs, targets)
loss        = loss_labels(outputs, targets, indices)
print(loss)