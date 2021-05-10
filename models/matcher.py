import paddle
from scipy.optimize import linear_sum_assignment
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Layer):
    '''
        This class computes an assignment between the targets and the predictions of the network.
    '''
    def __init__(self, cost_class: float=1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = paddle.flatten(outputs["pred_logits"], 0, 1)
        out_prob = F.softmax(out_prob)
        out_bbox = paddle.flatten(outputs["pred_boxes"], 0, 1)

        tgt_ids  = paddle.concat([v["labels"] for v in targets])
        tgt_bbox = paddle.concat([v["boxes"] for v in targets])
        '''
            tgt_ids's dim is [batch_size * num_target_boxes]
            tgt_bbox's dim is[batch_size * num_target_boxes, 4]
            original code is :
                cost_class = -out_prob[:, tgt_ids]
            It equal:
                cost_class = -paddle.index_select(out_prob, tgt_ids, axis=1)
        '''
        cost_class = -paddle.index_select(out_prob,tgt_ids, axis=1)
        temp_bbox  = out_bbox.numpy()
        temp_tgt   = tgt_bbox.numpy()
        cost_bbox  = paddle.to_tensor(cdist(temp_bbox, temp_tgt, p=1))

        #cost_giou  = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        #C          = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C          = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        C          = paddle.reshape(C, shape=[bs, num_queries, C.shape[-1]])
        size       = [len(v['boxes']) for v in targets]
        indices   = []
        indices    = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(paddle.split(C, size, axis=-1))] 
        return [(paddle.to_tensor(i, dtype='int64'), paddle.to_tensor(j, dtype='int64')) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)      


