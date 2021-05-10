import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.misc import *
from utils.box_ops import *
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
        self.layers = nn.LayerList(nn.Linear(n,k) for n,k in zip([input_dim] + h, h + [output_dim]))
            

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
        self.input_proj  = nn.Conv2D(backbone.feat_channels[-1], hidden_dim, kernel_size=1)      #input process.
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
        if isinstance(samples, (list, paddle.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        output_class = self.class_embed(hs)
        output_coord = F.sigmoid(self.bbox_embed(hs))

        out          = {'pred_logits': output_class[-1], 'pred_boxes': output_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_coord)
        return out

    def _set_aux_loss(self, output_class, output_coord):
        return [{
            'pred_logits':a, 'pred_boxes':b
        } for a,b in zip(output_class[:-1], output_coord[:-1])]

class SetCriterion(nn.Layer):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher
        self.weight_dict = weight_dict
        self.eos_coef    = eos_coef
        self.losses      = losses
        empty_weight     = paddle.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        '''
            from indices extract the predict result position....
        '''
        batch_idx = paddle.concat([paddle.full_like(src, i) for i, (src,_) in enumerate(indices)])
        src_idx   = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        '''
            from indices extract the target result position....
        '''
        batch_idx = paddle.concat([paddle.full_like(tgt, i) for i, (_,tgt) in enumerate(indices)])
        tgt_idx   = paddle.concat([tgt for (_,tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx        = self._get_src_permutation_idx(indices)

        target_classes_o = paddle.concat([paddle.index_select(t['labels'], J) for t, (_,j) in zip(targets, indices)])
        target_classes   = paddle.full(src_logits.shape[:2], self.num_classes-1 , dtype='int64')
        np_target_class = target_classes.numpy()
        np_target_class_o = target_classes_o.numpy()
        idx0 = idx[0].numpy()
        idx1 = idx[1].numpy()
        np_target_class[(idx0, idx1)] = np_target_class_o
        target_classes       = paddle.to_tensor(np_target_class)
        loss_ce = F.cross_entropy(src_logits, target_classes)
        losses  = {'loss_ce': loss_ce}
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        tgt_length  = paddle.to_tensor([len(v['labels']) for v in targets])
        card_pred   = paddle.sum((paddle.argmax(pred_logits, axis=-1) != pred_logits.shape[-1] - 1), axis=1)
        card_err    = F.l1_loss(paddle.cast(card_pred, dtype='float32'), paddle.cast(tgt_length, dtype='float32'))
        losses      = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        idx0 = idx[0].numpy()
        idx1 = idx[1].numpy()
        temp_pred = outputs['pred_boxes'].numpy()
        src_boxes = paddle.to_tensor(temp_pred[(idx0, idx1)])
        target_boxes = paddle.concat([paddle.select(t['boxes'], i) for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox    = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = paddle.sum(loss_bbox) / num_boxes

        loss_giou = 1 - paddle.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = paddle.sum(loss_giou) / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels' : self.loss_labels,
            'cardinality' : self.loss_cardinality
            'boxes' : self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices             = self.matcher(outputs_without_aux, targets)

        num_boxes           = sum(len(t['labels'] for t in targets))
        num_boxes           = paddle.to_tensor([num_boxes], dtype='float32')

        num_boxes           = paddle.nn.clip(num_boxes , min=1).numpy()[0]

        losses              = {}

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs={}
                    if loss == 'labels':
                        kwargs = {'log':False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}' : v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class PostProcess(nn.Layer):
    def forward(self, outputs, target_size)；
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_size)
        assert target_size.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = paddle.max(prob[.., :-1], axis=-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)

        img_h, img_w = target_size.unbind(1)

        scale_fct    = paddle.stack([img_w, img_h, img_w, img_h], axis=1)

        boxes   = boxes * paddle.unsqueeze(scale_fct, axis=1)
        results = [{'score':s, 'labels':l, 'boxes':b} for s,l,b in zip(scores, labels, boxes)]

        return results


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91

    if args.dataset_file == 'coco_panoptic':
        num_classes = 250

    device      = torch.device(args.device)

    backbone    = ResNet50_vd() 

    transformer = build_transformer(args)

    model       = DETR(
        backbone,
        transformer,
        num_classes,
        aux_loss = args.aux_loss
    )

    matcher     = build_matcher(args)

    weight_dict = {'loss_ce':1, 'loss_bbox':args.bbox_loss_coef}

    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}' : v for k, v in weight_dict.items())
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    postprocessors = {'bbox':PostProcess()}

    return model, criterion, postprocessors

    
