"""
Utilities for bounding box manipulation and GIoU.
"""
import paddle


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = paddle.unbind(x, axis=-1)
    b              = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                      (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)

def box_xyxy_cxcyhw(x):
    x0, y0, x1, y1 = paddle.unbind(x, axis=-1)
    b              = [(x0 + x1)/2, (y0 + y1)/2,
                      (x1 - x0),   (y1 - y0)]
    return paddle.stack(b, axis=-1)

def box_area(boxes):
    '''
        Input tensor the last deminsion must be x1, y1, x2, y2
        return area(Tensor[N]): area for each box
    '''
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 4])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    boxes1 = paddle.unsqueeze(boxes1, axis=1)
    lt = paddle.max(boxes1[:, :, :2], boxes[:, :2])
    rb = paddle.min(boxes1[:, :, 2:], boxes2[:, 2:])

    temp_wh = rb - lt
    wh = paddle.nn.functional.relu(temp_wh)
    inter   = wh[:, :, 0] * wh[:, :, 1]

    union = paddle.unsqueeze(area1, axis=1) + area2 - inter
    iou   = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    assert paddle.all(boxes1[:, 2:] >= boxes1[:, :2])
    assert paddle.all(boxes2[:, 2:] >= boxes2[:, :2])

    iou, union = box_iou(boxes1, boxes2)

    boxes1 = paddle.unsqueeze(boxes, axis=1)
    lt = paddle.min(boxes1[:, :, :2], boxes2[:, :2])
    rb = paddle.max(boxes1[:, :, 2:], boxes2[:, 2:])

    temp_wh = rb - lt
    wh      = paddle.nn.functional.relu(temp_wh)
    area    = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union)/ area
