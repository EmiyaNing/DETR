import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.misc import NestedTensor

'''
    The next step, we should fix this file.
    To let the backbone network can process the neastedTensor type data.
    Just use the neastedTensor.Tensor to preprocess asn use the neastedTensor.mask to interpolate....
'''

def SyncBatchNorm(*args, **kwargs):
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)

class Activation(nn.Layer):
    """
    The wrapper of activations.
    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.
    Returns:
        A callable object of Activation.
    Raises:
        KeyError: When parameter `act` is not in the optional range.
    Examples:
        from paddleseg.models.common.activation import Activation
        relu = Activation("relu")
        print(relu)
        # <class 'paddle.nn.layer.activation.ReLU'>
        sigmoid = Activation("sigmoid")
        print(sigmoid)
        # <class 'paddle.nn.layer.activation.Sigmoid'>
        not_exit_one = Activation("not_exit_one")
        # KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink',
        # 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax',
        # 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"
    """

    def __init__(self, act=None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = nn.layer.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval(
                    "nn.layer.activation.{}()".format(act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x

class ConvBNLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            is_vd_mode=False,
            act=None,
    ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if dilation == 1 else 0,
            dilation=dilation,
            groups=groups,
            bias_attr=False)

        self._batch_norm = SyncBatchNorm(out_channels)
        self._act_op = Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 dilation=1):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu')

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='gelu',
            dilation=dilation)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)

        ####################################################################
        # If given dilation rate > 1, using corresponding padding.
        # The performance drops down without the follow padding.
        if self.dilation > 1:
            padding = self.dilation
            y = F.pad(y, [padding, padding, padding, padding])
        #####################################################################

        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='gelu')
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)

        return y


class ResNet_vd(nn.Layer):
    """
    The ResNet_vd implementation based on PaddlePaddle.

    The original article refers to Jingdong
    Tong He, et, al. "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    (https://arxiv.org/pdf/1812.01187.pdf).

    Args:
        layers (int, optional): The layers of ResNet_vd. The supported layers are (18, 34, 50, 101, 152, 200). Default: 50.
        output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 8.
        multi_grid (tuple|list, optional): The grid of stage4. Defult: (1, 1, 1).
        pretrained (str, optional): The path of pretrained model.

    """

    def __init__(self,
                 layers=50,
                 output_stride=8,
                 multi_grid=(1, 1, 1),
                 pretrained=None):
        super(ResNet_vd, self).__init__()

        self.conv1_logit = None  # for gscnn shape stream
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024
                        ] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layers >= 50 else num_filters

        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}

        self.conv1_1 = ConvBNLayer(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, act='relu')
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu')
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # self.block_list = []
        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    ###############################################################################

                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0
                            and dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            dilation=dilation_rate))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pretrained = pretrained

    def forward(self, inputs: NestedTensor):
        data = paddle.cast(inputs.tensors, dtype='float32')
        y = self.conv1_1(data)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        #feat_list = []
        feat_list: Dict[str, NestedTensor] = {}
        for index,stage in enumerate(self.stage_list):
            m = inputs.mask
            for block in stage:
                y = block(y)
            m = paddle.unsqueeze(m, axis=0)
            mask = F.interpolate(paddle.cast(m, dtype='float'), size=y.shape[-2:], mode='nearest')[0]
            mask = paddle.squeeze(m, axis=0)
            mask = paddle.cast(mask, dtype='bool')
            feat_list['layer' + str(index)] = NestedTensor(y, mask)

        return feat_list




def ResNet18_vd(**args):
    model = ResNet_vd(layers=18, **args)
    return model


def ResNet34_vd(**args):
    model = ResNet_vd(layers=34, **args)
    return model


def ResNet50_vd(**args):
    model = ResNet_vd(layers=50, **args)
    return model


def ResNet101_vd(**args):
    model = ResNet_vd(layers=101, **args)
    return model


def ResNet152_vd(**args):
    model = ResNet_vd(layers=152, **args)
    return model


def ResNet200_vd(**args):
    model = ResNet_vd(layers=200, **args)
    return model
