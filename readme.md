## Detection Transformer
This is repositories that implement the paper **"End-to-End Object Detection with Transformer"**

We will use paddlepaddle 2.0 to implement those model.Then we will use the finished models to enter the competition "飞浆论文复现大赛", which was operated by baidu company.

The reference repositories is "https://github.com/facebookresearch/detr"

### Folder list
**1. models**
In this folder, we will implement our top level model class and some sub class.
It will contain these file:
(1) resnetvd.py             
> This file implement the resnet backbone.
> It was finished by Baidu.
(2) detr.py
> This file implement the DETR model. 
(3) matcher.py
> This file implement the HungarianMatcher.
(4) position_encoding.py
> This file implement the position_encoding class.
> It contain sine position encoding class and Learned position embedding.
(5) transformer.py
> In this file, we will implement the transformer structure.
> Maybe we can use paddle2.0's transformer classes.

**2.utils**
In this folder, we will implement some tool class.
(1) box_ops
(2) misc.py
(3) plot_utils.py