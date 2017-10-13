## full1\_D

1/8 image size feature map

using conv3\_4, conv4\_4, conv5\_4

### full1\_D1

1/8 image size feature map

using conv2\_3, conv3\_4, conv4\_4, conv5\_4


### full1\_D2

1/4 image size feature map

using conv3\_4, conv4\_4, conv5\_4

### full1\_DRoiAlignX\_FourParts\_Ohem\_Repool\_Loss1
all loss weights are 1

### full1\_DRoiAlignX\_FourParts\_Ohem\_Repool
all loss weights are 0.25, except rpn's loss weights are 1


### full1\_DRoiAlignX\_FourParts\_Ohem\_RepoolH
all loss weights are 1, just repool head


### full1\_DRoiAlignX\_FourParts\_Ohem\_RepoolFix
fix repool cls change which caused label error
可能repool后的分类器，由于正负样本不均衡，导致分类效果不好．


### full1\_DRoiAlignX\_FourParts\_Ohem\_RepoolRemove
除了rpn, loss都是0.25, 去除repool的cls预测


### full1\_DRoiAlignX\_kp
不仅回归bbox，还回归kp的位置


### full1\_DRoiAlignX\_kpCls
不仅回归kp的位置，还预测kp的类别


### kp\_cluster
将手工设计的anchor，用聚类得到的anchor来代替

### aichal\_cluster
将手工设计的anchor，用聚类得到的anchor来代替，并且只预测bbox
