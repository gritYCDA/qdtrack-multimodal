# Copyright (c) OpenMMLab. All rights reserved.
from .l2_loss import L2Loss
from .triplet_loss import TripletLoss
from .multipos_cross_entropy_loss import MultiPosCrossEntropyLoss

__all__ = ['L2Loss', 'TripletLoss', 'MultiPosCrossEntropyLoss']
