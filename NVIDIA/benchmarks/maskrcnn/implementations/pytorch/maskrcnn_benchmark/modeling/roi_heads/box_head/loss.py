# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.utils import cat
from torch.nn.utils.rnn import pad_sequence

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def __call__(self, class_logits, box_regression, proposals):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels.index_select(0, sampled_pos_inds_subset)
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        index_select_indices=((sampled_pos_inds_subset[:,None]) * box_regression.size(1) + map_inds).view(-1)
        box_regression_sampled=box_regression.view(-1).index_select(0, index_select_indices).view(map_inds.shape[0], 
                                                                                                  map_inds.shape[1]) 
        regression_targets_sampled = regression_targets.index_select(0, sampled_pos_inds_subset)

        box_loss = smooth_l1_loss(
            box_regression_sampled,
            regression_targets_sampled,
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
