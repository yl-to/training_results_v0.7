# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList

class ScoreHLRSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs,
                 regression_targets,
                 features,
                 box_head,
                 prop_boxes,
                 image_sizes,
                 real_matched_idxs,
                 is_rpn=0, objectness=None):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        num_images = len(matched_idxs)
        if num_images == 1:
            pos_idx = []
            neg_idx = []
            matched_idxs = [matched_idxs.view(-1)]
            # there is actually only 1 iteration of this for loop, but keeping the loop for completeness
            for matched_idxs_per_image in matched_idxs:
                if objectness is not None:
                    objectness = objectness.view(-1)
                    positive = torch.nonzero((matched_idxs_per_image >= 1)*(objectness > -1) ).squeeze(1)
                    negative = torch.nonzero((matched_idxs_per_image == 0)*(objectness > -1)).squeeze(1)
                else:
                    positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
                    negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
    
                num_pos = int(self.batch_size_per_image * self.positive_fraction)
                # protect against not enough positive examples
                num_pos = min(positive.numel(), num_pos)
                num_neg = self.batch_size_per_image - num_pos
                # protect against not enough negative examples
                num_neg = min(negative.numel(), num_neg)
    
                # randomly select positive and negative examples
                perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
                pos_idx_per_image = positive.index_select(0, perm1)
                neg_idx_per_image = negative.index_select(0, perm2)
                # negaive is the "neg_inds" in _sample_neg().
                # num_neg is "num_expected"

                with torch.no_grad():
                    print('debug')
                    box = BoxList(prop_boxes[0][negative], image_size=image_sizes[0])
                    box.add_field("matched_idxs", real_matched_idxs[0][negative])
                    box.add_field("regression_targets", regression_targets[0][negative])
                    box.add_field("labels", matched_idxs[0][negative])
                    bbox_result = box_head(features, [box])
                    loss_cls, loss_box = bbox_result[2]['loss_classifier'], bbox_result[2]['loss_box_reg']
                    cls_score, bbox_pred = bbox_result[3], bbox_result[4]
                    import pdb; pdb.set_trace()
                    ori_loss = box_head.loss_evaluator([cls_score.float()], None, [box])[0]
                    max_score, argmax_score = cls_score.softmax(-1)[:, :-1].max(-1)
                    SCORE_THR = 0.05 # this should be passed as parameter
                    valid_inds = (max_score > SCORE_THR).nonzero().view(-1)
                    invalid_inds = (max_score <= SCORE_THR).nonzero().view(-1)
                    num_valid = valid_inds.size(0)
                    num_invalid = invalid_inds.size(0)

                    num_hlr = min(num_valid, num_neg)
                    num_rand = num_neg - num_hlr
                    # if num_valid > 0:
                    #     valid_rois = neg_rois[valid_inds]
                    #     valid_max_score = max_score[valid_inds]
                    #     valid_argmax_score = argmax_score[valid_inds]
                    #     valid_bbox_pred = bbox_pred[valid_inds]
                    #     # valid_bbox_pred shape: [num_valid, #num_classes, 4]
                    #     valid_bbox_pred = valid_bbox_pred.view(
                    #         valid_bbox_pred.size(0), -1, 4)
                    #     selected_bbox_pred = valid_bbox_pred[range(num_valid), valid_argmax_score]

                # create binary mask from indices
                pos_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image, dtype=torch.bool
                )
                neg_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image, dtype=torch.bool
                )
                pos_idx_per_image_mask.index_fill_(0, pos_idx_per_image, 1)
                neg_idx_per_image_mask.index_fill_(0, neg_idx_per_image, 1)
    
                pos_idx.append(pos_idx_per_image_mask)
                neg_idx.append(neg_idx_per_image_mask)

                return pos_idx, neg_idx, bbox_result

        ## this implements a batched random subsampling using a tensor of random numbers and sorting
        if is_rpn:
            print('not single image')
            import pdb; pdb.set_trace()
            num_anchors_per_im = matched_idxs[0].size(0)
            num_images = len(matched_idxs)
            matched_idxs_cat = matched_idxs
            device = matched_idxs_cat.device
            ## generate a mask for positive samples
            pos_samples_mask = matched_idxs_cat >= 1
            num_pos_samples = pos_samples_mask.sum(dim=1)
            num_pos_samples_cum = num_pos_samples.cumsum(dim=0)
            max_pos_samples = torch.max(num_pos_samples)
            ## we are generating a tensor of consecutive numbers for each row.  
            consec = torch.arange(max_pos_samples, device = device).repeat(num_images,1)
            mask_to_hide = consec >= num_pos_samples.view(num_images,1)
            ## generate a tensor of random numbers, than fill the non-valid elements with 2 so 
            ## they are at the end when sorted
            rand_nums_batched = torch.rand([num_images, max_pos_samples], device=device)
            rand_nums_batched.masked_fill_(mask_to_hide, 2)
            rand_perm = rand_nums_batched.argsort(dim=1)
            max_pos_allowed = int(self.batch_size_per_image * self.positive_fraction)
            num_pos_subsamples = num_pos_samples.clamp(max=max_pos_allowed)
            subsampling_mask = rand_perm < num_pos_subsamples.view(num_images,1)
            if num_images>1:
                consec[1:,:] = consec[1:,:] + num_pos_samples_cum[:-1].view(num_images-1,1)
            sampling_inds = consec.masked_select(subsampling_mask)
            pos_samples_inds = pos_samples_mask.view(-1).nonzero().squeeze(1)
            pos_subsampled_inds = pos_samples_inds[sampling_inds]
            ## do the same for negative samples as well
            neg_samples_mask = matched_idxs_cat == 0
            num_neg_samples = neg_samples_mask.sum(dim=1)
            num_neg_samples_cum = num_neg_samples.cumsum(dim=0)
            max_neg_samples = torch.max(num_neg_samples)
            consec = torch.arange(max_neg_samples, device = device)
            consec = consec.repeat(num_images,1)
            mask_to_hide = consec >= num_neg_samples.view(num_images,1)
            rand_nums_batched = torch.rand([num_images, max_neg_samples], device=device)
            rand_nums_batched.masked_fill_(mask_to_hide, 2)
            rand_perm = rand_nums_batched.argsort(dim=1)
            num_subsamples = torch.min(num_neg_samples, self.batch_size_per_image - num_pos_subsamples) 
            subsampling_mask = rand_perm < num_subsamples.view(num_images,1)
            if num_images>1:
                consec[1:,:] = consec[1:,:] + num_neg_samples_cum[:-1].view(num_images-1,1)
            sampling_inds = consec.masked_select(subsampling_mask)
            neg_samples_inds = neg_samples_mask.view(-1).nonzero().squeeze(1)
            neg_subsampled_inds = neg_samples_inds[sampling_inds]
            return pos_subsampled_inds, neg_subsampled_inds
        else:
            matched_idxs_cat = matched_idxs
            device = matched_idxs_cat.device
            pos_samples_mask =( matched_idxs_cat >= 1) * (objectness > -1)
            num_pos_samples = pos_samples_mask.sum(dim=1)
            num_pos_samples_cum = num_pos_samples.cumsum(dim=0)
            max_pos_samples = torch.max(num_pos_samples)
            consec = torch.arange(max_pos_samples, device = device).repeat(num_images,1)
            mask_to_hide = consec >= num_pos_samples.view(num_images,1)
            rand_nums_batched = torch.rand([num_images, max_pos_samples], device=device)
            rand_nums_batched.masked_fill_(mask_to_hide, 2)
            rand_perm = rand_nums_batched.argsort(dim=1)
            max_pos_allowed = int(self.batch_size_per_image * self.positive_fraction)
            num_pos_subsamples = num_pos_samples.clamp(max=max_pos_allowed)
            subsampling_mask = rand_perm < num_pos_subsamples.view(num_images,1)
            if num_images>1:
                consec[1:,:] = consec[1:,:] + num_pos_samples_cum[:-1].view(num_images-1,1)
            sampling_inds = consec.masked_select(subsampling_mask)
            pos_samples_inds = pos_samples_mask.view(-1).nonzero().squeeze(1)
            pos_subsampled_inds = pos_samples_inds[sampling_inds]
    
            neg_samples_mask = (matched_idxs_cat == 0) *( objectness>-1)
            num_neg_samples = neg_samples_mask.sum(dim=1)
            num_neg_samples_cum = num_neg_samples.cumsum(dim=0)
            max_neg_samples = torch.max(num_neg_samples)
            consec = torch.arange(max_neg_samples, device = device)
            consec = consec.repeat(num_images,1)
            mask_to_hide = consec >= num_neg_samples.view(num_images,1)
            rand_nums_batched = torch.rand([num_images, max_neg_samples], device=device)
            rand_nums_batched.masked_fill_(mask_to_hide, 2)
            rand_perm = rand_nums_batched.argsort(dim=1)
            num_subsamples = torch.min(num_neg_samples, self.batch_size_per_image - num_pos_subsamples)
            subsampling_mask = rand_perm < num_subsamples.view(num_images,1)
            if num_images>1:
                consec[1:,:] = consec[1:,:] + num_neg_samples_cum[:-1].view(num_images-1,1)
            sampling_inds = consec.masked_select(subsampling_mask)
            neg_samples_inds = neg_samples_mask.view(-1).nonzero().squeeze(1)
            neg_subsampled_inds = neg_samples_inds[sampling_inds]
            return pos_subsampled_inds, neg_subsampled_inds, num_pos_subsamples, num_subsamples

