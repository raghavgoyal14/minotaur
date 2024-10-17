# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TubeDETR model and criterion classes.
"""
from typing import Dict, Optional

import json
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import math

import util.dist as dist
from util import box_ops
from pathlib import Path
from util.misc import NestedTensor

# from .backbone import build_backbone
# from .transformer_mq import build_transformer


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x


class TubeDETR(nn.Module):
    """This is the TubeDETR module that performs spatio-temporal video grounding"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        args=None,
    ):
        """
        :param backbone: visual backbone model
        :param transformer: transformer model
        :param num_queries: number of object queries per frame
        :param aux_loss: whether to use auxiliary losses at every decoder layer
        :param video_max_len: maximum number of frames in the model
        :param stride: temporal stride k
        :param guided_attn: whether to use guided attention loss
        :param fast: whether to use the fast branch
        :param fast_mode: which variant of fast branch to use
        :param sted: whether to predict start and end proba
        """
        super().__init__()
        assert args is not None
        self.args = args
        self.backbone = backbone
        self.transformer = transformer

        # common variables to all tasks
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.args.model.use_single_query_embed:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        else:
            self.query_embed = nn.ModuleDict({
                "visual": nn.Embedding(self.num_queries, hidden_dim),
                "text": nn.Embedding(self.num_queries, hidden_dim),
            })

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # self.video_max_len = video_max_len
        # self.stride = stride
        # self.guided_attn = guided_attn
        # self.fast = fast
        # self.sted = sted
        # self.use_score_per_frame = use_score_per_frame
        # self.use_segment_type_classification = use_segment_type_classification
        # self.use_frame_action_classification = use_frame_action_classification
        # self.no_spatial_feature_map = no_spatial_feature_map

        # common
        if self.args.sted:
            self.sted_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.2)
        if self.args.model.use_score_per_frame:
            self.score_per_frame_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.2)

        # mq
        if "mq" in self.args.tasks.names:
            mq_class_to_idx_file = Path(args.data.mq.path) / "moment_classes_idx.json"
            assert mq_class_to_idx_file.is_file()

            with mq_class_to_idx_file.open("rb") as fp:
                mq_class_to_idx = json.load(fp)
                self.mq_class_to_idx = mq_class_to_idx

            if self.args.model.mq.use_segment_type_classification:
                self.mq_segment_type_embed = MLP(hidden_dim, hidden_dim, 4, 2, dropout=0.2)
            if self.args.model.mq.use_frame_action_classification:
                self.mq_action_classifier = MLP(hidden_dim, hidden_dim, len(self.mq_class_to_idx), 2, dropout=0.2)

            if self.args.model.mq.concat_slowfast_features:
                raise ValueError("Currently model.mq.concat_slowfast_features is turned off")

        if "nlq" in self.args.tasks.names:
            pass

        if "vq2d" in self.args.tasks.names:
            if self.args.model.vq2d.use_projection_layer:
                self.input_proj_query = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def _task_specific_variables_for_forward(self, task_name):
        """
        Setting up task-specific variables
        - stride
        """
        if self.training:
            flags_task_specific = self.args.train_flags
        else:
            flags_task_specific = self.args.eval_flags

        assert task_name in flags_task_specific.keys()
        stride = flags_task_specific[task_name].stride

        return stride

    def forward(
        self,
        task_name: str,
        samples,
        durations,
        captions,
        encode_and_save=True,
        memory_cache=None,
        samples_fast=None,
        **kwargs,
    ):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched frames, of shape [n_frames x 3 x H x W]
           - samples.mask: a binary mask of shape [n_frames x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        stride = self._task_specific_variables_for_forward(task_name)

        # Start
        if not self.args.flop_count_mode and not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            b = len(durations)
            t = max(durations)

            if self.args.flop_count_mode:
                features, pos = samples['features'], samples['pos']
            else:
                features, pos = self.backbone(
                    samples,
                    # no_spatial_feature_map=False,
                )  # each frame from each video is forwarded through the backbone
                    # features: torch.Size([8, 2048, 5, 6]) pos: torch.Size([8, 256, 5, 6]))

            if isinstance(features[-1], dict):
                src, mask = features[-1]['tensors'], features[-1]['mask']
            else:
                src, mask = features[-1].decompose()  # src (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32)); mask (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))

            if task_name == "vq2d" and not self.training and self.args.flops_exp.enable:
                indices_orig = torch.arange(src.shape[0])
                indices_subsampled = indices_orig[::self.args.flops_exp.downsample_factor]
                indices_upsampled = []
                for e in indices_orig:
                    cc = (e - indices_subsampled).abs()
                    indices_upsampled.append(indices_subsampled[cc.argmin().item()].item())
                indices_upsampled = torch.LongTensor(indices_upsampled)

                # if self.args.debug: import ipdb; ipdb.set_trace()
                src = src[indices_upsampled]

                # src_downsampled = src[::self.args.flops_exp.downsample_factor]
                # src = src_downsampled.repeat_interleave(self.args.flops_exp.downsample_factor, dim=0)

            if self.args.fast:
                with torch.no_grad():  # fast branch does not backpropagate to the visual backbone
                    features_fast, pos_fast = self.backbone(samples_fast)
                src_fast, mask_fast = features_fast[-1].decompose()
                src_fast = self.input_proj(src_fast)

            # encode reference crop
            if task_name == "vq2d":
                # if self.args.debug: import ipdb; ipdb.set_trace()
                if self.args.model.vq2d.freeze_backbone_during_query_projection:
                    with torch.no_grad():
                        features_reference_crop, pos_reference_crop = self.backbone(kwargs["reference_crop"])
                else:
                    features_reference_crop, pos_reference_crop = self.backbone(kwargs["reference_crop"])

                if isinstance(features_reference_crop[-1], dict):
                    src_reference_crop, mask_reference_crop = features_reference_crop[-1]['tensors'], features_reference_crop[-1]['mask']
                else:
                    src_reference_crop, mask_reference_crop = features_reference_crop[-1].decompose()
                if self.args.model.vq2d.use_projection_layer:
                    kwargs["src_reference_crop"] = self.input_proj_query(src_reference_crop)  # torch.Size([8, 256, 5, 6])
                else:
                    kwargs["src_reference_crop"] = self.input_proj(src_reference_crop)  # torch.Size([8, 256, 5, 6])

            # if self.args.debug: import ipdb; ipdb.set_trace()
            # temporal padding pre-encoder
            src = self.input_proj(src)  # torch.Size([8, 256, 5, 6])
            _, f, h, w = src.shape
            f2 = pos[-1].size(1)
            device = src.device
            tpad_mask_t = None
            fast_src = None
            if not stride:
                tpad_src = torch.zeros(b, t, f, h, w).to(device)
                tpad_mask = torch.ones(b, t, h, w).bool().to(device)
                pos_embed = torch.zeros(b, t, f2, h, w).to(device)
                cur_dur = 0
                for i_dur, dur in enumerate(durations):
                    tpad_src[i_dur, :dur] = src[cur_dur: cur_dur + dur]
                    tpad_mask[i_dur, :dur] = mask[cur_dur: cur_dur + dur]
                    pos_embed[i_dur, :dur] = pos[-1][cur_dur: cur_dur + dur]
                    cur_dur += dur
                tpad_src = tpad_src.view(b * t, f, h, w)
                tpad_mask = tpad_mask.view(b * t, h, w)

                if self.args.flop_count_mode:
                    tpad_mask_long = tpad_mask.long()
                    tpad_mask_long[:, 0, 0] = 0
                    tpad_mask = tpad_mask_long.bool()
                else:
                    tpad_mask[:, 0, 0] = False  # avoid empty masks
                pos_embed = pos_embed.view(b * t, f2, h, w)
            else:  # temporal sampling
                n_clips = math.ceil(t / stride)
                tpad_src = src
                tpad_mask = mask
                pos_embed = pos[-1]
                if self.args.fast:
                    fast_src = torch.zeros(b, t, f, h, w).to(device)
                tpad_mask_t = (
                    torch.ones(b, t, h, w).bool().to(device)
                )  # temporally padded mask for all frames, will be used for the decoding
                cum_dur = 0  # updated for every video
                cur_dur = 0
                cur_clip = 0
                for i_dur, dur in enumerate(durations):
                    if self.args.fast:
                        fast_src[i_dur, :dur] = src_fast[cum_dur: cum_dur + dur]
                        tpad_mask_t[i_dur, :dur] = mask_fast[cum_dur: cum_dur + dur]
                    else:
                        for i_clip in range(math.ceil(dur / stride)):
                            clip_dur = min(stride, dur - i_clip * stride)
                            tpad_mask_t[
                                i_dur, cur_dur - cum_dur: cur_dur - cum_dur + clip_dur
                            ] = mask[cur_clip: cur_clip + 1].repeat(clip_dur, 1, 1)
                            cur_dur += clip_dur
                            cur_clip += 1
                    cum_dur += dur
                tpad_src = tpad_src.view(b * n_clips, f, h, w)
                tpad_mask = tpad_mask.view(b * n_clips, h, w)
                pos_embed = pos_embed.view(b * n_clips, f, h, w)
                tpad_mask_t = tpad_mask_t.view(b * t, h, w)
                if self.args.fast:
                    fast_src = fast_src.view(b * t, f, h, w)

                if self.args.flop_count_mode:
                    tpad_mask_long = tpad_mask.long()
                    tpad_mask_t_long = tpad_mask_t.long()
                    tpad_mask_long[:, 0, 0] = 0
                    tpad_mask_t_long[:, 0, 0] = 0
                    tpad_mask = tpad_mask_long.bool()
                    tpad_mask_t = tpad_mask_t.bool()
                else:
                    tpad_mask[:, 0, 0] = False  # avoid empty masks
                    tpad_mask_t[:, 0, 0] = False  # avoid empty masks

            # query embed
            # if self.args.debug: import ipdb; ipdb.set_trace()
            kwargs["vq2d_use_object_text"] = False
            if self.args.model.use_single_query_embed:
                query_embed = self.query_embed.weight
            else:
                if task_name in ['vq2d']:
                    query_embed = self.query_embed['visual'].weight

                    # use object title as query
                    if self.args.model.vq2d.use_text_query:
                        p_text_query = self.args[f"{'train' if self.training else 'eval'}_flags"].vq2d.p_text_query
                        kwargs["vq2d_use_object_text"] = np.random.random() < p_text_query
                    else:
                        kwargs["vq2d_use_object_text"] = False

                    if kwargs["vq2d_use_object_text"]:
                        query_embed = self.query_embed['text'].weight

                elif task_name in ['nlq', 'mq']:
                    query_embed = self.query_embed['text'].weight
                else:
                    raise ValueError("Task name: {task_name} not recognized")

            # video-text encoder
            memory_cache = self.transformer(
                task_name,
                tpad_src,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                tpad_mask,  # (n_clips)x(math.ceil(H/32))x(math.ceil(W/32))
                query_embed,  # num_queriesxF
                pos_embed,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                captions,  # list of length batch_size
                encode_and_save=True,
                durations=durations,  # list of length batch_size
                tpad_mask_t=tpad_mask_t,  # (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))
                fast_src=fast_src,  # (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32))
                **kwargs
            )
            return memory_cache

        else:
            assert memory_cache is not None
            # space-time decoder
            hs = self.transformer(
                task_name=task_name,
                img_memory=memory_cache[
                    "img_memory"
                ],  # (math.ceil(H/32)*math.ceil(W/32) + n_tokens)x(BT)xF
                mask=memory_cache[
                    "mask"
                ],  # (BT)x(math.ceil(H/32)*math.ceil(W/32) + n_tokens)
                pos_embed=memory_cache["pos_embed"],  # n_tokensx(BT)xF
                query_embed=memory_cache["query_embed"],  # (num_queries)x(BT)xF
                query_mask=memory_cache["query_mask"],  # Bx(Txnum_queries)
                encode_and_save=False,
                text_memory=memory_cache["text_memory"],
                text_mask=memory_cache["text_attention_mask"],
                **kwargs
            )
            if self.args.guided_attn:
                hs, weights, cross_weights = hs
            out = {}

            # outputs heads
            if self.args.sted:
                outputs_sted = self.sted_embed(hs)

            if self.args.model.use_score_per_frame:
                outputs_score_per_frame = self.score_per_frame_embed(hs)

            # mq
            if task_name == "mq":
                if self.args.model.mq.use_segment_type_classification:
                    outputs_segment_type = self.mq_segment_type_embed(hs.mean(2))

                if self.args.model.mq.use_frame_action_classification:
                    outputs_action_per_frame = self.mq_action_classifier(hs)

            hs = hs.flatten(1, 2)  # n_layersxbxtxf -> n_layersx(b*t)xf

            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update({"pred_boxes": outputs_coord[-1]})

            # common
            if self.args.sted:
                out.update({"pred_sted": outputs_sted[-1]})
            if self.args.model.use_score_per_frame:
                out.update({"pred_score_per_frame": outputs_score_per_frame[-1]})
            if self.args.guided_attn:
                out["weights"] = weights[-1]
                out["ca_weights"] = cross_weights[-1]

            # mq
            if task_name == "mq":
                if self.args.model.mq.use_segment_type_classification:
                    out.update({"segment_type": outputs_segment_type[-1]})
                if self.args.model.mq.use_frame_action_classification:
                    out.update({"action_per_frame": outputs_action_per_frame[-1]})

            # auxiliary outputs
            if self.aux_loss:
                out["aux_outputs"] = [
                    {
                        "pred_boxes": b,
                    }
                    for b in outputs_coord[:-1]
                ]
                for i_aux in range(len(out["aux_outputs"])):
                    if self.args.sted:
                        out["aux_outputs"][i_aux]["pred_sted"] = outputs_sted[i_aux]
                    if self.args.model.use_score_per_frame:
                        out["aux_outputs"][i_aux]["pred_score_per_frame"] = outputs_score_per_frame[i_aux]
                    if self.args.guided_attn:
                        out["aux_outputs"][i_aux]["weights"] = weights[i_aux]
                        out["aux_outputs"][i_aux]["ca_weights"] = cross_weights[i_aux]

                    if task_name == "mq":
                        if self.args.model.mq.use_segment_type_classification:
                            out["aux_outputs"][i_aux]["segment_type"] = outputs_segment_type[i_aux]
                        if self.args.model.mq.use_frame_action_classification:
                            out["aux_outputs"][i_aux]["action_per_frame"] = outputs_action_per_frame[i_aux]
            return out


class SetCriterion(nn.Module):
    """This class computes the loss for TubeDETR."""

    def __init__(self, args, losses, sigma=1):
        """Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            sigma: standard deviation for the Gaussian targets in the start and end Kullback Leibler divergence loss
        """
        super().__init__()
        self.args = args
        self.losses = losses
        self.sigma = sigma

    def loss_boxes(self, outputs, targets, num_boxes, segment_type_selected, task_name, device):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        if (task_name in ['mq', 'nlq']
            or segment_type_selected[0] == "bg"
        ):
            return {"loss_bbox": torch.tensor(0.0).to(device),
                    "loss_giou": torch.tensor(0.0).to(device)}

        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / max(num_boxes, 1)

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / max(num_boxes, 1)
        return losses

    def loss_sted(self, outputs, num_boxes, inter_idx, positive_map, action_class_map,
                  time_mask=None, segment_type_selected=False, task_name=None, device=None):
        """Compute the losses related to the start & end prediction, a KL divergence loss
        targets dicts must contain the key "pred_sted" containing a tensor of logits of dim [T, 2]
        """
        # if self.args.debug: import ipdb; ipdb.set_trace()

        # return 0 loss when bg
        if segment_type_selected[0] == "bg":
            return {"loss_sted": torch.tensor(0.0).to(device)}

        assert "pred_sted" in outputs
        sted = outputs["pred_sted"]
        losses = {}

        target_start = torch.tensor([x[0] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        target_end = torch.tensor([x[1] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        sted = sted.masked_fill(
            ~time_mask[:, :, None], -1e32
        )  # put very low probability on the padded positions before softmax
        eps = 1e-6  # avoid log(0) and division by 0

        sigma = self.sigma
        start_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_start[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
        pred_start_prob = (sted[:, :, 0]).softmax(1)
        loss_start = (
            pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()
        )  # KL div loss
        loss_start = loss_start * time_mask  # not count padded values in the loss

        end_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_end[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
        pred_end_prob = (sted[:, :, 1]).softmax(1)
        loss_end = (
            pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()
        )  # KL div loss
        loss_end = loss_end * time_mask  # do not count padded values in the loss

        loss_sted = loss_start + loss_end
        losses["loss_sted"] = loss_sted.mean()

        return losses

    def loss_guided_attn(
        self, outputs, num_boxes, inter_idx, positive_map, action_class_map,
        time_mask=None, segment_type_selected=False, task_name=None, device=None
    ):
        """Compute guided attention loss
        targets dicts must contain the key "weights" containing a tensor of attention matrices of dim [B, T, T]
        """
        # return 0 loss when bg
        if segment_type_selected[0] == "bg":
            return {"loss_guided_attn": torch.tensor(0.0).to(device)}

        weights = outputs["weights"]  # BxTxT

        positive_map = positive_map + (
            ~time_mask
        )  # the padded positions also have to be taken out
        eps = 1e-6  # avoid log(0) and division by 0

        loss = -(1 - weights + eps).log()
        loss = loss.masked_fill(positive_map[:, :, None], 0)
        nb_neg = (~positive_map).sum(1) + eps
        loss = loss.sum(2) / nb_neg[:, None]  # sum on the column
        loss = loss.sum(1)  # mean on the line normalized by the number of negatives
        loss = loss.mean()  # mean on the batch

        losses = {"loss_guided_attn": loss}
        return losses

    def loss_score_per_frame(
        self, outputs, num_boxes, inter_idx, positive_map, action_class_map,
        time_mask=None, segment_type_selected=False, task_name=None, device=None,
    ):
        positive_map = positive_map + (~time_mask)
        pos_weight = None
        if positive_map.float().sum() > 0:
            pos_weight = positive_map.float().shape[1] / positive_map.float().sum()
        loss = F.binary_cross_entropy_with_logits(
            outputs['pred_score_per_frame'].squeeze(-1),
            positive_map.float(),
            pos_weight=pos_weight
        )

        losses = {"loss_score_per_frame": loss}
        return losses

    def loss_segment_type(
        self, outputs, num_boxes, inter_idx, positive_map, action_class_map,
        time_mask=None, segment_type_selected=False, task_name=None, device=None
    ):
        if task_name != "mq":
            return {"loss_segment_type": torch.tensor(0.0).to(device)}
        segment_types = ["fg", "left_trun", "right_trun", "bg"]  # match it w/ mq.py
        target_segment_type = torch.Tensor(
            [segment_types.index(segment_type_selected[0])]).long().to(positive_map.device)
        # import ipdb; ipdb.set_trace()
        loss = F.cross_entropy(outputs['segment_type'], target_segment_type)
        losses = {"loss_segment_type": loss}
        return losses

    # def loss_frame_action_classification(self, outputs, inter_idx, target_map, segment_type_selected):
    def loss_frame_action_classification(
        self, outputs, num_boxes, inter_idx, positive_map, action_class_map,
        time_mask=None, segment_type_selected=False, task_name=None, device=None
    ):
        if task_name != "mq":
            return {"loss_segment_type": torch.tensor(0.0).to(device)}
        device = outputs['action_per_frame'].device
        assert outputs['action_per_frame'].shape[0] == 1  # BS=1

        loss = F.cross_entropy(outputs['action_per_frame'][0], action_class_map[0].long().to(device))
        return {"loss_frame_action_classification": loss}

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        num_boxes,
        inter_idx,
        positive_map,
        action_class_map,
        time_mask,
        segment_type_selected,
        task_name,
        device,
        # **kwargs,
    ):
        loss_map = {
            "boxes": self.loss_boxes,
            "sted": self.loss_sted,
            "guided_attn": self.loss_guided_attn,
            "score_per_frame": self.loss_score_per_frame,
            "segment_type": self.loss_segment_type,
            "frame_action_classification": self.loss_frame_action_classification,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss in ["sted", "guided_attn", "score_per_frame", "segment_type", "frame_action_classification"]:
            return loss_map[loss](
                outputs, num_boxes, inter_idx, positive_map, action_class_map,
                time_mask, segment_type_selected, task_name, device,
            )
        return loss_map[loss](outputs, targets, num_boxes, segment_type_selected, task_name, device)  # for bbox

    def forward(self, outputs, targets, inter_idx=None, time_mask=None,
                segment_type_selected=False, task_name=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == n_annotated_frames.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             inter_idx: list of [start index of the annotated moment, end index of the annotated moment] for each video
             time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the loss computation
        """
        # import ipdb; ipdb.set_trace()
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        device = time_mask.device
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=device
        )
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        if inter_idx is not None and time_mask is not None:
            # construct a map such that positive_map[k, i] = True iff num_frame i lies inside the annotated moment k
            positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
            action_class_map = torch.zeros(time_mask.shape)
            for k, idx in enumerate(inter_idx):
                if idx[0] < 0:  # empty intersection
                    continue
                positive_map[k][idx[0]: idx[1] + 1].fill_(True)
                if task_name == "mq":
                    action_class_map[k][idx[0]: idx[1] + 1].fill_(targets[k]['caption_idx'].item())

            positive_map = positive_map.to(device)
        elif time_mask is None:
            positive_map = None
            action_class_map = None

        # Compute all the requested losses
        # flag_bg_segment = segment_type_selected[0] == "bg"

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    num_boxes,
                    inter_idx,
                    positive_map,
                    action_class_map,
                    time_mask,
                    segment_type_selected,
                    task_name,
                    device,
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    # kwargs = {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        num_boxes,
                        inter_idx,
                        positive_map,
                        action_class_map,
                        time_mask,
                        segment_type_selected,
                        task_name,
                        device,
                        # **kwargs,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_tubedetr(args, backbone, transformer):
    device = torch.device(args.device)

    model = TubeDETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args,
    )
    weight_dict = {
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_sted": args.sted_loss_coef,
    }

    if args.guided_attn:
        weight_dict["loss_guided_attn"] = args.guided_attn_loss_coef
    if args.model.use_score_per_frame:
        weight_dict["loss_score_per_frame"] = args.loss_coef.score_per_frame_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # losses = ["boxes", "sted"] if args.sted else ["boxes"]

    losses = []
    if args.sted:
        losses.append("sted")

    if args.boxes:
        losses.append("boxes")

    if args.guided_attn:
        losses += ["guided_attn"]
    if args.model.use_score_per_frame:
        losses += ["score_per_frame"]

    # vq2d-specific
    pass

    # nlq-specific
    pass

    # mq-specific
    # if args.model.mq.use_segment_type_classification:
    #     losses += ["segment_type"]
    #     weight_dict["loss_segment_type"] = args.segment_type_loss_coef
    # if args.model.mq.use_frame_action_classification:
    #     losses += ["frame_action_classification"]
    #     weight_dict["loss_frame_action_classification"] = args.frame_action_classification_loss_coef

    criterion = SetCriterion(
        args=args,
        losses=losses,
        sigma=args.sigma,
    )
    criterion.to(device)

    return model, criterion, weight_dict
