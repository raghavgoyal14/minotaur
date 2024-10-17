# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
# import os
# import json
# import shutil
import math
import sys
# import copy
from typing import Dict, Iterable, Optional
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

import torch
import torch.nn
import torch.optim

import util.dist as dist
# from datasets.vidstg_eval import VidSTGEvaluator
# from datasets.hcstvg_eval import HCSTVGEvaluator
# from datasets.vq2d_eval import VQ2DEvaluator
# from datasets.vq2d_orig_eval import VQ2DOrigEvaluator
# from datasets.nlq_orig_eval import NLQOrigEvaluator
# from datasets.mq_orig_eval import MQOrigEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema
# from scipy.signal import find_peaks, medfilt
from util.misc import targets_to, NestedTensor

# from pathlib import Path
# from PIL import Image


def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    data_loader: Dict,
    weight_dict: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
    writer=None,
):
    model.train()
    if criterion is not None:
        criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = args.train_flags.print_freq
    num_training_steps = int(len(data_loader) * args.epochs)

    for i, batch_dict in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # if args.debug:
        # print(f"Fetched batch {i} | Rank: {dist.get_rank()} | "
        #     f"Task: {batch_dict['task_name']} | Samples shape: {batch_dict['samples'].tensors.shape}")

        task_name = batch_dict['task_name'][0]
        curr_step = epoch * len(data_loader) + i

        samples = batch_dict["samples"].to(device)
        if "samples_fast" in batch_dict:
            samples_fast = batch_dict["samples_fast"].to(device)
        else:
            samples_fast = None

        durations = batch_dict["durations"]
        captions = batch_dict["captions"]
        targets = batch_dict["targets"]
        targets = targets_to(targets, device)

        kwargs = {}
        # flag_bg_segment
        if task_name in ["mq", "nlq", "vq2d"]:
            segment_type_selected = batch_dict['segment_type_selected']
            assert len(segment_type_selected) == 1  # BS = 1 constraint
            flag_bg_segment = segment_type_selected[0] == "bg"

        if task_name == "nlq":
            if args.model.nlq.use_sentence_text_embeddings:
                kwargs["key_for_nlq_sentence_embedding"] = (
                    batch_dict["video_ids"][0], batch_dict["annotation_uid"][0], batch_dict["query_idx"][0]
                )

        if task_name == "vq2d":
            kwargs["reference_crop"] = batch_dict["reference_crop"].to(device)

        # if args.debug: import ipdb; ipdb.set_trace()

        # forward
        memory_cache = model(
            task_name,
            samples,
            durations,
            captions,
            encode_and_save=True,
            samples_fast=samples_fast,
            **kwargs
        )
        outputs = model(
            task_name,
            samples,
            durations,
            captions,
            encode_and_save=False,
            memory_cache=memory_cache,
            **kwargs
        )

        # only keep box predictions in the annotated moment
        max_duration = max(durations)
        inter_idx = batch_dict["inter_idx"]
        if not flag_bg_segment:
            keep_list = []
            for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
                keep_list.extend(
                    [
                        elt
                        for elt in range(
                            i_dur * max_duration + inter[0],
                            (i_dur * max_duration) + inter[1] + 1,
                        )
                    ]
                )
            keep = torch.tensor(keep_list).long().to(device)
            outputs["pred_boxes"] = outputs["pred_boxes"][keep]
            for i_aux in range(len(outputs["aux_outputs"])):
                outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux][
                    "pred_boxes"
                ][keep]
        b = len(durations)
        targets = [
            x for x in targets if len(x["boxes"])
        ]  # keep only targets in the annotated moment
        if not flag_bg_segment:
            assert len(targets) == len(outputs["pred_boxes"]), (
                len(outputs["pred_boxes"]),
                len(targets),
            )
        # mask with padded positions set to False for loss computation
        if args.sted:
            time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        # compute losses
        loss_dict = {}
        # if args.debug: import ipdb; ipdb.set_trace()
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, inter_idx, time_mask, segment_type_selected, task_name))

        # if args.debug: import ipdb; ipdb.set_trace()
        # loss scaling
        for k, v in loss_dict.items():
            loss_dict[k] = v * args.joint.scale_loss[task_name]

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        loss_dict_scaled = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
        # print(f"task_name: {task_name}, losses: {losses}")

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {
        #     f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        # }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        loss_dict_reduced_scaled_task = {f"{k}_{task_name}": v for k, v in loss_dict_scaled.items()}
        # loss_dict_reduced_unscaled_task = {f"{k}_{task_name}": v for k, v in loss_dict.items()}

        # if args.debug: import ipdb; ipdb.set_trace()
        metric_logger.update(
            **{"loss_total": loss_value}, **{f"loss_{task_name}": losses}, **loss_dict_reduced_scaled_task,
            # **loss_dict_reduced_unscaled_task
        )

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
