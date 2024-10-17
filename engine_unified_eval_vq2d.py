# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
import copy
import os
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import seaborn as sns

import util.dist as dist
from pathlib import Path
from PIL import Image
from scipy.signal import find_peaks, medfilt
# from datasets.vidstg_eval import VidSTGEvaluator
# from datasets.vq2d_eval import VQ2DEvaluator
from datasets.vq2d_orig_eval import VQ2DOrigEvaluator
# from datasets.hcstvg_eval import HCSTVGEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to, NestedTensor
from util.optim import adjust_learning_rate, update_ema
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class Wrapper(nn.Module):
    def forward(self, task_name, samples_window, duration, captions,
                encode_and_save, reference_crop):
        memory_cache_window = self.model(
            task_name,
            samples_window,
            duration,
            captions,
            encode_and_save=encode_and_save,
            reference_crop=reference_crop,
        )
        outputs_window = self.model(
            task_name,
            samples_window,
            duration,
            captions,
            encode_and_save=False,
            memory_cache=memory_cache_window,
            reference_crop=reference_crop,
        )
        return outputs_window


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    postprocessors: Dict[str, torch.nn.Module],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        samples = batch_dict["samples"].to(device)
        if "samples_fast" in batch_dict:
            samples_fast = batch_dict["samples_fast"].to(device)
        elif not args.eval_flags.vq2d.stride:
            samples_fast = copy.deepcopy(batch_dict["samples"]).to(device)
        else:
            raise ValueError("asdasd")
        durations = batch_dict["durations"]
        reference_crop = batch_dict["reference_crop"].to(device)
        captions = batch_dict["captions"]
        targets = batch_dict["targets"]

        targets = targets_to(targets, device)

        targets_non_empty_boxes = [x for x in targets if len(x["boxes"])]
        if len(targets_non_empty_boxes) == 0:
            continue

        # forward
        assert len(durations) == 1  # works only on batch size 1
        outputs_aggregate = {}
        outputs_frame_wise = {__i: {} for __i in range(durations[0])}
        window_step_size = min(args.eval_flags.vq2d.window_step_size, args.model.vq2d.video_max_len)
        for ind_start in range(0, durations[0], window_step_size):
            ind_end = min(durations[0], ind_start + args.model.vq2d.video_max_len)
            # print(ind_start, ind_end)

            samples_fast_window = NestedTensor(
                samples_fast.tensors[ind_start: ind_end], samples_fast.mask[ind_start: ind_end])

            if args.eval_flags.vq2d.stride:
                samples_window = NestedTensor(
                    samples_fast.tensors[ind_start: ind_end][::args.eval_flags.vq2d.stride],
                    samples_fast.mask[ind_start: ind_end][::args.eval_flags.vq2d.stride])
            else:
                samples_window = samples_fast_window

            memory_cache_window = model(
                "vq2d",
                samples_window,
                [ind_end - ind_start],
                captions,
                encode_and_save=True,
                samples_fast=samples_fast_window,
                reference_crop=reference_crop,
            )
            outputs_window = model(
                "vq2d",
                samples_window,
                [ind_end - ind_start],
                captions,
                encode_and_save=False,
                memory_cache=memory_cache_window,
                reference_crop=reference_crop,
            )

            for id_frame in range(ind_start, ind_end):
                if len(outputs_frame_wise[id_frame]) == 0:
                    for k, v in outputs_window.items():
                        if k in ['aux_outputs', 'weights', 'ca_weights']:
                            continue
                        v_frame = v[id_frame - ind_start] if k == 'pred_boxes' else v[0, id_frame - ind_start]
                        outputs_frame_wise[id_frame][k] = [v_frame]
                else:
                    for k, v in outputs_window.items():
                        if k in ['aux_outputs', 'weights', 'ca_weights']:
                            continue
                        v_frame = v[id_frame - ind_start] if k == 'pred_boxes' else v[0, id_frame - ind_start]
                        outputs_frame_wise[id_frame][k].append(v_frame)

            if len(outputs_aggregate) == 0:
                for k, v in outputs_window.items():
                    outputs_aggregate[k] = [v]
            else:
                for k, v in outputs_window.items():
                    outputs_aggregate[k].append(v)

        # frame-wise aggregation
        for id_frame in range(durations[0]):
            for k in outputs_frame_wise[id_frame].keys():
                outputs_frame_wise[id_frame][k] = torch.stack(outputs_frame_wise[id_frame][k]).mean(0)

        outputs_frame_wise_aggregate = {}
        for id_frame in range(durations[0]):
            for k in outputs_frame_wise[id_frame].keys():
                if k not in outputs_frame_wise_aggregate:
                    outputs_frame_wise_aggregate[k] = [outputs_frame_wise[id_frame][k]]
                else:
                    outputs_frame_wise_aggregate[k].append(outputs_frame_wise[id_frame][k])

        outputs = {}
        outputs['pred_boxes'] = torch.stack(outputs_frame_wise_aggregate['pred_boxes'])
        outputs['pred_sted'] = torch.stack(outputs_frame_wise_aggregate['pred_sted']).unsqueeze(0)
        outputs['pred_score_per_frame'] = torch.stack(outputs_frame_wise_aggregate['pred_score_per_frame']).unsqueeze(0)

        # only keep box predictions in the annotated moment
        max_duration = max(durations)
        inter_idx = batch_dict["inter_idx"]
        keep_list = []
        for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
            if inter[0] >= 0:
                keep_list.extend(
                    [
                        elt
                        for elt in range(
                            i_dur * max_duration + inter[0],
                            (i_dur * max_duration) + inter[1] + 1,
                        )
                    ]
                )
        keep = torch.tensor(keep_list).long().to(outputs["pred_boxes"].device)
        pred_boxes_all = outputs["pred_boxes"]
        if args.test:
            pred_boxes_all = outputs["pred_boxes"]
            targets_all = [x for x in targets]
        outputs["pred_boxes"] = outputs["pred_boxes"][keep]
        # for i_aux in range(len(outputs["aux_outputs"])):
        #     outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux][
        #         "pred_boxes"
        #     ][keep]
        b = len(durations)
        targets = [x for x in targets if len(x["boxes"])]
        assert len(targets) == len(outputs["pred_boxes"]), (
            len(targets),
            len(outputs["pred_boxes"]),
        )
        # mask with padded positions set to False for loss computation
        if args.sted:
            time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        # update evaluator
        # if args.test:
        # outputs["pred_boxes"] = pred_boxes_all
        if args.test:
            targets = targets_all
            outputs["pred_boxes"] = pred_boxes_all
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)

        vq2d_res = {} if "vq2d" in postprocessors.keys() else None
        vq2d_video_res = {} if "vq2d" in postprocessors.keys() else None

        # if args.debug: import ipdb; ipdb.set_trace()
        if "vq2d" in postprocessors.keys():
            video_ids = batch_dict["video_ids"]
            frames_id = batch_dict["frames_id"]

            if args.sted:
                pred_steds, _ = postprocessors["vq2d"](
                    outputs, frames_id, video_ids=video_ids, time_mask=time_mask
                )

            if args.model.use_score_per_frame:
                signal = outputs['pred_score_per_frame'].sigmoid().squeeze().cpu().numpy()
                signal_raw = outputs['pred_score_per_frame'].squeeze().cpu().numpy()
                signal_sm = medfilt(signal, kernel_size=5)
                signal_raw_sm = medfilt(signal_raw, kernel_size=5)

                peaks = []

                # default criteria
                # peaks, peaks_stats = find_peaks(signal_sm, distance=35, prominence=0.3, width=3)

                # import ipdb; ipdb.set_trace()
                # based on a threshold
                if len(peaks) == 0:
                    for _thresh in [0.8, 0.7, 0.6, 0.5]:
                        peaks, peaks_stats = find_peaks(signal_sm, height=_thresh)
                        if len(peaks) > 0:
                            break

                # import ipdb; ipdb.set_trace()
                pred_steds_adjusted_by_peak = None
                if len(peaks) > 0:
                    time_mask_adjusted_by_peak = torch.full_like(time_mask, True)
                    time_mask_adjusted_by_peak[:, max(0, peaks[-1] - args.eval_flags.vq2d.win_size_around_peak): peaks[-1] + args.eval_flags.vq2d.win_size_around_peak] = False
                    pred_steds_adjusted_by_peak, _ = postprocessors["vq2d"].forward_w_pred_sted_arg(
                        outputs["pred_sted"].masked_fill(time_mask_adjusted_by_peak[..., None], -float("inf")),
                        frames_id, video_ids=video_ids, time_mask=time_mask
                    )
                    # print(f"pred_steds: {pred_steds}")
                    # print(f"pred_steds_adjusted_by_peak: {pred_steds_adjusted_by_peak}")
                    pred_steds = pred_steds_adjusted_by_peak

            image_ids = [t["image_id"] for t in targets]
            for im_id, result in zip(image_ids, results):
                vq2d_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}

            # qtypes = batch_dict["qtype"]
            # assert len(set(video_ids)) == len(qtypes)
            if args.sted:
                # assert len(pred_steds) == len(qtypes)
                for video_id, pred_sted in zip(video_ids, pred_steds):
                    vq2d_video_res[video_id] = {
                        "sted": pred_sted,
                        # "qtype": qtypes[video_id],
                    }
            else:
                pass
                # for video_id in video_ids:
                #     vq2d_video_res[video_id] = {
                #         "qtype": qtypes[video_id],
                #     }
            res = {
                target["image_id"]: output for target, output in zip(targets, results)
            }
        # else:
        #     res = {
        #         target["image_id"].item(): output
        #         for target, output in zip(targets, results)
        #     }

        # if args.debug: import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        # if args.use_vq2d_orig_metric:
        for __i, video_id in enumerate(batch_dict['video_ids']):
            # print(video_ids)
            reference_crop_annotation_video = batch_dict['reference_crop_annotation'][__i]
            response_track_annotation_video = batch_dict['response_track_annotation'][__i]
            w_video_orig = reference_crop_annotation_video['original_width']
            h_video_orig = reference_crop_annotation_video['original_height']
            scale_fct = torch.Tensor([w_video_orig, h_video_orig, w_video_orig, h_video_orig]).to(torch.int).to(device)

            # print(f"pred_steds: {pred_steds[__i]}")
            # print(f"frame ids: {frames_id[__i]}")

            # visual crop
            visual_crop_boxes_video = {
                'frame_number': reference_crop_annotation_video['frame_number'],
                'x': reference_crop_annotation_video['x'],
                'y': reference_crop_annotation_video['y'],
                'width': reference_crop_annotation_video['width'],
                'height': reference_crop_annotation_video['height'],
            }

            # predictions
            pred_response_track_video = []
            for _frame_id, _pred_box in zip(frames_id[__i], pred_boxes_all):
                if (
                    pred_steds[__i][0] <= _frame_id < pred_steds[__i][1]
                ):
                    pred_box_xyxy = box_cxcywh_to_xyxy(_pred_box) * scale_fct
                    x1, y1, x2, y2 = pred_box_xyxy.cpu().int().numpy()

                    pred_response_track_video.append({
                        'frame_number': _frame_id,
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                    })

            # ground-truth
            gt_response_track_video = []
            for _ann in response_track_annotation_video:
                gt_response_track_video.append({
                    'frame_number': _ann['frame_number'],
                    'x': _ann['x'],
                    'y': _ann['y'],
                    'width': _ann['width'],
                    'height': _ann['height'],
                })

            # import ipdb; ipdb.set_trace()
            for evaluator in evaluator_list:
                if isinstance(evaluator, VQ2DOrigEvaluator):
                    evaluator.update({
                        video_id: {
                            'pred_response_track_video': pred_response_track_video,
                            'gt_response_track_video': gt_response_track_video,
                            'visual_crop_boxes_video': visual_crop_boxes_video,
                        }
                    })
        if args.eval_flags.plot_pred:
            if i_batch > 100:
                print("WARNING WARNING WARNING WARNING STOPPING TESTING ARBITRARILY")
                break
            assert len(batch_dict['frames_id']) == 1  # only works on batch size=1

            for __i, video_id in enumerate(batch_dict['video_ids']):
                # import ipdb; ipdb.set_trace()
                p_out_video = Path(args.output_dir) / 'plot_pred' / video_id
                p_out_video.mkdir(parents=True, exist_ok=True)

                p_out_reference = p_out_video / "reference_crop.jpg"
                reference_crop_orig = batch_dict['reference_orig'][__i]

                im_reference_crop_orig = Image.fromarray(reference_crop_orig)
                im_reference_crop_orig.save(p_out_reference)

                assert len(frames_id[__i]) == len(batch_dict['images_list_pims'][__i])

                print(f"pred_steds: {pred_steds[__i]}")
                if pred_steds_adjusted_by_peak is not None:
                    print(f"pred_steds_adjusted_by_peak: {pred_steds_adjusted_by_peak[__i]}")
                print(f"num frames: {len(frames_id[__i])}\n")

                # write info to text
                p_text_file = p_out_video.parent / "info.txt"
                with p_text_file.open('a') as f:
                    f.write(f"Video: {video_id}\n")
                    f.write(f"Ground truth: [{min([int(e['image_id'].split('_')[-1]) for e in targets])}, "
                            f"{max([int(e['image_id'].split('_')[-1]) for e in targets])}]\n")
                    f.write(f"pred_steds: {pred_steds[__i]}\n")
                    if pred_steds_adjusted_by_peak is not None:
                        f.write(f"pred_steds_adjusted_by_peak: {pred_steds_adjusted_by_peak[__i]}\n")
                    f.write(f"num frames: {len(frames_id[__i])}\n")
                    f.write("\n")

                # write frames
                p_out_frames = p_out_video / "frames"
                p_out_frames.mkdir(exist_ok=True)
                for _frame_id, _image, _pred_box in zip(
                    frames_id[__i], batch_dict['images_list_pims'][__i], pred_boxes_all
                ):
                    _frame_id_str = f"{video_id}_{_frame_id}"
                    _im = Image.fromarray(_image)
                    img_w, img_h = _im.size
                    scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(torch.int).to(device)

                    _target = None
                    for e in targets:
                        if e['image_id'] == _frame_id_str:
                            _target = e
                            break

                    fig, ax = plt.subplots()
                    ax.axis("off")
                    ax.imshow(_im, aspect="auto")

                    if _target is not None:
                        gt_box_xyxy = box_cxcywh_to_xyxy(_target['boxes'][0]) * scale_fct
                        x1, y1, x2, y2 = gt_box_xyxy.cpu().int().numpy()
                        w = x2 - x1
                        h = y2 - y1
                        rect = plt.Rectangle(
                            (x1, y1), w, h, linewidth=2, edgecolor="#00FF00", fill=False  # green
                        )
                        ax.add_patch(rect)

                    if (
                        pred_steds[__i][0] <= _frame_id < pred_steds[__i][1]
                    ):
                        pred_box_xyxy = box_cxcywh_to_xyxy(_pred_box) * scale_fct
                        x1, y1, x2, y2 = pred_box_xyxy.cpu().int().numpy()
                        w = x2 - x1
                        h = y2 - y1
                        rect = plt.Rectangle(
                            (x1, y1), w, h, linewidth=2, edgecolor="#FAFF00", fill=False  # yellowish
                        )
                        ax.add_patch(rect)

                    if pred_steds_adjusted_by_peak is not None:
                        if (
                            pred_steds_adjusted_by_peak[__i][0] <= _frame_id < pred_steds_adjusted_by_peak[__i][1]
                        ):
                            pred_box_xyxy = box_cxcywh_to_xyxy(_pred_box) * scale_fct
                            x1, y1, x2, y2 = pred_box_xyxy.cpu().int().numpy()
                            w = x2 - x1
                            h = y2 - y1
                            rect = plt.Rectangle(
                                (x1, y1), w, h, linewidth=2, edgecolor="#0000FF", fill=False  # blue
                            )
                            ax.add_patch(rect)

                    # place a text box in upper left in axes coords
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.05, 0.95, f"f_hat: {signal[frames_id[__i].index(_frame_id)]:.4f}",
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)

                    fig.set_dpi(100)
                    fig.set_size_inches(img_w / 100, img_h / 100)
                    fig.tight_layout(pad=0)

                    # save image with eventual box
                    fig.savefig(
                        p_out_frames / f"{video_id}_{_frame_id:04d}.jpg",
                        format="jpg",
                    )
                    plt.close(fig)

                # make score plot
                data_score_plot = {'frame_id': [], 'target': []}
                for _frame_id in frames_id[__i]:
                    _frame_id_str = f"{video_id}_{_frame_id}"
                    data_score_plot['frame_id'].append(_frame_id)
                    flag_target = False
                    for e in targets:
                        if e['image_id'] == _frame_id_str:
                            flag_target = True
                    if flag_target:
                        data_score_plot['target'].append(1)
                    else:
                        data_score_plot['target'].append(0)

                data_score_plot['signal_raw'] = signal_raw
                data_score_plot['signal'] = signal
                data_score_plot['signal_sm'] = signal_sm

                sns.set_theme()
                sns.set(style="white", font_scale=1.5)
                sns.set_palette("crest")

                plt.rcParams["figure.figsize"] = (15, 12)

                fig, axs = plt.subplots(1, 1)

                axs.plot(data_score_plot['frame_id'], data_score_plot['target'], label='Ground truth', linewidth=4, color='teal')
                axs.plot(data_score_plot['frame_id'], data_score_plot['signal'], label='Score', linewidth=3, color='orange')
                axs.set_ylabel('Score')
                axs.set_xlabel('Frame ids')
                axs.legend()

                fig.tight_layout()
                plt.savefig(p_out_video / "score_target_plot.png", format="png")
                plt.close()

                # import ipdb; ipdb.set_trace()
                # stitch frames to video
                os.system(
                    f"ffmpeg -loglevel error -framerate 3 -pattern_type glob -i '{str(p_out_frames)}/*.jpg' -c:v libx264 -pix_fmt yuv420p {str(p_out_video)}/out.mp4"
                )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    # vidstg_res = None
    vq2d_res = None
    # hcstvg_res = None
    vq2d_res_orig = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, VQ2DOrigEvaluator):
            vq2d_res_orig = evaluator.summarize()

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # if vidstg_res is not None:
    #     stats["vidstg"] = vidstg_res

    # if vq2d_res is not None:
    #     stats["vq2d_res"] = vq2d_res

    if vq2d_res_orig is not None:
        stats["vq2d_res_orig"] = vq2d_res_orig

    # if hcstvg_res is not None:
    #     stats["hcstvg"] = hcstvg_res

    return stats
