# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import shutil
import math
import sys
import copy
from typing import Dict, Iterable, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn
import torch.optim

import util.dist as dist
from datasets.nlq_orig_eval import NLQOrigEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema
from scipy.signal import find_peaks, medfilt
from util.misc import targets_to, NestedTensor
from datasets.mq_orig_eval import nms
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from pathlib import Path
from PIL import Image


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    postprocessors: Dict[str, torch.nn.Module],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    """
    NLQ evaluator format (list of dicts w/ length = 3874):
        {'annotation_uid': 'ca7e11a2-cd1e-40dd-9d2f-ea810ab6a99b',
        'clip_uid': '93231c7e-1cf4-4a20-b1f8-9cc9428915b2',
        'predicted_times': [[0.0, 3.75],
                            [0.0, 480.0],
                            [0.0, 18.75],
                            [0.0, 15.0],
                            [0.0, 11.25]],
        'query_idx': 0}
    """
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.eval_flags.print_freq, header)
    ):
        # samples = batch_dict["samples"].to(device)
        # if "samples_fast" in batch_dict:
        #     samples_fast = batch_dict["samples_fast"].to(device)
        # else:
        #     samples_fast = None
        durations = batch_dict["durations"]
        captions = batch_dict["captions"]
        annotation_uid = batch_dict["annotation_uid"]
        query_idx = batch_dict["query_idx"]
        targets = batch_dict["targets"]
        targets = targets_to(targets, device)
        kwargs = {}
        if args.model.nlq.use_sentence_text_embeddings:
            kwargs["key_for_nlq_sentence_embedding"] = (
                batch_dict["video_ids"][0], batch_dict["annotation_uid"][0], batch_dict["query_idx"][0]
            )

        targets_non_empty_boxes = [x for x in targets if len(x["boxes"])]
        if len(targets_non_empty_boxes) == 0:
            continue

        pred_steds_sorted_all_scales = []
        pred_steds_sorted_seconds_all_scales = []
        pred_scores_sorted_all_scales = []

        # forward
        for scale_prediction in args.eval_flags.nlq.scale_prediction:
            if "samples_fast" in batch_dict:
                samples_fast = copy.deepcopy(batch_dict["samples_fast"]).to(device)
            elif not args.eval_flags.nlq.stride:
                samples_fast = copy.deepcopy(batch_dict["samples"]).to(device)
            else:
                raise ValueError("asdasd")

            # MOD SCALE
            indices_frames_to_subsample = [*range(0, samples_fast.tensors.shape[0], scale_prediction)]

            samples_fast.tensors = samples_fast.tensors[indices_frames_to_subsample]
            samples_fast.mask = samples_fast.mask[indices_frames_to_subsample]

            durations = [len(indices_frames_to_subsample)]

            assert len(durations) == 1  # works only on batch size 1
            outputs_aggregate = {}
            outputs_frame_wise = {__i: {} for __i in range(durations[0])}
            window_step_size = min(args.eval_flags.nlq.window_step_size, args.model.nlq.video_max_len)
            for ind_start in range(0, durations[0], window_step_size):
                ind_end = min(durations[0], ind_start + args.model.nlq.video_max_len)
                # print(ind_start, ind_end)

                samples_fast_window = NestedTensor(
                    samples_fast.tensors[ind_start: ind_end], samples_fast.mask[ind_start: ind_end])

                if args.eval_flags.nlq.stride:
                    samples_window = NestedTensor(
                        samples_fast.tensors[ind_start: ind_end][::args.eval_flags.nlq.stride],
                        samples_fast.mask[ind_start: ind_end][::args.eval_flags.nlq.stride])
                else:
                    samples_window = samples_fast_window

                memory_cache_window = model(
                    "nlq",
                    samples_window,
                    [ind_end - ind_start],
                    captions,
                    encode_and_save=True,
                    samples_fast=samples_fast_window,
                    **kwargs
                )
                outputs_window = model(
                    "nlq",
                    samples_window,
                    [ind_end - ind_start],
                    captions,
                    encode_and_save=False,
                    memory_cache=memory_cache_window,
                    **kwargs
                )
                for id_frame in range(ind_start, ind_end):
                    if len(outputs_frame_wise[id_frame]) == 0:
                        for k, v in outputs_window.items():
                            if k in ['aux_outputs', 'ca_weights', 'weights']:
                                continue
                            v_frame = v[id_frame - ind_start] if k == 'pred_boxes' else v[0, id_frame - ind_start]
                            outputs_frame_wise[id_frame][k] = [v_frame]
                    else:
                        for k, v in outputs_window.items():
                            if k in ['aux_outputs', 'ca_weights', 'weights']:
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

            # mask with padded positions set to False for loss computation
            if args.sted:
                time_mask = torch.zeros(len(durations), outputs["pred_sted"].shape[1]).bool().to(device)
                for i_dur, duration in enumerate(durations):
                    time_mask[i_dur, :duration] = True
            else:
                time_mask = None

            if "nlq" in postprocessors.keys():
                video_ids = batch_dict["video_ids"]
                frames_id = batch_dict["frames_id"]

                if args.model.use_score_per_frame:
                    signal = outputs['pred_score_per_frame'].sigmoid().squeeze().cpu().numpy()
                    # signal_raw = outputs['pred_score_per_frame'].squeeze().cpu().numpy()
                    signal_sm = medfilt(signal, kernel_size=5)

                    # default criteria
                    peaks_fp, peaks_stats_fp = find_peaks(signal_sm, distance=35, prominence=0.3, width=3)

                    # based on a threshold
                    for _thresh in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
                        peaks, peaks_stats = find_peaks(signal_sm, height=_thresh)
                        if len(peaks) > 1000:
                            break

                    # import ipdb; ipdb.set_trace()
                    pred_steds_adjusted_by_peak = None
                    if len(peaks) > 0:
                        peak_heights_tensor = torch.Tensor(peaks_stats['peak_heights'])
                        ind_sorted_desc = peak_heights_tensor.argsort(descending=True)

                        peak_heights_sorted = peak_heights_tensor[ind_sorted_desc]
                        peaks_sorted = peaks[ind_sorted_desc]

                        # top_5_peaks = peaks[torch.Tensor(peaks_stats['peak_heights']).topk(k=min(5, len(peaks))).indices]
                        if len(peaks) == 1:
                            peaks_sorted = peaks  # otherwise it will be a scalar entry

                        # consider predictions from find_peaks function
                        # if len(peaks_fp) > 0:
                        #     top_5_peaks = peaks_fp[torch.Tensor(peaks_stats_fp['prominences']).topk(k=min(5, len(peaks_fp))).indices]
                        #     if len(peaks_fp) == 1:
                        #         top_5_peaks = peaks_fp  # otherwise it will be a scalar entry

                        pred_steds_sorted = []
                        pred_scores_sorted = []
                        for _peak, _peak_height in zip(peaks_sorted, peak_heights_sorted):
                            for win_size in args.eval_flags.nlq.win_size_around_peak:
                                time_mask_adjusted_by_peak = torch.full_like(time_mask, True)
                                time_mask_adjusted_by_peak[:, max(0, _peak - win_size): _peak + win_size] = False
                                pred_steds_adjusted_by_peak, score_pred_sted = postprocessors["nlq"].forward_w_pred_sted_arg(
                                    outputs["pred_sted"].masked_fill(time_mask_adjusted_by_peak[..., None], -float("inf")),
                                    frames_id, video_ids=video_ids, time_mask=time_mask
                                )
                                pred_steds_sorted.append(pred_steds_adjusted_by_peak[0])
                                pred_scores_sorted.append(_peak_height)

            # generate predictions in the format of NLQ
            fps_for_tubedetr = 5 / scale_prediction
            pred_steds_sorted_seconds = np.array(
                [[e[0] / fps_for_tubedetr, e[1] / fps_for_tubedetr] for e in pred_steds_sorted]
            )
            pred_steds_sorted_all_scales.append(pred_steds_sorted)
            pred_steds_sorted_seconds_all_scales.append(pred_steds_sorted_seconds)
            pred_scores_sorted_all_scales.append(np.asarray(pred_scores_sorted))

        # assemble preds from different scales
        pred_steds_sorted_all_scales = np.concatenate(pred_steds_sorted_all_scales)
        pred_steds_sorted_seconds_all_scales = np.concatenate(pred_steds_sorted_seconds_all_scales)
        pred_scores_sorted_all_scales = np.concatenate(pred_scores_sorted_all_scales)

        # sort them
        sort_order_across_scales = pred_scores_sorted_all_scales.argsort()[::-1]
        pred_steds_sorted_seconds_all_scales = pred_steds_sorted_seconds_all_scales[sort_order_across_scales]
        pred_scores_sorted_all_scales = pred_scores_sorted_all_scales[sort_order_across_scales]
        pred_steds_sorted_all_scales = pred_steds_sorted_all_scales[sort_order_across_scales]

        # nms
        keep = nms(np.concatenate((pred_steds_sorted_seconds_all_scales, pred_scores_sorted_all_scales[:, None]), axis=-1))
        pred_steds_sorted_seconds_all_scales = pred_steds_sorted_seconds_all_scales[keep]
        pred_steds_sorted_all_scales = pred_steds_sorted_all_scales[keep]

        for evaluator in evaluator_list:
            if isinstance(evaluator, NLQOrigEvaluator):
                key_for_sample = (video_ids[0], annotation_uid[0], query_idx[0])
                evaluator.update({key_for_sample: {
                    "clip_uid": video_ids[0],
                    "annotation_uid": annotation_uid[0],
                    "query_idx": query_idx[0],
                    "predicted_times": copy.deepcopy(pred_steds_sorted_seconds_all_scales.tolist())
                }})

        if args.eval_flags.plot_pred:
            if i_batch > 30:
                print("WARNING WARNING WARNING WARNING STOPPING TESTING ARBITRARILY")
                break
                asdas
            assert len(video_ids) == 1

            # save video frames so that it can be reused by multiple annotations            
            p_out_video_bank = Path(args.output_dir) / 'plot_pred' / f"{video_ids[0]}" / "frames"
            if not p_out_video_bank.is_dir():
                print("Writing video bank frames")
                p_out_video_bank.mkdir(parents=True, exist_ok=True)
                for _frame_id, _image in tqdm(zip(
                    frames_id[0], batch_dict['images_list_pims'][0]
                ), total=len(frames_id[0])):
                    _im = Image.fromarray(_image)
                    img_w, img_h = _im.size

                    fig, ax = plt.subplots()
                    ax.axis("off")
                    ax.imshow(_im, aspect="auto")

                    fig.set_dpi(100)
                    fig.set_size_inches(img_w / 100, img_h / 100)
                    fig.tight_layout(pad=0)

                    # save image with eventual box
                    fig.savefig(
                        p_out_video_bank / f"{_frame_id:04d}.jpg",
                        format="jpg",
                    )
                    plt.close(fig)

            p_out_video = Path(args.output_dir) / 'plot_pred' / f"{video_ids[0]}_{annotation_uid[0]}_{query_idx[0]}"
            p_out_video.mkdir(parents=True, exist_ok=True)

            assert len(frames_id[0]) == len(batch_dict['images_list_pims'][0])

            gt_extent_img_ids = [
                min([int(e['image_id'].split('_')[-1]) for e in targets_non_empty_boxes]),
                max([int(e['image_id'].split('_')[-1]) for e in targets_non_empty_boxes])
            ]

            pred_steds_top5_mm_ss_format = [
                [e[0] // 60 + e[0] % 60 / 100, e[1] // 60 + e[1] % 60 / 100]
                for e in pred_steds_sorted_seconds_all_scales
            ]

            print(f"pred_steds_top_5: {pred_steds_sorted_all_scales}")
            print(f"pred_steds_top5_mm_ss_format: {pred_steds_top5_mm_ss_format}")
            print(f"GT steds: {gt_extent_img_ids}")
            print(f"Caption: {captions[0]}")
            print(f"num frames: {len(frames_id[0])}\n")

            # write info to text
            p_text_file = p_out_video.parent / "info.txt"
            with p_text_file.open('a') as f:
                f.write(f"Video: {video_ids[0]}\n")
                f.write(f"Annotation id: {annotation_uid[0]}\n")
                f.write(f"Query id: {query_idx[0]}\n")
                f.write(f"Caption: {captions[0]}\n")
                f.write(f"Ground truth: [{gt_extent_img_ids[0]}, {gt_extent_img_ids[1]}]\n")
                f.write(f"pred_steds_top_5: {pred_steds_sorted_all_scales}\n")
                f.write(f"pred_steds_top_5_mm_ss: {pred_steds_top5_mm_ss_format}\n")
                f.write(f"num frames: {len(frames_id[0])}\n")
                f.write("\n")

            # write frames
            p_out_frames = p_out_video / "frames"
            p_out_frames.mkdir(exist_ok=True)

            shutil.copytree(p_out_video_bank, p_out_frames, dirs_exist_ok=True)
            # """
            for _frame_id, _image, _pred_box in tqdm(zip(
                frames_id[0], batch_dict['images_list_pims'][0], outputs['pred_boxes']
            ), total=len(frames_id[0])):
                flag_relevant_frame = False
                if (_frame_id >= gt_extent_img_ids[0] and _frame_id <= gt_extent_img_ids[1]):
                    flag_relevant_frame = True

                for __k, __sted in enumerate(pred_steds_sorted_all_scales):
                    if _frame_id >= __sted[0] and _frame_id <= __sted[1]:
                        flag_relevant_frame = True

                if not flag_relevant_frame:
                    continue

                _im = Image.fromarray(_image)
                img_w, img_h = _im.size
                scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(torch.int).to(device)

                fig, ax = plt.subplots()
                ax.axis("off")
                ax.imshow(_im, aspect="auto")

                if _frame_id >= gt_extent_img_ids[0] and _frame_id <= gt_extent_img_ids[1]:
                    props = dict(boxstyle='round', facecolor='green', alpha=0.5)
                    ax.text(0.05, 0.95, "GT",
                            transform=ax.transAxes, fontsize=15,
                            verticalalignment='top', bbox=props)

                for __k, __sted in enumerate(pred_steds_sorted_all_scales):
                    if _frame_id >= __sted[0] and _frame_id <= __sted[1]:
                        pred_box_xyxy = box_cxcywh_to_xyxy(_pred_box) * scale_fct
                        x1, y1, x2, y2 = pred_box_xyxy.cpu().int().numpy()
                        w = x2 - x1
                        h = y2 - y1
                        rect = plt.Rectangle(
                            (x1, y1), w, h, linewidth=2, edgecolor="#FAFF00", fill=False  # yellowish
                        )
                        ax.add_patch(rect)

                        props = dict(boxstyle='round', facecolor='blue', alpha=0.5)
                        ax.text(0.05, 0.95, f"{__k + 1}th",
                                transform=ax.transAxes, fontsize=15,
                                verticalalignment='top', bbox=props)

                fig.set_dpi(100)
                fig.set_size_inches(img_w / 100, img_h / 100)
                fig.tight_layout(pad=0)

                # save image with eventual box
                fig.savefig(
                    p_out_frames / f"{_frame_id:04d}.jpg",
                    format="jpg",
                )
                plt.close(fig)
            # """
            # if args.debug: import ipdb; ipdb.set_trace()

            # make score plot
            data_score_plot = {'frame_id': [], 'target': []}
            data_score_plot.update({f"top_{__i + 1}": [] for __i in range(len(pred_steds_sorted_all_scales))})

            for _frame_id in frames_id[0]:
                data_score_plot['frame_id'].append(_frame_id)
                if _frame_id >= gt_extent_img_ids[0] and _frame_id <= gt_extent_img_ids[1]:
                    data_score_plot['target'].append(1)
                else:
                    data_score_plot['target'].append(0)

                for __i, __e in enumerate(pred_steds_sorted_all_scales):
                    if _frame_id >= __e[0] and _frame_id <= __e[1]:
                        data_score_plot[f"top_{__i + 1}"].append(1)
                    else:
                        data_score_plot[f"top_{__i + 1}"].append(0)

            # data_score_plot['signal_raw'] = signal_raw
            # data_score_plot['signal'] = signal
            data_score_plot['signal_sm'] = signal_sm

            # plt.rcParams["figure.figsize"] = (10, 10)
            # plt.plot(data_score_plot['frame_id'], data_score_plot['target'], label='Ground truth')
            # plt.plot(data_score_plot['frame_id'], data_score_plot['signal_sm'], label='Score')
            # for __i in range(len(pred_steds_top_5)):
            #     plt.plot(data_score_plot['frame_id'], data_score_plot[f"top_{__i + 1}"], label=f"top_{__i + 1}")
            # plt.legend()

            # plt.savefig(p_out_video / "score_target_plot.png", format="png")
            # plt.close()
            sns.set_theme()
            sns.set(style="white", font_scale=1.5)
            sns.set_palette("crest")

            plt.rcParams["figure.figsize"] = (15, 12)

            fig, axs = plt.subplots(2, 1)
            axs[0].plot(data_score_plot['frame_id'], data_score_plot['target'], label='Ground truth', linewidth=4, color='teal')
            axs[0].plot(data_score_plot['frame_id'], data_score_plot['signal_sm'], label='Score', linewidth=3, color='orange')
            axs[0].set_ylabel('Score')
            axs[0].set_title(captions[0])
            axs[0].legend()

            for __i in range(len(pred_steds_sorted_all_scales)):
                axs[1].plot(data_score_plot['frame_id'], data_score_plot[f"top_{__i + 1}"], label=f"top_{__i + 1}", linewidth=4)

            axs[1].annotate(
                'top 1',
                xy=(int(np.median(np.nonzero(np.asarray(data_score_plot['top_1']) == 1)[0])), 1),
                xytext=(int(np.median(np.nonzero(np.asarray(data_score_plot['top_1']) == 1)[0])), 1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=7)
            )
            axs[1].set_xlabel('Frame ids')
            axs[1].set_ylabel('Score')
            axs[1].legend()

            fig.tight_layout()
            plt.savefig(p_out_video / "score_target_plot.png", format="png")
            plt.close()

            torch.save(data_score_plot, p_out_video / 'data_score_plot.pth')

            # stitch frames to video
            os.system(
                f"ffmpeg -y -loglevel error -framerate 5 -pattern_type glob -i '{str(p_out_frames)}/*.jpg' -c:v libx264 -pix_fmt yuv420p {str(p_out_video)}/out.mp4"
            )
            # import ipdb; ipdb.set_trace()

    # if args.debug: import ipdb; ipdb.set_trace()
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    nlq_res_orig = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, NLQOrigEvaluator):
            nlq_res_orig = evaluator.summarize()

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if nlq_res_orig is not None:
        stats["nlq_res_orig"] = nlq_res_orig

    return stats
