# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import json
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
from datasets.mq_orig_eval import MQOrigEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from scipy.signal import find_peaks, medfilt
from util.misc import targets_to, NestedTensor

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
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    class_to_idx_file = Path(args.data.mq.path) / "moment_classes_idx.json"
    with class_to_idx_file.open("rb") as fp:
        class_to_idx = json.load(fp)

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        if args.eval_flags.mq.generate_and_save_preds_for_all_captions:
            captions_list = [k for k, v in class_to_idx.items() if k != "Background"]
            raise NotImplementedError(
                "The implementation is broken since the introduction of scale")
        else:
            captions_list = batch_dict["captions"]

        pred_steds_sorted_seconds_all_scales = []
        pred_scores_sorted_all_scales = []

        for scale_prediction in args.eval_flags.mq.scale_prediction:
            if "samples_fast" in batch_dict:
                samples_fast = copy.deepcopy(batch_dict["samples_fast"]).to(device)
            elif not args.eval_flags.mq.stride:
                samples_fast = copy.deepcopy(batch_dict["samples"]).to(device)
            else:
                raise ValueError()

            durations = batch_dict["durations"]
            # captions = batch_dict["captions"]

            # MOD SCALE
            indices_frames_to_subsample = [*range(0, samples_fast.tensors.shape[0], scale_prediction)]

            samples_fast.tensors = samples_fast.tensors[indices_frames_to_subsample]
            samples_fast.mask = samples_fast.mask[indices_frames_to_subsample]

            durations = [len(indices_frames_to_subsample)]

            for __caption in captions_list:
                # print(batch_dict['video_ids'], __caption)
                captions = [__caption]
                captions_idx = [class_to_idx[__caption]]
                query_idx = batch_dict["query_idx"]
                targets = batch_dict["targets"]

                targets = targets_to(targets, device)

                targets_non_empty_boxes = [x for x in targets if len(x["boxes"])]
                if len(targets_non_empty_boxes) == 0:
                    continue

                # forward
                assert len(durations) == 1  # works only on batch size 1
                outputs_aggregate = {}
                outputs_frame_wise = {__i: {} for __i in range(durations[0])}
                window_step_size = min(args.eval_flags.mq.window_step_size, args.model.mq.video_max_len)
                for ind_start in range(0, durations[0], window_step_size):
                    ind_end = min(durations[0], ind_start + args.model.mq.video_max_len)
                    # print(ind_start, ind_end)

                    samples_fast_window = NestedTensor(
                        samples_fast.tensors[ind_start: ind_end], samples_fast.mask[ind_start: ind_end])

                    if args.eval_flags.mq.stride:
                        samples_window = NestedTensor(
                            samples_fast.tensors[ind_start: ind_end][::args.eval_flags.mq.stride],
                            samples_fast.mask[ind_start: ind_end][::args.eval_flags.mq.stride])
                    else:
                        samples_window = samples_fast_window

                    memory_cache_window = model(
                        "mq",
                        samples_window,
                        [ind_end - ind_start],
                        captions,
                        encode_and_save=True,
                        samples_fast=samples_fast_window,
                    )
                    outputs_window = model(
                        "mq",
                        samples_window,
                        [ind_end - ind_start],
                        captions,
                        encode_and_save=False,
                        memory_cache=memory_cache_window,
                    )
                    for id_frame in range(ind_start, ind_end):
                        if len(outputs_frame_wise[id_frame]) == 0:
                            for k, v in outputs_window.items():
                                if k in ['aux_outputs', 'ca_weights', 'weights']:
                                    continue
                                if k == 'pred_boxes':
                                    v_frame = v[id_frame - ind_start]
                                elif k == 'segment_type' and args.model.mq.use_segment_type_classification:
                                    v_frame = v[0]
                                else:
                                    v_frame = v[0, id_frame - ind_start]
                                # v_frame = v[id_frame - ind_start] if k == 'pred_boxes' else v[0, id_frame - ind_start]
                                outputs_frame_wise[id_frame][k] = [v_frame]
                        else:
                            for k, v in outputs_window.items():
                                if k in ['aux_outputs', 'ca_weights', 'weights']:
                                    continue
                                if k == 'pred_boxes':
                                    v_frame = v[id_frame - ind_start]
                                elif k == 'segment_type' and args.model.mq.use_segment_type_classification:
                                    v_frame = v[0]
                                else:
                                    v_frame = v[0, id_frame - ind_start]
                                # v_frame = v[id_frame - ind_start] if k == 'pred_boxes' else v[0, id_frame - ind_start]
                                outputs_frame_wise[id_frame][k].append(v_frame)
                    if len(outputs_aggregate) == 0:
                        for k, v in outputs_window.items():
                            outputs_aggregate[k] = [v]
                    else:
                        for k, v in outputs_window.items():
                            outputs_aggregate[k].append(v)

                ################################################################
                ############ UNDERSTANDING SEGMENT CLASSIFICATION ##############
                # import ipdb; ipdb.set_trace()
                if args.model.mq.use_segment_type_classification:

                    # pred_score_avged
                    frame_ids = []
                    pred_score_avged = []

                    for _frame_id, v in outputs_frame_wise.items():
                        frame_ids.append(_frame_id)
                        pred_score_avged.append(torch.stack(v['pred_score_per_frame']).mean().sigmoid().item())

                    # votes_pecentage, votes_score
                    votes_pecentage = []
                    votes_score = []
                    for _frame_id, v in outputs_frame_wise.items():
                        prob_argmax, indices_argmax = torch.stack(v['segment_type']).softmax(-1).max(-1)
                        votes_pecentage.append((indices_argmax != 3).float().mean().item())

                        if (indices_argmax != 3).any().item():
                            votes_score.append((prob_argmax * (indices_argmax != 3).float()).mean().item())
                        else:
                            votes_score.append(0)

                        # v['pred_sted_modulated'] = torch.stack(v['pred_sted']) * (prob_argmax * (indices_argmax != 3).float())[..., None]

                    votes_score = np.asarray(votes_score)
                    votes_score_div_max = votes_score / votes_score.max()
                    # pred_score_mul_votes_score
                    pred_score_mul_votes_score = np.asarray([e1*e2 for e1, e2 in zip(votes_score_div_max, pred_score_avged)])
                    # pred_score_mul_votes_score = pred_score_mul_votes_score / pred_score_mul_votes_score.max()

                    # pred_sted_modulated


                # import ipdb; ipdb.set_trace()


                """
                # obtain all GT segments
                annos_gt = [
                    e for e in data_loader.dataset.data_json[batch_dict['video_ids'][0]]['annotations']
                    if e['label'] == captions[0]
                ]
                gt_segments = {__e: [] for __e in range(len(frame_ids))}
                for __i, __ann in enumerate(annos_gt):
                    for __e in range(int(__ann['start_time'] * 5), int(__ann['end_time'] * 5)):
                        gt_segments[__e].append(__i)

                segment_types = ["fg", "left_trun", "right_trun", "bg"]
                for k, v in outputs_frame_wise.items():
                    print(f"{k} | GT: {gt_segments[k]}"
                        f" | {[segment_types[e.argmax().item()] for e in v['segment_type']]}"
                        f" | {torch.stack([e.softmax(-1)[e.argmax().item()] for e in v['segment_type']]).cpu()}"
                        f" | fg/bg: {torch.stack(v['pred_score_per_frame']).sigmoid().flatten().cpu()}")
                import ipdb; ipdb.set_trace()
                continue

                # making plot
                p_plot_pred = Path(args.output_dir) / 'plot_pred'
                p_plot_pred.mkdir(parents=True, exist_ok=True)

                # pred_score_avged
                frame_ids = []
                pred_score_avged = []

                for _frame_id, v in outputs_frame_wise.items():
                    frame_ids.append(_frame_id)
                    pred_score_avged.append(torch.stack(v['pred_score_per_frame']).mean().sigmoid().item())

                # gt
                gt_segments = {__i: [0] * len(frame_ids) for __i in range(len(annos_gt))}
                for __i, __ann in enumerate(annos_gt):
                    for __e in range(int(__ann['start_time'] * 5), min(int(__ann['end_time'] * 5), len(frame_ids))):
                        gt_segments[__i][__e] = 1

                # votes_pecentage, votes_score
                votes_pecentage = []
                votes_score = []
                for _frame_id, v in outputs_frame_wise.items():
                    prob_argmax, indices_argmax = torch.stack(v['segment_type']).softmax(-1).max(-1)
                    votes_pecentage.append((indices_argmax != 3).float().mean().item())

                    if (indices_argmax != 3).any().item():
                        votes_score.append((prob_argmax * (indices_argmax != 3).float()).mean().item())
                    else:
                        votes_score.append(0)

                # pred_score_mul_votes_score
                pred_score_mul_votes_score = [e1*e2 for e1, e2 in zip(votes_score, pred_score_avged)]

                sns.set_theme()
                sns.set(style="white", font_scale=1.5)
                sns.set_palette("crest", 5)

                plt.rcParams["figure.figsize"] = (15, 23)

                fig, axs = plt.subplots(4, 1)

                for __k, __gt in gt_segments.items():
                    axs[0].plot(frame_ids, __gt, label=f"GT_{__k}", linewidth=4, color='teal')
                axs[0].plot(frame_ids, pred_score_avged, label='Score', linewidth=3, color='orange')
                # axs[0].set_xlabel('frame id')
                axs[0].set_ylabel('Mean fg/bg score')
                axs[0].set_title('Mean fg/bg score with GTs')
                axs[0].legend()


                axs[1].plot(frame_ids, votes_pecentage, label='Votes (%)', linewidth=3, color='orange')
                # axs[0].set_xlabel('frame id')
                axs[1].set_ylabel('Votes (%)')
                axs[1].set_title('Votes (%) from segment classification')
                axs[1].legend()

                axs[2].plot(frame_ids, votes_score, label='Votes (mean prob)', linewidth=3, color='orange')
                # axs[0].set_xlabel('frame id')
                axs[2].set_ylabel('Votes (mean prob)')
                axs[2].set_title('Votes (mean prob) from segment classification')
                axs[2].legend()

                axs[3].plot(frame_ids, pred_score_mul_votes_score, label='Votes (mean prob) x fg/bg score', linewidth=3, color='orange')
                # axs[0].set_xlabel('frame id')
                axs[3].set_ylabel('Votes (mean prob) x fg/bg score')
                axs[3].set_title('Votes (mean prob) x fg/bg score')
                axs[3].legend()

                fig.tight_layout()
                plt.savefig(p_plot_pred / f"{batch_dict['video_ids'][0]}_{captions_idx[0]}.png", format="png")
                plt.close()

                # import ipdb; ipdb.set_trace()

                continue
                """
                ################################################################

                # frame-wise aggregation
                outputs_frame_wise_simple_averaging = {}
                for id_frame in range(durations[0]):
                    outputs_frame_wise_simple_averaging[id_frame] = {}
                    for k in outputs_frame_wise[id_frame].keys():
                        outputs_frame_wise_simple_averaging[id_frame][k] = torch.stack(outputs_frame_wise[id_frame][k]).mean(0)

                outputs_frame_wise_aggregate = {}
                for id_frame in range(durations[0]):
                    for k in outputs_frame_wise[id_frame].keys():
                        if k not in outputs_frame_wise_aggregate:
                            outputs_frame_wise_aggregate[k] = [outputs_frame_wise_simple_averaging[id_frame][k]]
                        else:
                            outputs_frame_wise_aggregate[k].append(outputs_frame_wise_simple_averaging[id_frame][k])

                outputs = {}
                outputs['pred_boxes'] = torch.stack(outputs_frame_wise_aggregate['pred_boxes'])
                outputs['pred_sted'] = torch.stack(outputs_frame_wise_aggregate['pred_sted']).unsqueeze(0)
                outputs['pred_score_per_frame'] = torch.stack(outputs_frame_wise_aggregate['pred_score_per_frame']).unsqueeze(0)

                if args.model.mq.use_segment_type_classification:
                    pred_sted_segment_aggregated = []
                    for _frame_id, v in outputs_frame_wise.items():
                        prob_argmax, indices_argmax = torch.stack(v['segment_type']).softmax(-1).max(-1)
                        pred_sted_all_segments = torch.stack(v['pred_sted'])

                        pred_sted_aggregate = []
                        for __id_segment in range(len(v['segment_type'])):
                            if indices_argmax[__id_segment] == 0:
                                pred_sted_aggregate.append(prob_argmax[__id_segment] * pred_sted_all_segments[__id_segment])
                            elif indices_argmax[__id_segment] == 1:  # left_trun
                                pred_sted_left_trun = pred_sted_all_segments[__id_segment]
                                pred_sted_left_trun[0] = 0
                                pred_sted_aggregate.append(prob_argmax[__id_segment] * pred_sted_left_trun)
                            elif indices_argmax[__id_segment] == 2:  # right_trun
                                pred_sted_left_trun = pred_sted_all_segments[__id_segment]
                                pred_sted_left_trun[1] = 0
                                pred_sted_aggregate.append(prob_argmax[__id_segment] * pred_sted_left_trun)

                        if len(pred_sted_aggregate) == 0:
                            pred_sted_aggregate = torch.ones_like(v['pred_sted'][0]) * -float("inf")
                        else:
                            pred_sted_aggregate = torch.stack(pred_sted_aggregate).mean(0)
                            pred_sted_aggregate = pred_sted_aggregate.masked_fill(pred_sted_aggregate == 0, -float("inf"))

                        pred_sted_segment_aggregated.append(pred_sted_aggregate)

                    pred_sted_segment_aggregated = torch.stack(pred_sted_segment_aggregated)[None, ...]

                if args.sted:
                    time_mask = torch.zeros(len(durations), outputs["pred_sted"].shape[1]).bool().to(device)
                    for i_dur, duration in enumerate(durations):
                        time_mask[i_dur, :duration] = True
                else:
                    time_mask = None

                if "mq" in postprocessors.keys():
                    video_ids = batch_dict["video_ids"]
                    frames_id = batch_dict["frames_id"]

                    if args.model.use_score_per_frame:
                        signal = outputs['pred_score_per_frame'].sigmoid().squeeze().cpu().numpy()
                        if args.model.mq.use_segment_type_classification:
                            signal_mul_vote_score = medfilt(pred_score_mul_votes_score, kernel_size=5)
                            # signal = np.asarray(pred_score_mul_votes_score)
                            # signal = np.asarray(votes_score)
                        # signal_raw = outputs['pred_score_per_frame'].squeeze().cpu().numpy()
                        signal_sm = medfilt(signal, kernel_size=5)

                        # default criteria
                        peaks_fp, peaks_stats_fp = find_peaks(signal_sm, distance=35, prominence=0.3, width=3)

                        # import ipdb; ipdb.set_trace()
                        # based on a threshold
                        for _thresh in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                            peaks, peaks_stats = find_peaks(signal_sm, height=_thresh)
                            if len(peaks) > 1000:
                                break

                        """
                        if args.use_segment_type_classification:
                            peaks_mul_vote_score, peaks_stats_mul_vote_score = find_peaks(signal_mul_vote_score, height=0.1)
                            # peaks, peaks_stats = find_peaks(signal_mul_vote_score, height=0.1)

                            peaks = np.concatenate((peaks, peaks_mul_vote_score))
                            peaks_stats['peak_heights'] = np.concatenate(
                                (peaks_stats['peak_heights'], peaks_stats_mul_vote_score['peak_heights'])
                            )
                        """

                        # import ipdb; ipdb.set_trace()
                        pred_steds_adjusted_by_peak = None
                        if len(peaks) > 0:
                            peak_heights_tensor = torch.Tensor(peaks_stats['peak_heights'])
                            ind_sorted_desc = peak_heights_tensor.argsort(descending=True)

                            peak_heights_sorted = peak_heights_tensor[ind_sorted_desc]
                            peaks_sorted = peaks[ind_sorted_desc]

                            # top_5_peaks = peaks[peak_heights_tensor.topk(k=min(5, len(peaks))).indices]
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
                                for win_size in args.eval_flags.mq.win_size_around_peak:
                                    time_mask_adjusted_by_peak = torch.full_like(time_mask, True)
                                    time_mask_adjusted_by_peak[:, max(0, _peak - win_size): _peak + win_size] = False
                                    pred_steds_adjusted_by_peak, score_pred_sted = postprocessors["mq"].forward_w_pred_sted_arg(
                                        outputs["pred_sted"].masked_fill(time_mask_adjusted_by_peak[..., None], -float("inf")),
                                        # pred_sted_segment_aggregated.masked_fill(time_mask_adjusted_by_peak[..., None], -float("inf")),
                                        frames_id, video_ids=video_ids, time_mask=time_mask
                                    )
                                    pred_steds_sorted.append(pred_steds_adjusted_by_peak[0])
                                    pred_scores_sorted.append(_peak_height)

                    # if args.sted:
                    #     # assert len(pred_steds) == len(qtypes)
                    #     for video_id, pred_sted in zip(video_ids, pred_steds):
                    #         nlq_video_res[video_id] = {
                    #             "sted": pred_sted,
                    #             # "qtype": qtypes[video_id],
                    #         }

                # generate predictions in the format of NLQ
                # import ipdb; ipdb.set_trace()
                fps_for_tubedetr = 5 / scale_prediction
                pred_steds_sorted_seconds = np.array(
                    [[e[0] / fps_for_tubedetr, e[1] / fps_for_tubedetr] for e in pred_steds_sorted]
                )

                pred_steds_sorted_seconds_all_scales.append(pred_steds_sorted_seconds)
                pred_scores_sorted_all_scales.append(np.asarray(pred_scores_sorted))

        for evaluator in evaluator_list:
            if isinstance(evaluator, MQOrigEvaluator):
                evaluator.update({
                    # {
                    "clip_uid": video_ids[0],
                    # "annotation_uid": annotation_uid[0],
                    "captions": captions,
                    "captions_idx": captions_idx,
                    # "query_idx": query_idx[0],
                    "predicted_times": np.concatenate(pred_steds_sorted_seconds_all_scales),
                    # "predicted_scores": copy.deepcopy(torch.stack(pred_scores_sorted).numpy()),
                    "predicted_scores": np.concatenate(pred_scores_sorted_all_scales),
                    # }
                })

        # if args.generate_and_save_preds_for_all_captions:
        # evaluator_list[0].save_results_of_clip_to_csv(batch_dict["video_ids"][0])

        if args.eval_flags.plot_pred:
            if i_batch > 50:
                print("WARNING WARNING WARNING WARNING STOPPING TESTING ARBITRARILY")
                break
            assert len(video_ids) == 1

            PATH_PREDS = "/checkpoint/raghavgoyal/experiments/joint/mq_nlq_vq2d_round_robin_vq2d_proj_layer_lower_text_enc_lr/eval/retrieval_postNMS.json"
            with open(PATH_PREDS, "rb") as fp:
                predictions_from_json = json.load(fp)

            predictions_from_json_video = predictions_from_json['results'][video_ids[0]]
            predictions_from_json_video_caption = [
                e for e in predictions_from_json_video if e['label'] in captions
            ]
            list_segment_score_tuple = []
            dict_segment_to_score = {}
            for __e in predictions_from_json_video_caption:
                if tuple(__e['segment']) not in dict_segment_to_score:
                    dict_segment_to_score[tuple(__e['segment'])] = 0
                    list_segment_score_tuple.append(tuple((__e['segment'][0], __e['segment'][1], __e['score'])))
                #     dict_segment_to_score[tuple(__e['segment'])] = __e['score']
            # if args.debug: import ipdb; ipdb.set_trace()

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

            p_out_video = Path(args.output_dir) / 'plot_pred' / f"{video_ids[0]}_{captions_idx[0]}"
            p_out_video.mkdir(parents=True, exist_ok=True)

            assert len(frames_id[0]) == len(batch_dict['images_list_pims'][0])

            # gt_extent_img_ids = [
            #     min([int(e['image_id'].split('_')[-1]) for e in targets_non_empty_boxes]),
            #     max([int(e['image_id'].split('_')[-1]) for e in targets_non_empty_boxes])
            # ]

            gt_img_ids = batch_dict['inter_frames_all_segments'][0]

            pred_steds_top5_mm_ss_format = [
                [e[0] // 60 + e[0] % 60 / 100, e[1] // 60 + e[1] % 60 / 100]
                for e in pred_steds_sorted_seconds
            ]

            print(f"pred_steds_top_5: {pred_steds_sorted}")
            print(f"pred_steds_top5_mm_ss_format: {pred_steds_top5_mm_ss_format}")
            print(f"GT steds: {gt_img_ids}")
            print(f"Caption: {captions[0]}")
            print(f"num frames: {len(frames_id[0])}\n")

            # write info to text
            # p_text_file = p_out_video.parent / "info.txt"
            # with p_text_file.open('a') as f:
            #     f.write(f"Video: {video_ids[0]}\n")
            #     # f.write(f"Annotation id: {annotation_uid[0]}\n")
            #     f.write(f"Query id: {query_idx[0]}\n")
            #     f.write(f"Caption: {captions[0]}\n")
            #     f.write(f"Ground truth: [{gt_extent_img_ids[0]}, {gt_extent_img_ids[1]}]\n")
            #     f.write(f"pred_steds_sorted: {pred_steds_sorted}\n")
            #     f.write(f"pred_steds_top_5_mm_ss: {pred_steds_top5_mm_ss_format}\n")
            #     f.write(f"num frames: {len(frames_id[0])}\n")
            #     f.write("\n")

            # write frames
            p_out_frames = p_out_video / "frames"
            p_out_frames.mkdir(exist_ok=True)

            shutil.copytree(p_out_video_bank, p_out_frames, dirs_exist_ok=True)
            # """
            for _frame_id, _image in tqdm(zip(
                frames_id[0], batch_dict['images_list_pims'][0]
            ), total=len(frames_id[0])):
                # flag_relevant_frame = False
                # if _frame_id in gt_img_ids:
                #     flag_relevant_frame = True

                # for __k, __sted in enumerate(pred_steds_sorted):
                #     if _frame_id >= __sted[0] and _frame_id <= __sted[1]:
                #         flag_relevant_frame = True

                # if not flag_relevant_frame:
                #     continue

                _im = Image.fromarray(_image)
                img_w, img_h = _im.size

                fig, ax = plt.subplots()
                ax.axis("off")
                ax.imshow(_im, aspect="auto")

                if _frame_id in gt_img_ids:
                    props = dict(boxstyle='round', facecolor='green', alpha=0.5)
                    ax.text(0.05, 0.95, "GT",
                            transform=ax.transAxes, fontsize=15,
                            verticalalignment='top', bbox=props)

                k_selected = []
                for __k, __sted_secs in enumerate(list_segment_score_tuple[:5]):
                    __sted = [int(__sted_secs[0] * 5), int(__sted_secs[1] * 5)]
                    if _frame_id >= __sted[0] and _frame_id <= __sted[1]:
                        k_selected.append(__k + 1)
                if len(k_selected) > 0:
                    props = dict(boxstyle='round', facecolor='blue', alpha=0.5)
                    ax.text(0.05, 0.95, f"{k_selected}th",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                        # verticalalignment='top', horizontalalignment='right', bbox=props)

                fig.set_dpi(100)
                fig.set_size_inches(img_w / 100, img_h / 100)
                fig.tight_layout(pad=0)

                # save image with eventual box
                fig.savefig(
                    p_out_frames / f"{_frame_id:04d}.jpg",
                    format="jpg",
                )
                plt.close(fig)
            # if args.debug: import ipdb; ipdb.set_trace()
            # """

            # make score plot
            # data_score_plot = {'frame_id': [], 'target': []}
            # data_score_plot.update({f"top_{__i + 1}": [] for __i in range(len(pred_steds_sorted))})

            # for _frame_id in frames_id[0]:
            #     data_score_plot['frame_id'].append(_frame_id)
            #     if _frame_id >= gt_extent_img_ids[0] and _frame_id <= gt_extent_img_ids[1]:
            #         data_score_plot['target'].append(1)
            #     else:
            #         data_score_plot['target'].append(0)

            #     for __i, __e in enumerate(pred_steds_sorted):
            #         if _frame_id >= __e[0] and _frame_id <= __e[1]:
            #             data_score_plot[f"top_{__i + 1}"].append(1)
            #         else:
            #             data_score_plot[f"top_{__i + 1}"].append(0)

            # data_score_plot['signal_sm'] = signal_sm

            # plt.rcParams["figure.figsize"] = (10, 10)
            # plt.plot(data_score_plot['frame_id'], data_score_plot['target'], label='Ground truth')
            # plt.plot(data_score_plot['frame_id'], data_score_plot['signal_sm'], label='Score')
            # for __i in range(len(pred_steds_top_5)):
            #     plt.plot(data_score_plot['frame_id'], data_score_plot[f"top_{__i + 1}"], label=f"top_{__i + 1}")
            # plt.legend()

            # plt.savefig(p_out_video / "score_target_plot.png", format="png")
            # plt.close()
            # sns.set_theme()
            # sns.set(style="white", font_scale=1.5)
            # sns.set_palette("crest")

            # plt.rcParams["figure.figsize"] = (15, 12)

            # fig, axs = plt.subplots(2, 1)
            # axs[0].plot(data_score_plot['frame_id'], data_score_plot['target'], label='Ground truth', linewidth=4, color='teal')
            # axs[0].plot(data_score_plot['frame_id'], data_score_plot['signal_sm'], label='Score', linewidth=3, color='orange')
            # axs[0].set_ylabel('Score')
            # axs[0].set_title(captions[0])
            # axs[0].legend()

            # for __i in range(len(pred_steds_sorted)):
            #     axs[1].plot(data_score_plot['frame_id'], data_score_plot[f"top_{__i + 1}"], label=f"top_{__i + 1}", linewidth=4)

            # axs[1].annotate(
            #     'top 1',
            #     xy=(int(np.median(np.nonzero(np.asarray(data_score_plot['top_1']) == 1)[0])), 1),
            #     xytext=(int(np.median(np.nonzero(np.asarray(data_score_plot['top_1']) == 1)[0])), 1.2),
            #     arrowprops=dict(facecolor='black', shrink=0.05, width=7)
            # )
            # axs[1].set_xlabel('frame id')
            # axs[1].set_ylabel('Score')
            # axs[1].legend()

            # fig.tight_layout()
            # plt.savefig(p_out_video / "score_target_plot.png", format="png")
            # plt.close()

            # torch.save(data_score_plot, p_out_video / 'data_score_plot.pth')

            # stitch frames to video
            os.system(
                f"ffmpeg -y -loglevel error -framerate 5 -pattern_type glob -i '{str(p_out_frames)}/*.jpg' -c:v libx264 -pix_fmt yuv420p {str(p_out_video)}/out.mp4"
            )
            # import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    print("Finished generating proposal files")

    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    for evaluator in evaluator_list:
        if isinstance(evaluator, MQOrigEvaluator):
            print("Starting generating results")
            mq_retrieval_res = evaluator.summarize(args)

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if mq_retrieval_res is not None:
        stats["mq_retrieval_res"] = mq_retrieval_res

    return stats
