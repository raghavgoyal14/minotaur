from pathlib import Path
from typing import Dict, List

import numpy as np

import util.dist as dist

import json
from functools import reduce
from util.box_ops import np_box_iou

import sys
sys.path.insert(0, "/scratch/hdd001/home/rgoyal/FB/code_dump/episodic-memory-old/VQ2D")
from vq2d.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack, BBox
from typing import Any, Dict, List, Sequence, Union
from pprint import pprint
# from vq2d.baselines import convert_annot_to_bbox


def convert_annot_to_bbox(annot: Dict[str, Any]) -> BBox:
    return BBox(
        annot["frame_number"],
        annot["x"],
        annot["y"],
        annot["x"] + annot["width"],
        annot["y"] + annot["height"],
    )


class VQ2DOrigEvaluator(object):
    def __init__(self, ):
        self.predictions = {}

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def save(
        self,
        tsa_weights,
        text_weights,
        spatial_weights,
        pred_sted,
        image_ids,
        video_ids,
    ):
        pass

    def postproces_predictions(self, predictions):
        pred_rt = []
        gt_rt = []
        vc_boxes = []
        dataset_uids = []
        acc_frames = []
        tot_frames = []

        for k, v in predictions.items():
            gt_rt.append(
                ResponseTrack(
                    [convert_annot_to_bbox(rf) for rf in v["gt_response_track_video"]]
                )
            )

            pred_rt.append(
                [
                    ResponseTrack([convert_annot_to_bbox(rf) for rf in v["pred_response_track_video"]], score=1.0)
                ]
            )
            vc_boxes.append(convert_annot_to_bbox(v["visual_crop_boxes_video"]))
            acc_frames.append(v["visual_crop_boxes_video"]["frame_number"])  # dummy number
            tot_frames.append(v["visual_crop_boxes_video"]["frame_number"])  # dummy number
            dataset_uids.append(k)
        return pred_rt, gt_rt, vc_boxes, acc_frames, tot_frames

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        self.predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})

    def summarize(self):
        if dist.is_main_process():
            pred_rt, gt_rt, vc_boxes, acc_frames, tot_frames = self.postproces_predictions(self.predictions)
            self.results = compute_visual_query_metrics(pred_rt, gt_rt, vc_boxes, acc_frames, tot_frames)
            pprint(self.results)

            return self.results
        return None, None
