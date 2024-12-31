from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import util.dist as dist

import shutil
import json
from functools import reduce
from util.box_ops import np_box_iou

import sys
sys.path.insert(0, "/private/home/raghavgoyal/code/episodic-memory-b6a7ec4/MQ/")
from Evaluation.ego4d.generate_retrieval import gen_retrieval_multicore as gen_retrieval
from Evaluation.ego4d.get_retrieval_performance import evaluation_retrieval as eval_retrieval

from utils.evaluate_ego4d_nlq import evaluate_nlq_performance, display_results
from typing import Any, Dict, List, Sequence, Union
from pprint import pprint


class MQOrigEvaluator(object):
    def __init__(self, path_data, path_output, split):
        self.split = split
        self.path_data = path_data
        if split == 'test':
            self.p_gt_json_file = Path(self.path_data) / "clip_annotations_test_annotated_huiyu.json"
        else:
            self.p_gt_json_file = Path(self.path_data) / "clip_annotations.json"
        assert self.p_gt_json_file.is_file()

        with self.p_gt_json_file.open('rb') as file_id:
            self.ground_truth = json.load(file_id)

        self.p_path_proposal = Path(path_output) / "proposals"
        if dist.is_main_process():
            if self.p_path_proposal.is_dir():
                print(f" > Removing already present proposals dir: {self.p_path_proposal}")
                shutil.rmtree(str(self.p_path_proposal), ignore_errors=True)
                # self.p_path_proposal.rmdir()

            self.p_path_proposal.mkdir(exist_ok=True)

        # self.thresholds = [0.3, 0.5, 0.01]
        # self.topK = [1, 3, 5]
        self.predictions = {}  # clip_name to predictions

    def accumulate(self):
        pass

    def update(self, prediction):
        predictions_np_format = np.concatenate(
            (
                prediction["predicted_times"],
                prediction["predicted_scores"][:, None],
                np.array(prediction["captions_idx"] * prediction["predicted_times"].shape[0])[:, None],
            ),
            axis=1
        )
        if prediction["clip_uid"] in self.predictions:
            self.predictions[prediction["clip_uid"]].append(predictions_np_format)
        else:
            self.predictions[prediction["clip_uid"]] = [predictions_np_format]

        # save current results to csv file
        self.save_results_of_clip_to_csv(predictions_np_format, prediction["clip_uid"])

    def save_results_of_clip_to_csv(self, predictions, clip_uid):
        col_name = ["xmin", "xmax", "score", "label"]
        # import ipdb; ipdb.set_trace()
        keep = nms(predictions)
        new_df = pd.DataFrame(predictions[keep], columns=col_name)
        path = self.p_path_proposal / f"{clip_uid}.csv"

        new_df.to_csv(path, index=False, mode='a', header=not path.is_file())
        # new_df.to_csv(path, index=False)

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

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        # # self.predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})
        self.predictions = all_predictions[0]

    def postprocess_predictions(self, predictions):
        return predictions

    def summarize(self, args):
        if dist.is_main_process():
            data_dir = Path(self.path_data)
            opt = {
                "clip_anno": self.p_gt_json_file,
                "infer_datasplit": "validation" if self.split == 'val' else 'test',
                "moment_classes": data_dir / "moment_classes_idx.json",
                "output_path": Path(args.output_dir),
                "detect_result_file": "detections_postNMS.json",
                "retrieval_result_file": "retrieval_postNMS.json",
                "prop_path": "proposals",
                "nms_alpha_detect": 0.46,
                "tIoU_thr": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
            gen_retrieval(opt)
            eval_result = eval_retrieval(opt)
            return eval_result
        return None


def nms(dets, thresh=0.4):
        """Pure Python NMS baseline."""
        if len(dets) == 0: return []
        x1 = dets[:, 0]
        x2 = dets[:, 1]
        scores = dets[:, 2]
        lengths = x2 - x1
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1)
            ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
