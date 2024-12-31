from pathlib import Path
from typing import Dict, List

import numpy as np

import util.dist as dist

import json
from functools import reduce
from util.box_ops import np_box_iou

import sys
sys.path.insert(0, "/private/home/raghavgoyal/code/episodic-memory-b6a7ec4/NLQ/VSLNet/")
from utils.evaluate_ego4d_nlq import evaluate_nlq_performance, display_results
from typing import Any, Dict, List, Sequence, Union
from pprint import pprint


class NLQOrigEvaluator(object):
    def __init__(self, root, split):
        p_gt_json_file = Path(root) / f"nlq_{split}.json"
        assert p_gt_json_file.is_file()

        with p_gt_json_file.open('rb') as file_id:
            self.ground_truth = json.load(file_id)
        self.thresholds = [0.3, 0.5, 0.01]
        self.topK = [1, 3, 5]
        self.predictions = {}

    def accumulate(self):
        pass

    def update(self, predictions):
        # self.predictions.append(predictions)
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

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        self.predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})
        # print(f"Rank: {dist.get_rank()}, all_predictions: {all_predictions}")

        # self.predictions = all_predictions[0]

    def postprocess_predictions(self, predictions):
        return predictions

    def summarize(self):
        if dist.is_main_process():
            # import ipdb; ipdb.set_trace()
            self.predictions = [*self.predictions.values()]

            print(f"Length of predictions: {len(self.predictions)}")
            self.results, mIoU = evaluate_nlq_performance(
                self.predictions, self.ground_truth, self.thresholds, self.topK
            )
            results_str = display_results(
                self.results, mIoU, self.thresholds, self.topK
            )

            print(results_str)

            # pprint(self.results)
            # pprint(mIoU)

            return self.results
        return None, None
