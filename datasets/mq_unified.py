import os
import copy
import torch
# import gzip
import json
import pims
from torch.utils.data import Dataset
from pathlib import Path
from .video_transforms import make_video_transforms, prepare
from util.vq2d_utils import get_clip_name_from_clip_uid, extract_window_with_context
from util.misc import flatten_list_of_list
from einops import rearrange, asnumpy
import time
import numpy as np
import random
from scipy import stats

SKIP_VIDS = []


class MQDatasetWrapper(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        class_to_idx_file,
        transforms,
        is_train=False,
        # video_max_len=200,
        fps=5,
        tmp_crop=False,
        tmp_loc=True,
        # stride=0,
        # plot_pred=False,
        # use_full_video_for_eval=False,
        # generate_and_save_preds_for_all_captions=False,
        # extract_backbone_features=False,
        # use_fps_jitter=False,
        # debug=False,
        # eval_only=False,
        args=None,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
        :param stride: temporal stride k
        """
        assert args is not None
        self.args = args
        self.vid_folder = vid_folder
        with ann_file.open("rb") as fp:
            self.data_json = json.load(fp)

        with class_to_idx_file.open("rb") as fp:
            self.class_to_idx = json.load(fp)

        self._transforms = transforms
        self.is_train = is_train
        self.split_name = 'train' if self.is_train else 'val'
        if self.args.test:
            self.split_name = 'test'
        #     raise ValueError(f"self.split_name: {self.split_name}. NOT SUITABLE FOR TEST SET in mq.py __init__")

        self.video_max_len = self.args.model.mq.video_max_len
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.vid2imgids = {}

        self.segment_types = self.args.model.segment_types
        self.prob_segment_type = self.args.train_flags.mq.prob_segment_type
        assert self.segment_types == ["fg", "left_trun", "right_trun", "bg"]
        assert sum(self.prob_segment_type) == 1

        self.use_fps_jitter = self.args.train_flags.mq.use_fps_jitter
        self.plot_pred = self.args.eval_flags.plot_pred
        self.use_full_video_for_eval = self.args.eval_flags.use_full_video_for_eval

        # flags specific tp train / eval
        if self.is_train:
            self.stride = self.args.train_flags.mq.stride
        else:
            self.stride = self.args.eval_flags.mq.stride

        self.annotations = []
        empty_queries_count = 0
        for clip_uid, clip_annotation in self.data_json.items():
            if clip_annotation['subset'] != self.split_name:
                continue

            clip_path = os.path.join(self.vid_folder, get_clip_name_from_clip_uid(clip_uid))

            # import ipdb; ipdb.set_trace()
            assert os.path.exists(clip_path)

            video_reader = pims.Video(clip_path)
            duration_clip = video_reader.duration  # in secs
            frame_rate_clip = int(video_reader.frame_rate)  # in secs
            assert frame_rate_clip == 5
            assert int(duration_clip * frame_rate_clip) == len(video_reader)  # modify this condition?

            sampling_rate = fps / frame_rate_clip
            assert sampling_rate <= 1  # downsampling at fps

            video_start_frame = 0
            video_end_frame = len(video_reader)

            for _query_idx, _ann in enumerate(clip_annotation['annotations']):
                video_tube_start_frame = int(_ann['start_time'] * frame_rate_clip)
                video_tube_end_frame = int(_ann['end_time'] * frame_rate_clip)

                if self.is_train and video_tube_start_frame >= video_tube_end_frame:
                    # video_tube_end_frame = video_tube_start_frame + np.random.randint(low=1, high=5)
                    empty_queries_count += 1
                    continue

                start_frame = (
                    video_start_frame if self.tmp_loc else video_tube_start_frame
                )
                end_frame = video_end_frame if self.tmp_loc else video_tube_end_frame

                frame_ids = [start_frame]
                for frame_id in range(start_frame, end_frame):
                    if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                        frame_ids.append(frame_id)

                inter_frames = set(
                    [
                        frame_id
                        for frame_id in frame_ids
                        if video_tube_start_frame <= frame_id <= video_tube_end_frame
                    ]
                )  # frames in the annotated moment
                # import ipdb; ipdb.set_trace()
                key = (clip_uid, _query_idx)
                self.vid2imgids[key] = [frame_ids, inter_frames]

                self.annotations.append({
                    "clip_uid": clip_uid,
                    "caption": _ann['label'],
                    "caption_idx": self.class_to_idx[_ann['label']],
                    "query_idx": _query_idx,
                    "clip_path": clip_path,
                })

        print(f"\nNumber of empty queries removed: {empty_queries_count}")
        print(f"Size of the split: {len(self.annotations)}")

        self.annotations = np.asarray(self.annotations)

        if not self.is_train:
            print("[EVAL] Trimming down duplicate pairs of (clip_uid, caption), so as"
                    " to reduce test time")
            annotations_trimmed = {}
            for _ann in self.annotations:
                key = f"{_ann['clip_uid']}_{_ann['caption_idx']}"
                value = copy.deepcopy(_ann)
                value['query_idx'] = 0
                if key not in annotations_trimmed:
                    annotations_trimmed[key] = value
                else:
                    pass
            self.annotations = np.asarray([e for e in annotations_trimmed.values()])

        # DEBUG
        if self.is_train and self.args.debug:
            _size_subsample = 20
            print(f"[TRAIN] WARNING WARNING WARNING WARNING WARNING:"
                  f" Subsampling train set for debugging"
                  f" from {len(self.annotations)} to size: {_size_subsample}")
            self.annotations = self.annotations[:_size_subsample]

        if not self.is_train:
            _size_subsample = None
            if self.args.debug:
                _size_subsample = self.args.train_flags.mq.eval_set_size.debug
            elif not self.args.eval:
                _size_subsample = self.args.train_flags.mq.eval_set_size.train
            else:
                pass

            if _size_subsample is not None:
                print(f"[EVAL] WARNING WARNING WARNING WARNING WARNING:"
                    f" Subsampling eval set for debugging"
                    f" from {len(self.annotations)} to size: {_size_subsample}")

                indices_to_subsample = [
                    *range(0, len(self.annotations), len(self.annotations) // _size_subsample)
                ][:_size_subsample]

                print(f"Indices used to subsample eval set "
                    f"(len: {len(indices_to_subsample)}) : {indices_to_subsample}")
                print(f"Num unique clip_uids before: {len(set([e['clip_uid'] for e in self.annotations]))}")
                self.annotations = [self.annotations[e] for e in indices_to_subsample]
                print(f"Num unique clip_uids after subsampling: {len(set([e['clip_uid'] for e in self.annotations]))}")

        self._summarize()

    def _summarize(self, ):
        len_frames_ids = []
        len_inter_frames = []
        for k, v in self.vid2imgids.items():
            len_frames_ids.append(len(v[0]))
            len_inter_frames.append(len(v[1]))

        len_frames_ids = np.array(len_frames_ids)            
        len_inter_frames = np.array(len_inter_frames)

        bins = [0, 10, 50, 100, 200, 500, 1000, 10000]
        hist = np.histogram(len_inter_frames, bins=bins)[0]
        print(f"Histogram:\n\tbins: {bins}\n\thist: {hist}")

        print(f"Length inter frames. "
              f"Mean: {np.mean(len_inter_frames)}, "
              f"Mode: {stats.mode(len_inter_frames).mode}, "
              f"Min: {np.min(len_inter_frames)}, Max: {np.max(len_inter_frames)}"
            )
        print(f"Length frame ids. "
              f"Mean: {np.mean(len_frames_ids)}, "
              f"Mode: {stats.mode(len_frames_ids).mode}, "
              f"Min: {np.min(len_frames_ids)}, Max: {np.max(len_frames_ids)}"
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, qtype, inter_idx, frames_id, caption
        """
        video = self.annotations[idx]
        # clip_uid = video["clip_uid"]
        clip_path = video["clip_path"]
        clip_uid = video["clip_uid"]
        caption = video["caption"]
        caption_idx = video["caption_idx"]
        query_idx = video["query_idx"]
        key = (clip_uid, query_idx)
        frame_ids, inter_frames = self.vid2imgids[key]
        inter_frames = sorted(inter_frames)
        clip_start = frame_ids[0]  # included
        clip_end = frame_ids[-1]  # excluded

        # Gather inter_frames from all instances of the (clip, caption) pair - useful in multiple instances case
        annotations_from_all_instances = [e for e in self.annotations if e['clip_uid'] == clip_uid and e['caption'] == caption]
        inter_frames_all_segments = []
        for e in annotations_from_all_instances:
            __key = (e["clip_uid"], e["query_idx"])
            __frame_ids, __inter_frames = self.vid2imgids[__key]
            assert frame_ids == __frame_ids
            inter_frames_all_segments.append(__inter_frames)
        inter_frames_all_segments_flattened = sorted(set(flatten_list_of_list([list(e) for e in inter_frames_all_segments])))

        bg_segments = []
        bg_segment_candidate = []
        for e in frame_ids:
            if e in inter_frames_all_segments_flattened:
                if len(bg_segment_candidate) > 0:
                    bg_segments.append(copy.deepcopy(bg_segment_candidate))
                bg_segment_candidate = []
            else:
                bg_segment_candidate.append(e)
        if len(bg_segment_candidate) > 0:
            bg_segments.append(copy.deepcopy(bg_segment_candidate))

        # SAMPLING
        segment_type_selected = None
        # import ipdb; ipdb.set_trace()
        if self.is_train:
            assert self.tmp_loc  # ensures that frame_ids start from 0
            frame_ids_segment_type = {e: None for e in self.segment_types}

            # bg
            frame_ids_segment_type["bg"] = None
            if len(bg_segments) > 0:
                bg_proposal = bg_segments[np.random.choice(len(bg_segments), size=1)[0]]
                if len(bg_proposal) > self.video_max_len:
                    offset = np.random.randint(low=0, high=len(bg_proposal) - self.video_max_len, size=1)[0]
                    bg_proposal = bg_proposal[offset:]
                frame_ids_segment_type["bg"] = bg_proposal

            # left_truncated
            if len(frame_ids) > self.video_max_len and (max(frame_ids) - max(inter_frames)) > self.video_max_len // 4 and len(inter_frames) > 5:
                if len(inter_frames) < (self.video_max_len - 1):
                    offset = np.random.randint(
                        low=min(inter_frames),
                        high=(list(inter_frames)[len(list(inter_frames)) // 2]),
                        size=1
                    )[0]
                else:
                    offset = np.random.randint(
                        low=max(inter_frames) - (self.video_max_len // 2),
                        high=max(inter_frames) - (self.video_max_len // 4),
                        size=1
                    )[0]
                frame_ids_segment_type["left_trun"] = frame_ids[offset: offset + self.video_max_len]

            # right_truncated
            if len(frame_ids) > self.video_max_len and (min(inter_frames) - min(frame_ids)) > self.video_max_len // 4 and len(inter_frames) > 5:
                if len(inter_frames) < (self.video_max_len - 1):
                    offset = np.random.randint(
                        low=(inter_frames[len(inter_frames) // 2]),
                        high=max(inter_frames),
                        size=1
                    )[0]
                else:
                    offset = np.random.randint(
                        low=min(inter_frames) + (self.video_max_len // 4),
                        high=min(inter_frames) + (self.video_max_len // 2),
                        size=1
                    )[0]
                frame_ids_segment_type["right_trun"] = frame_ids[max(offset - self.video_max_len, 0): offset]

            # fg: video length > window size && GT length < window size
            if len(frame_ids) > self.video_max_len and len(inter_frames) < (self.video_max_len - 1):
                len_window = self.video_max_len
                offset_from_start_of_tube = np.random.randint(low=0, high=(len_window - len(inter_frames) - 1), size=1)[0]
                frame_ids_segment_type["fg"] = frame_ids[max(min(inter_frames) - offset_from_start_of_tube, 0): min(inter_frames) - offset_from_start_of_tube + len_window]

            # fg: video length > window size && GT length > window size
            elif len(frame_ids) > self.video_max_len and len(inter_frames) > (self.video_max_len - 1):
                len_window = self.video_max_len
                offset_from_start_of_tube = np.random.randint(low=0, high=(len(inter_frames) - len_window + 1), size=1)[0]
                frame_ids_segment_type["fg"] = inter_frames[offset_from_start_of_tube: offset_from_start_of_tube + len_window]
            else:
                pass

            if len(frame_ids) <= self.video_max_len:
                print("NOoOOOoooooooooooooooo: len(frame_ids) <= self.video_max_len")

            segment_type_selected = None
            while segment_type_selected is None:
                segment_type_selected = np.random.choice(self.segment_types, size=1, p=self.prob_segment_type)[0]
                if frame_ids_segment_type[segment_type_selected] is None:
                    segment_type_selected = None

            frame_ids = frame_ids_segment_type[segment_type_selected]

        # FPS JITTER
        if self.is_train and self.use_fps_jitter:
            assert not self.stride
            if len(frame_ids) < 200:
                stride_for_jitter = np.random.choice([1, 2, 3], size=1, replace=False)[0]
            elif len(frame_ids) >= 200 and len(frame_ids) <= 400:
                stride_for_jitter = np.random.choice([2, 3, 5], size=1, replace=False)[0]
            elif len(frame_ids) >= 400 and len(frame_ids) <= 600:
                stride_for_jitter = np.random.choice([3, 5, 8], size=1, replace=False)[0]
            else:
                stride_for_jitter = 10

            indices_frames_to_subsample = [*range(0, len(frame_ids), stride_for_jitter)]
            frame_ids_candidate = [frame_ids[e] for e in indices_frames_to_subsample]
            inter_frames_candidate = [e for e in inter_frames if e in frame_ids_candidate]

            # print(f"len(frame_ids): {len(frame_ids)}, len(inter_frames): {len(inter_frames)}")
            # print(f"stride_for_jitter: {stride_for_jitter}, len(frame_ids_candidate): {len(frame_ids_candidate)}, len(inter_frames_candidate): {len(inter_frames_candidate)}")
            # print()

            # import ipdb; ipdb.set_trace()

            frame_ids = frame_ids_candidate
            inter_frames = inter_frames_candidate

            if len(inter_frames) == 0:
                segment_type_selected = 'bg'

        video_reader = pims.Video(clip_path)

        h = video_reader.shape[1]
        w = video_reader.shape[2]

        # import ipdb; ipdb.set_trace()
        images_list_pims = [video_reader[e] for e in frame_ids]
        images_list = np.array(images_list_pims)

        # prepare frame-level targets
        targets_list = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids):
            if img_id in inter_frames:
                anns = {'bbox': [5, 5, 5, 5]}
                anns = [anns]
                inter_idx.append(i_img)
            else:
                anns = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{clip_uid}_{img_id}"
            target["caption_idx"] = torch.tensor(caption_idx)
            targets_list.append(target)

        # import ipdb; ipdb.set_trace()
        # video spatial transform
        if self._transforms is not None:
            images, targets = self._transforms(images_list, targets_list)
        else:
            images, targets = images_list, targets_list

        if segment_type_selected == "bg":
            assert len([x for x in targets if len(x["boxes"])]) == 0

        if (
            inter_idx
        ):  # number of boxes should be the number of frames in annotated moment
            assert (
                len([x for x in targets if len(x["boxes"])])
                == inter_idx[-1] - inter_idx[0] + 1
            ), (len([x for x in targets if len(x["boxes"])]), inter_idx)

        # import ipdb; ipdb.set_trace()
        # temporal crop
        if self.tmp_crop:
            p = random.random()
            if p > 0.5:  # random crop
                # list possible start indexes
                if inter_idx:
                    starts_list = [i for i in range(len(frame_ids)) if i < inter_idx[0]]
                else:
                    starts_list = [i for i in range(len(frame_ids))]

                # sample a new start index
                if starts_list:
                    new_start_idx = random.choice(starts_list)
                else:
                    new_start_idx = 0

                # list possible end indexes
                if inter_idx:
                    ends_list = [i for i in range(len(frame_ids)) if i > inter_idx[-1]]
                else:
                    ends_list = [i for i in range(len(frame_ids)) if i > new_start_idx]

                # sample a new end index
                if ends_list:
                    new_end_idx = random.choice(ends_list)
                else:
                    new_end_idx = len(frame_ids) - 1

                # update everything
                prev_start_frame = frame_ids[0]
                prev_end_frame = frame_ids[-1]
                frame_ids = [
                    x
                    for i, x in enumerate(frame_ids)
                    if new_start_idx <= i <= new_end_idx
                ]
                images = images[:, new_start_idx: new_end_idx + 1]  # CTHW
                targets = [
                    x
                    for i, x in enumerate(targets)
                    if new_start_idx <= i <= new_end_idx
                ]
                # import ipdb; ipdb.set_trace()
                clip_start += frame_ids[0] - prev_start_frame
                clip_end += frame_ids[-1] - prev_end_frame
                if inter_idx:
                    inter_idx = [x - new_start_idx for x in inter_idx]

        if (
            self.is_train and len(frame_ids) > self.video_max_len
        ):  # densely sample video_max_len frames
            if inter_idx:
                starts_list = [
                    i
                    for i in range(len(frame_ids))
                    if inter_idx[0] - self.video_max_len < i <= inter_idx[-1]
                ]
            else:
                starts_list = [i for i in range(len(frame_ids))]

            # sample a new start index
            if starts_list:
                new_start_idx = random.choice(starts_list)
            else:
                new_start_idx = 0

            # select the end index
            new_end_idx = min(
                new_start_idx + self.video_max_len - 1, len(frame_ids) - 1
            )

            # update everything
            prev_start_frame = frame_ids[0]
            prev_end_frame = frame_ids[-1]
            frame_ids = [
                x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx
            ]
            images = images[:, new_start_idx: new_end_idx + 1]  # CTHW
            targets = [
                x for i, x in enumerate(targets) if new_start_idx <= i <= new_end_idx
            ]
            clip_start += frame_ids[0] - prev_start_frame
            clip_end += frame_ids[-1] - prev_end_frame
            if inter_idx:
                inter_idx = [
                    x - new_start_idx
                    for x in inter_idx
                    if new_start_idx <= x <= new_end_idx
                ]

        tmp_target = {
            "video_id": clip_uid,
            "query_idx": query_idx,
            "inter_idx": [inter_idx[0], inter_idx[-1]]
            if inter_idx
            else [
                -100,
                -100,
            ],  # start and end (included) indexes for the annotated moment
            "frames_id": frame_ids,
            "caption": caption,
            "caption_idx": caption_idx,
            "segment_type_selected": segment_type_selected,
            "task_name": "mq",
        }
        if self.plot_pred:
            tmp_target['images_list_pims'] = images_list_pims
            tmp_target['inter_frames_all_segments'] = inter_frames_all_segments_flattened
        if self.stride:
            return images[:, :: self.stride], targets, tmp_target, images
        return images, targets, tmp_target


def build(image_set, args):
    # if args.debug: import ipdb; ipdb.set_trace()
    data_dir = Path(args.data.mq.path)

    if args.test:
        ann_file = data_dir / "clip_annotations_test_annotated_huiyu.json"
        vid_dir = data_dir / 'clips_test_shorter_side_320'
    elif image_set == "val":
        ann_file = data_dir / "clip_annotations.json"
        vid_dir = data_dir / 'clips_val_shorter_side_320'
    else:
        ann_file = data_dir / "clip_annotations.json"
        vid_dir = data_dir / 'clips_train_shorter_side_320'

    class_to_idx_file = data_dir / "moment_classes_idx.json"

    dataset = MQDatasetWrapper(
        vid_dir,
        ann_file,
        class_to_idx_file,
        transforms=make_video_transforms(
            image_set, cautious=True, resolution=args.resolution
        ),
        is_train=image_set == "train",
        fps=args.fps,
        tmp_crop=args.tmp_crop and image_set == "train",
        tmp_loc=args.sted,
        args=args,
    )
    return dataset
