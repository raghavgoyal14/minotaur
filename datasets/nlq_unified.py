import os
import copy
import torch
# import gzip
import json
import pims
from torch.utils.data import Dataset
from pathlib import Path
from .video_transforms import make_video_transforms, prepare
from util.vq2d_utils import get_clip_name_from_clip_uid
from util.misc import flatten_list_of_list
import numpy as np
import random
from scipy import stats


class NLQDatasetWrapper(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        transforms,
        is_train=False,
        fps=5,
        tmp_crop=False,
        tmp_loc=True,
        args=None,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
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

        self._transforms = transforms
        self.is_train = is_train
        self.split_name = 'train' if self.is_train else 'val'
        # if self.args.test:
        #     raise ValueError(f"self.split_name: {self.split_name}. NOT SUITABLE FOR TEST SET in nlq_unified.py __init__")

        self.video_max_len = self.args.model.nlq.video_max_len
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.vid2imgids = {}

        self.segment_types = self.args.model.segment_types
        self.prob_segment_type = self.args.train_flags.nlq.prob_segment_type
        assert self.segment_types == ["fg", "left_trun", "right_trun", "bg"]
        assert sum(self.prob_segment_type) == 1

        # flags specific tp train / eval
        if self.is_train:
            self.stride = self.args.train_flags.nlq.stride
        else:
            self.stride = self.args.eval_flags.nlq.stride

        self.plot_pred = self.args.eval_flags.plot_pred
        self.use_full_video_for_eval = self.args.eval_flags.use_full_video_for_eval

        self.annotations = []
        empty_queries_count = 0
        for clip_uid, clip_annotation in self.data_json.items():
            clip_path = os.path.join(self.vid_folder, get_clip_name_from_clip_uid(clip_uid))
            video_reader = pims.Video(clip_path)

            duration_clip = video_reader.duration  # in secs
            frame_rate_clip = int(video_reader.frame_rate)  # in secs
            assert frame_rate_clip == 5
            assert int(duration_clip * frame_rate_clip) == len(video_reader)  # modify this condition?

            sampling_rate = fps / frame_rate_clip
            assert sampling_rate <= 1  # downsampling at fps

            video_start_frame = 0
            video_end_frame = len(video_reader)

            zipper = zip(
                clip_annotation["exact_times"],
                clip_annotation["sentences"],
                clip_annotation["annotation_uids"],
                clip_annotation["query_idx"],
            )

            for start_end_time, sentence, ann_uid, query_idx in zipper:
                video_tube_start_frame = int(start_end_time[0] * frame_rate_clip)
                video_tube_end_frame = int(start_end_time[1] * frame_rate_clip)

                if self.is_train and video_tube_start_frame >= video_tube_end_frame:
                    if self.args.train_flags.nlq.remove_empty_queries:
                        empty_queries_count += 1
                        continue
                    else:
                        video_tube_end_frame = video_tube_start_frame + np.random.randint(low=1, high=5)

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
                key = (clip_uid, ann_uid, query_idx)
                self.vid2imgids[key] = [frame_ids, inter_frames]

                self.annotations.append({
                    "clip_uid": clip_uid,
                    "caption": sentence,
                    "annotation_uid": ann_uid,
                    "query_idx": query_idx,
                })
        print(f"\nNumber of empty queries removed: {empty_queries_count}")
        print(f"Size of the split: {len(self.annotations)}")

        # self.annotations = np.asarray(self.annotations)

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
                _size_subsample = self.args.train_flags.nlq.eval_set_size.debug
            elif not self.args.eval:
                _size_subsample = self.args.train_flags.nlq.eval_set_size.train
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
        clip_uid = video["clip_uid"]
        annotation_uid = video["annotation_uid"]
        caption = video["caption"]

        key = (clip_uid, annotation_uid, video["query_idx"])
        frame_ids, inter_frames = self.vid2imgids[key]
        inter_frames = sorted(inter_frames)
        clip_start = frame_ids[0]  # included
        clip_end = frame_ids[-1]  # excluded

        # if self.args.debug: import ipdb; ipdb.set_trace()

        # MQ sampling
        # Gather inter_frames from all instances of the (clip, caption) pair - useful in multiple instances case
        annotations_from_all_instances = [e for e in self.annotations if e['clip_uid'] == clip_uid and e['caption'] == caption]
        inter_frames_all_segments = []
        for e in annotations_from_all_instances:
            __key = (e["clip_uid"], e["annotation_uid"], e["query_idx"])
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

        # if self.args.debug: import ipdb; ipdb.set_trace()

        # shorten window during training and TESTINGGGGGGGGGGG
        # if self.is_train and len(frame_ids) > self.video_max_len_train and len(inter_frames) < self.video_max_len_train:
        # flag_bg_segment = False
        # if self.is_train and len(frame_ids) > self.video_max_len and len(inter_frames) < (self.video_max_len - 1):
        #     if random.random() > 1.0 and frame_ids.index(min(inter_frames)) > 50:
        #         # raise ValueError("flag_bg_segment is currently disabled")
        #         flag_bg_segment = True
        #         offset_from_start_of_clip = np.random.randint(low=0, high=frame_ids.index(min(inter_frames)), size=1)[0]
        #         frame_ids = frame_ids[offset_from_start_of_clip: min(offset_from_start_of_clip + self.video_max_len, frame_ids.index(min(inter_frames)))]
        #     else:
        #         len_window = self.video_max_len
        #         offset_from_start_of_tube = np.random.randint(low=0, high=(len_window - len(inter_frames) - 1), size=1)[0]
        #         frame_ids = frame_ids[max(frame_ids.index(min(inter_frames)) - offset_from_start_of_tube, 0): frame_ids.index(min(inter_frames)) - offset_from_start_of_tube + len_window]
        #     # print(flag_bg_segment)
        # elif self.is_train and len(frame_ids) > self.video_max_len and len(inter_frames) > (self.video_max_len - 1):
        #     len_window = self.video_max_len
        #     offset_from_start_of_tube = np.random.randint(low=0, high=(len(inter_frames) - len_window + 1), size=1)[0]
        #     frame_ids = sorted(inter_frames)[offset_from_start_of_tube: offset_from_start_of_tube + len_window]

        # decoding video
        clip_path = os.path.join(
            self.vid_folder, get_clip_name_from_clip_uid(clip_uid)
        )
        video_reader = pims.Video(clip_path)

        h = video_reader.shape[1]
        w = video_reader.shape[2]

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

        # sentence_embedding = None
        # if self.args.model.nlq.use_sentence_text_embeddings:
        #     sentence_embedding = self.dict_nlq_sentences_to_embeddings[key]

        tmp_target = {
            "video_id": clip_uid,
            "annotation_uid": annotation_uid,
            "query_idx": video["query_idx"],
            "inter_idx": [inter_idx[0], inter_idx[-1]]
            if inter_idx
            else [
                -100,
                -100,
            ],  # start and end (included) indexes for the annotated moment
            "frames_id": frame_ids,
            "caption": caption,
            "segment_type_selected": segment_type_selected,
            "task_name": "nlq",
        }
        if self.plot_pred:
            tmp_target['images_list_pims'] = images_list_pims
        if self.stride:
            return images[:, :: self.stride], targets, tmp_target, images
        return images, targets, tmp_target


def build(image_set, args):
    data_dir = Path(args.data.nlq.path)

    if args.test:
        ann_file = data_dir / "dataset/nlq_official_v1" / "test_annotated.json"
        vid_dir = data_dir / 'clips_test_shorter_side_320'
    elif image_set == "val":
        ann_file = data_dir / "dataset/nlq_official_v1" / "val.json"
        vid_dir = data_dir / 'clips_val_shorter_side_320'
    else:
        ann_file = data_dir / "dataset/nlq_official_v1" / "train.json"
        vid_dir = data_dir / 'clips_train_shorter_side_320'

    dataset = NLQDatasetWrapper(
        vid_dir,
        ann_file,
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
