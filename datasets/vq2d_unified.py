import os
import copy
import torch
import gzip
import json
import pims
from torch.utils.data import Dataset
from pathlib import Path
from .video_transforms import make_video_transforms, prepare
from util.vq2d_utils import get_clip_name_from_clip_uid, extract_window_with_context
from einops import rearrange, asnumpy
import time
import ffmpeg
import numpy as np
import random

from scipy import stats

SKIP_VIDS = ['val_0000003782', ]


class VQ2DDatasetWrapper(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        transforms,
        transforms_reference_crop,
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
        :param video_max_len: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
        :param stride: temporal stride k
        """
        assert args is not None
        self.args = args
        self.vid_folder = vid_folder
        with gzip.open(ann_file, "rt") as fp:
            self.annotations = json.load(fp)

        self._transforms = transforms
        self._transforms_reference_crop = transforms_reference_crop
        self.is_train = is_train

        self.video_max_len = self.args.model.vq2d.video_max_len
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.vid2imgids = (
            {}
        )  # map video_id to [list of frames to be forwarded, list of frames in the annotated moment]

        # flags specific to train / eval
        if self.is_train:
            self.stride = self.args.train_flags.vq2d.stride
        else:
            self.stride = self.args.eval_flags.vq2d.stride
        self.plot_pred = self.args.eval_flags.plot_pred
        self.use_full_video_for_eval = self.args.eval_flags.use_full_video_for_eval

        # dict_object_title = {}
        for i_vid, video in enumerate(self.annotations):
            video_fps = video['metadata']["clip_fps"]  # used for extraction
            sampling_rate = fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps

            # object title
            if 'object_title' not in video:
                raise ValueError("NOOOOO: object title anno not present in video")
            # else:
            #     if video['object_title'] in dict_object_title:
            #         dict_object_title[video['object_title']] += 1
            #     else:
            #         dict_object_title[video['object_title']] = 1

            #### metadata for vq2d
            query_frame = video["query_frame"]
            vcfno = video["visual_crop"]["frame_number"]

            video_start_frame = 0
            # video_end_frame = max(query_frame, vcfno) + 1
            video_end_frame = query_frame + 1

            video_tube_start_frame = min([e['frame_number'] for e in video['response_track']])
            video_tube_end_frame = max([e['frame_number'] for e in video['response_track']])
            ####

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
            self.vid2imgids[video["dataset_uid"]] = [frame_ids, inter_frames]

        # Modding annotations based on SKIP IDS
        # for e in SKIP_VIDS:
        #     if e in self.vid2imgids:
        #         print(f"Removing video id from vid2d: {e}")
        #         del self.vid2imgids[e]

        # self.annotations = [e for e in self.annotations if e['dataset_uid'] not in SKIP_VIDS]

        # if args.debug: import ipdb; ipdb.set_trace()

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
                _size_subsample = self.args.train_flags.vq2d.eval_set_size.debug
            elif not self.args.eval:
                _size_subsample = self.args.train_flags.vq2d.eval_set_size.train
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
            len_inter_frames.append(len(v[1]))
            len_frames_ids.append(len(v[0]))

        len_frames_ids = np.array(len_frames_ids)
        len_inter_frames = np.array(len_inter_frames)

        # if self.args.debug: import ipdb; ipdb.set_trace()

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
        video_id = video["dataset_uid"]
        vcfno = video["visual_crop"]["frame_number"]
        query_frame = video["query_frame"]
        # if self.args.debug: import ipdb; ipdb.set_trace()
        caption = f"Where is the {video['object_title']}?"
        clip_start = 0  # included
        clip_end = max(query_frame, vcfno) + 1  # excluded
        frame_ids, inter_frames = self.vid2imgids[video_id]

        # if args.debug: import ipdb; ipdb.set_trace()
        flag_bg_segment = False
        if self.is_train and len(frame_ids) > self.video_max_len and len(inter_frames) < self.video_max_len:
            if random.random() > 1.0 and frame_ids.index(min(inter_frames)) > 50:
                flag_bg_segment = True
                offset_from_start_of_clip = np.random.randint(low=0, high=frame_ids.index(min(inter_frames)), size=1)[0]
                frame_ids = frame_ids[offset_from_start_of_clip: min(offset_from_start_of_clip + self.video_max_len, frame_ids.index(min(inter_frames)))]
            else:
                len_window = self.video_max_len
                offset_from_start_of_tube = np.random.randint(low=0, high=(len_window - len(inter_frames) - 1), size=1)[0]
                frame_ids = frame_ids[max(frame_ids.index(min(inter_frames)) - offset_from_start_of_tube, 0): frame_ids.index(min(inter_frames)) - offset_from_start_of_tube + len_window]

        # decoding video
        clip_path = os.path.join(
            self.vid_folder, get_clip_name_from_clip_uid(video["clip_uid"])
        )
        video_reader = pims.Video(clip_path)
        # query_frame = video["query_frame"]
        vcfno = video["visual_crop"]["frame_number"]

        # clip_frames = video_reader[clip_start: clip_end + 1]

        h = video_reader.shape[1]
        w = video_reader.shape[2]

        h_orig = video['response_track'][0]['original_height']
        w_orig = video['response_track'][0]['original_width']

        scale_h = h / h_orig
        scale_w = w / w_orig

        trajectory = {}
        for _rf in video['response_track']:
            if _rf['width'] == 0 or _rf['height'] == 0:
                if len(trajectory) > 0:
                    trajectory[_rf['frame_number']] = trajectory[max(trajectory.keys())]
                continue
            bbox = [
                int(_rf['x'] * scale_w),
                int(_rf['y'] * scale_h),
                int(_rf['width'] * scale_w),
                int(_rf['height'] * scale_h),
            ]
            bbox[2] = max(bbox[2], 1)
            bbox[3] = max(bbox[3], 1)
            # bbox_area = bbox[2] * bbox[3]
            # if bbox_area > 0:
            trajectory[_rf['frame_number']] = {'bbox': bbox}

        # remove frames with no bboxes
        inter_frames = set([img_id for img_id in inter_frames if img_id in trajectory])

        images_list_pims = [video_reader[e] for e in frame_ids]
        assert len(images_list_pims) == len(frame_ids)

        # TAKE CARE OF THE CASE WHEN THE RESPONSE WINDOW IS BEYOND 200 FRAMES
        # CURRENT VERSION WILL LOAD VIDEO FROM 0th FRAME

        images_list = np.array(images_list_pims)

        # images_list = np.random.randint(low=0, high=255, size=(len(frame_ids), video_reader.shape[1], video_reader.shape[2], 3), dtype=np.uint8)
        # import ipdb; ipdb.set_trace()

        # rescale trajectory to shorter side
        # import ipdb; ipdb.set_trace()

        # trajectory_scaled = {}
        # for k, v in trajectory.items():
        #     bbox_scaled = [
        #         int(v['bbox'][0] * scale_w),
        #         int(v['bbox'][1] * scale_h),
        #         int(v['bbox'][2] * scale_w),
        #         int(v['bbox'][3] * scale_h)
        #     ]
        #     trajectory_scaled[k] = {'bbox': bbox_scaled}
        # trajectory = trajectory_scaled

        # import ipdb; ipdb.set_trace()

        # prepare frame-level targets
        targets_list = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids):
            if img_id in inter_frames:
                anns = trajectory[
                    img_id
                ]  # dictionary with bbox [left, top, width, height] key
                anns = [anns]
                inter_idx.append(i_img)
            else:
                anns = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)

        # import ipdb; ipdb.set_trace()
        # video spatial transform
        if self._transforms is not None:
            images, targets = self._transforms(images_list, targets_list)
        else:
            images, targets = images_list, targets_list

        if flag_bg_segment:
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

        # reference crop
        # import ipdb; ipdb.set_trace()
        if not flag_bg_segment:
            visual_crop = video["visual_crop"]
            vc_fno = visual_crop["frame_number"]
            reference = video_reader[vc_fno]
            # print(f"Time to retrieve reference frame ({vc_fno}th) from video: {time.time() - start_time}")

            reference = torch.as_tensor(rearrange(reference, "h w c -> () c h w"))
            reference = reference.float()
            ref_bbox = (
                visual_crop["x"] * scale_w,
                visual_crop["y"] * scale_h,
                visual_crop["x"] * scale_w + visual_crop["width"] * scale_w,
                visual_crop["y"] * scale_h + visual_crop["height"] * scale_h,
            )
            reference = extract_window_with_context(
                image=reference,
                bbox=ref_bbox,
                p=16,  # default arg
                size=224,
                pad_value=125,  # default arg
            )
        else:
            # import ipdb; ipdb.set_trace()
            # augment reference crop from another video
            set_video_uids = set([e['metadata']['video_uid'] for e in self.annotations])
            set_video_uids.remove(self.annotations[idx]['metadata']['video_uid'])  # remove current video_uid
            video_uid_bg = random.sample(set_video_uids, k=1)[0]
            annotations_wrt_video_uid_bg = [e for e in self.annotations if e['metadata']['video_uid'] == video_uid_bg]
            video_bg = random.sample(annotations_wrt_video_uid_bg, k=1)[0]

            visual_crop_bg = video_bg["visual_crop"]
            vc_fno_bg = visual_crop_bg["frame_number"]
            clip_path_bg = os.path.join(
                self.vid_folder, get_clip_name_from_clip_uid(video_bg["clip_uid"])
            )
            video_reader_bg = pims.Video(clip_path_bg)

            h_bg = video_reader_bg.shape[1]
            w_bg = video_reader_bg.shape[2]

            h_orig_bg = video_bg['response_track'][0]['original_height']
            w_orig_bg = video_bg['response_track'][0]['original_width']

            scale_h_bg = h_bg / h_orig_bg
            scale_w_bg = w_bg / w_orig_bg

            reference_bg = video_reader_bg[vc_fno_bg]

            reference_bg = torch.as_tensor(rearrange(reference_bg, "h w c -> () c h w"))
            reference_bg = reference_bg.float()
            ref_bbox_bg = (
                visual_crop_bg["x"] * scale_w_bg,
                visual_crop_bg["y"] * scale_h_bg,
                visual_crop_bg["x"] * scale_w_bg + visual_crop_bg["width"] * scale_w_bg,
                visual_crop_bg["y"] * scale_h_bg + visual_crop_bg["height"] * scale_h_bg,
            )
            reference_bg = extract_window_with_context(
                image=reference_bg,
                bbox=ref_bbox_bg,
                p=16,  # default arg
                size=224,
                pad_value=125,  # default arg
            )
            reference = reference_bg

        reference = rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
        reference_orig = copy.deepcopy(reference)

        # import ipdb; ipdb.set_trace()
        if self._transforms_reference_crop is not None:
            reference, _ = self._transforms_reference_crop([reference], None)

        # video level annotations
        tmp_target = {
            "video_id": video_id,
            # "qtype": video["qtype"],
            "inter_idx": [inter_idx[0], inter_idx[-1]]
            if inter_idx
            else [
                -100,
                -100,
            ],  # start and end (included) indexes for the annotated moment
            "frames_id": frame_ids,
            "reference_crop": reference,
            "reference_crop_annotation": video["visual_crop"],
            "response_track_annotation": video['response_track'],
            "caption": caption,
            # "flag_bg_segment": flag_bg_segment,
            "segment_type_selected": "bg" if flag_bg_segment else "fg",
            "task_name": "vq2d",
        }
        if self.plot_pred:
            tmp_target['images_list_pims'] = images_list_pims
            tmp_target['reference_orig'] = reference_orig
        if self.stride:
            return images[:, :: self.stride], targets, tmp_target, images
        return images, targets, tmp_target


def build(image_set, args):
    data_dir = Path(args.data.vq2d.path)
    vid_dir = data_dir / 'clips_shorter_side_320'

    if args.test:
        ann_file = data_dir / "test_annot.json.gz"
        vid_dir = data_dir / 'clips_test_shorter_side_320'
    elif image_set == "val":
        ann_file = data_dir / "val_annot.json.gz"
    else:
        ann_file = data_dir / "train_annot.json.gz"
        vid_dir = data_dir / 'clips_train_shorter_side_320'

    dataset = VQ2DDatasetWrapper(
        vid_dir,
        ann_file,
        transforms=make_video_transforms(
            image_set, cautious=True, resolution=args.resolution
        ),
        transforms_reference_crop=make_video_transforms(
            'val', cautious=True, resolution=args.resolution
        ),
        is_train=image_set == "train",
        fps=args.fps,
        tmp_crop=args.tmp_crop and image_set == "train",
        tmp_loc=args.sted,
        args=args,
    )
    return dataset
