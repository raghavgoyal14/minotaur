# Code for Minotaur VQ2D


## Prerequisites
- Please follow requirements from [TubeDETR](https://github.com/antoyang/TubeDETR), and add packages as you need them in the code such as `pims`, `seaborn`, etc.
- Please clone the official [VQ2D](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D) repo as it contains eval metrics and preprocessing scripts (maybe outside this repo --  not sure if this matters)
- Download MDETR weights of [R101](https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1) taken from [MDETR repo](https://github.com/ashkamath/mdetr?tab=readme-ov-file#pre-training). This is used for initialization.

## Dataset (VQ2D)

**Note**
- The dataset may have changed, so please look at the official instructions in the [VQ2D](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D) repo. Below I provided the commands I used quite a while ago.
- Also, after getting `clips_train` and `clips_test`, I post-process them to a shorter side size of 320 to obtain `clips_train_shorter_side_320` and `clips_test_shorter_side_320`. This helps with faster data loading and not overloading the memory. (I cannot seem to locate the command for this)


#### train
`python convert_videos_to_clips.py --annot-paths data/vq_train.json --save-root data/clips_train --ego4d-videos-root data/v1/full_scale --num-workers 10`

#### test
`python convert_videos_to_clips.py --annot-paths data/vq_test_annotated.json --save-root data/clips_test --ego4d-videos-root data/v1/full_scale --num-workers 30`

#### process annotation files
`python process_vq_dataset.py --annot-root /private/home/raghavgoyal/data/vq2d_root/data/safe --save-root /private/home/raghavgoyal/data/vq2d_root/data/safe`



## Setup paths
- `main_unified_single_data_loader.py`
    - Line 52: change the default path (`config_path`) to your clone
- `datasets/vq2d_orig_eval.py`
    - Change the path in line 13 (`sys.path.insert`) to your clone of [VQ2D](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D)
    - numpy no longer has `np.float`, so please change the occurrences of `np.float` to `float` in files [spatio_temporal_metrics.py](https://github.com/EGO4D/episodic-memory/blob/main/VQ2D/vq2d/metrics/spatio_temporal_metrics.py) and [temporal_metrics.py](https://github.com/EGO4D/episodic-memory/blob/main/VQ2D/vq2d/metrics/temporal_metrics.py) of your cloned repo
- `config/all_tasks/default.yaml`
    - Line 1 point the path of `_BASE_` to your clone
- `config/all_tasks/BASE.yaml`
    - Path to your copy of VQ2D dataset


## Train
```
python main_unified_single_data_loader.py load=path-to-mdetr-checkpoint/pretrained_resnet101_checkpoint.pth tasks.names="['vq2d']" eval=False epochs=25 eval_skip=1 num_workers=0
```


## Eval
For evaluation
- Set the `eval=False`
- Set the checkpoint to evaluate using `resume=<path-to-ckpt>`


## Debug
You can also debug the train/eval pipeline using `debug=True`. It will subsample the dataset to facilicate debugging.


## Final Note
- There are a bunch of comments in the scripts which are mainly concerned with adding more datasets into the mix. Since you only need VQ2D, I commented out the others. If you want to include more datasets and need any help, let me know


## Paper
https://arxiv.org/abs/2302.08063