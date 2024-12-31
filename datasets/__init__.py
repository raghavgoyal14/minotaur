# from .vidstg import build as build_vidstg
# from .hcstvg import build as build_hcstvg
# from .vq2d import build as build_vq2d
# from .nlq import build as build_nlq
# from .mq import build as build_mq
# from .mq_slowfast_features import build as build_mq_slowfast_features
# from .mq_w_slowfast_features import build as build_mq_w_slowfast_features

# vidstg adapted
# from .vidstg_adapted import build as build_vidstg_adapted

# hcstvg adapted
# from .hcstvg_adapted import build as build_hcstvg_adapted

# unified
# from .vq2d_unified import build as build_vq2d_unified
from .nlq_unified import build as build_nlq_unified
from .mq_unified import build as build_mq_unified
from .vq2d_unified import build as build_vq2d_unified


# def build_dataset(dataset_file: str, image_set: str, args):
#     if dataset_file == "vidstg":
#         return build_vidstg(image_set, args)
#     if dataset_file == "vq2d":
#         return build_vq2d(image_set, args)
#     if dataset_file == "nlq":
#         return build_nlq(image_set, args)
#     if dataset_file == "mq":
#         return build_mq(image_set, args)
#     if dataset_file == "mq_slowfast_features":
#         return build_mq_slowfast_features(image_set, args)
#     if dataset_file == "mq_w_slowfast_features":
#         return build_mq_w_slowfast_features(image_set, args)
#     if dataset_file == "hcstvg":
#         return build_hcstvg(image_set, args)
#     if dataset_file == "vidstg_adapted":
#         return build_vidstg_adapted(image_set, args)
#     if dataset_file == "hcstvg_adapted":
#         return build_hcstvg_adapted(image_set, args)
#     raise ValueError(f"dataset {dataset_file} not supported")


def build_dataset_unified(dataset_file: str, image_set: str, args):
    # if dataset_file == "vidstg":
    #     return build_vidstg(image_set, args)
    if dataset_file == "vq2d":
        return build_vq2d_unified(image_set, args)
    if dataset_file == "nlq":
        return build_nlq_unified(image_set, args)
    if dataset_file == "mq":
        return build_mq_unified(image_set, args)
    # if dataset_file == "mq_slowfast_features":
    #     return build_mq_slowfast_features(image_set, args)
    # if dataset_file == "mq_w_slowfast_features":
    #     return build_mq_w_slowfast_features(image_set, args)
    # if dataset_file == "hcstvg":
    #     return build_hcstvg(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")
