# from .tubedetr import build


# def build_model(args):
#     return build(args)


def build_model_unified(args):
    # backbone
    if args.model.name.backbone == "backbone":
        from .backbone import build_backbone
        backbone = build_backbone(args)
    else:
        raise ValueError(f"Invalid backbone name: {args.model.name.backbone}")

    # transformer
    if args.model.name.transformer == "transformer_unified":
        from .transformer_unified import build_transformer
        transformer = build_transformer(args)
    else:
        raise ValueError(f"Invalid transformer name: {args.model.name.transformer}")

    # tubedetr
    if args.model.name.tubedetr == "tubedetr_unified":
        from .tubedetr_unified import build_tubedetr
    elif args.model.name.tubedetr == "tubedetr_unified_single_DL":
        from .tubedetr_unified_single_DL import build_tubedetr
    else:
        raise ValueError(f"Invalid tubedetr name: {args.model.name.tubedetr}")

    return build_tubedetr(args, backbone, transformer)

