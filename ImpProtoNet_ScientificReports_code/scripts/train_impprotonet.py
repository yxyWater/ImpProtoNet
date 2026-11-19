#!/usr/bin/env python3
"""Train the manuscript ImpProtoNet model and component ablations."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyelid_fsl.data.episode_dataset import (
    EpisodeTensorDataset,
    build_episode_dataloader,
    resolve_device,
)
from eyelid_fsl.enhanced_protocol import load_enhanced_spec
from eyelid_fsl.m2_training import load_best_m2_encoder_checkpoint
from eyelid_fsl.training import (
    load_protonet_training_protocol,
    resolve_training_seed,
    set_reproducible_seed,
    train_protonet_run,
)
from eyelid_fsl.training import collect_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--split-id", default="split_01")
    parser.add_argument("--shot", type=int, choices=[3, 5], required=True)
    parser.add_argument("--learning-rate", type=float, choices=[0.00005, 0.0001, 0.0003, 0.0005, 0.001], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--episode-protocol", type=Path, required=True)
    parser.add_argument("--roi-mode", choices=[
        "none", "central_soft_crop", "central_soft_crop_95",
        "central_soft_crop_98", "central_soft_crop_99",
        "central_soft_crop_90", "central_soft_crop_85",
        "central_soft_crop_80", "central_soft_crop_75", "dual_view",
    ], default="none")
    parser.add_argument("--pretrain-mode", choices=["imagenet", "m2_frozen"], default="m2_frozen")
    parser.add_argument("--attention-type", choices=["residual_cbam", "none"], default="residual_cbam")
    parser.add_argument("--attention-mode", default="residual_cbam", choices=["residual_cbam", "none"])
    parser.add_argument("--reduction-ratio", type=int, default=32)
    parser.add_argument("--spatial-kernel-size", type=int, default=7)
    parser.add_argument("--sampling-mode", choices=["standard", "patient_exposure_balanced"], default="standard")
    parser.add_argument("--sampling-variant", default="standard")
    parser.add_argument("--learnable-temp", action="store_true", default=False)
    parser.add_argument("--unfreeze-layer4", action="store_true", default=False)
    parser.add_argument("--m2-checkpoint", type=Path)
    parser.add_argument("--train-manifest-override", type=Path)
    parser.add_argument("--m2-learning-rate", type=float, default=0.0001)
    parser.add_argument("--m2-split-id", default=None, help="Override split_id for M2 checkpoint validation (for split reuse)")
    parser.add_argument("--training-seed", type=int, default=20261004)
    parser.add_argument("--loss-type", choices=["focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--color-jitter", type=float, default=0.0, help="Color jitter strength (0=off, 0.3=moderate)")
    parser.add_argument("--rotation-degrees", type=float, default=0.0, help="Max rotation degrees (0=off, 10=moderate)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    spec = load_enhanced_spec(
        args.training_config, args.episode_protocol,
        roi_mode=args.roi_mode,
        pretrain_mode=args.pretrain_mode,
        attention_mode="residual_cbam",
        sampling_mode=args.sampling_mode,
        sampling_variant=args.sampling_variant,
    )
    protocol = load_protonet_training_protocol(args.training_config, args.episode_protocol)
    seed = resolve_training_seed(protocol, args.split_id, args.training_seed)
    set_reproducible_seed(seed)

    partition = "train_balanced" if args.sampling_mode != "standard" else "train"
    train_manifest = (
        args.train_manifest_override
        if args.train_manifest_override is not None
        else args.episodes_root / args.split_id / f"{partition}_episodes.jsonl.gz"
    )
    validation_manifest = args.episodes_root / args.split_id / "validation_episodes.jsonl.gz"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing training manifest: {train_manifest}")

    train_dataset = EpisodeTensorDataset(train_manifest, args.source_root, args.shot, training=True, augmentation_seed=seed, color_jitter_strength=args.color_jitter, rotation_degrees=args.rotation_degrees)
    validation_dataset = EpisodeTensorDataset(validation_manifest, args.source_root, args.shot, training=False)
    train_loader = build_episode_dataloader(train_dataset, device=device, num_workers=args.num_workers)
    validation_loader = build_episode_dataloader(validation_dataset, device=device, num_workers=args.num_workers)

    def load_m2(encoder) -> None:
        if args.m2_checkpoint is None:
            raise ValueError("--m2-checkpoint is required with --pretrain-mode m2_frozen")
        load_best_m2_encoder_checkpoint(
            encoder, args.m2_checkpoint,
            expected_split_id=args.m2_split_id or args.split_id,
            expected_training_seed=20261004,
            expected_learning_rate=args.m2_learning_rate,
        )

    from eyelid_fsl.models.enhanced import build_enhanced_encoder
    encoder = build_enhanced_encoder(
        roi_mode=args.roi_mode,
        pretrain_mode=args.pretrain_mode,
        attention_mode=args.attention_mode,
        m2_checkpoint=load_m2 if args.pretrain_mode == "m2_frozen" else None,
        attention_type=args.attention_type,
        reduction_ratio=args.reduction_ratio,
        spatial_kernel_size=args.spatial_kernel_size,
    )
    if args.pretrain_mode == "m2_frozen" and args.unfreeze_layer4:
        for name, param in encoder.named_parameters():
            param.requires_grad_(True)
        # Remove instance-level train override so class default is used
        if 'train' in encoder.__dict__:
            del encoder.__dict__['train']


    from eyelid_fsl.models.focal_protonet import FocalLossProtoNet, FocalTempProtoNet
    model_type = FocalTempProtoNet if args.learnable_temp else FocalLossProtoNet
    model = model_type(encoder, way=3, gamma=args.focal_gamma, alpha=args.focal_alpha)

    environment = collect_environment(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps({
            "split_id": args.split_id, "shot": args.shot,
            "learning_rate": args.learning_rate, "training_seed": seed,
            "attention_type": args.attention_type,
            "reduction_ratio": args.reduction_ratio,
            "spatial_kernel_size": args.spatial_kernel_size,
            "roi_mode": args.roi_mode, "pretrain_mode": args.pretrain_mode,
            "sampling_mode": args.sampling_mode,
            "loss_type": args.loss_type,
            "learnable_temp": args.learnable_temp,
            "unfreeze_layer4": args.unfreeze_layer4,
            "model": "ImpProtoNet",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = train_protonet_run(
        model, train_loader, validation_loader,
        protocol,
        learning_rate=args.learning_rate,
        shot=args.shot,
        split_id=args.split_id,
        device=device,
        output_dir=args.output_dir,
        training_seed=seed,
        model_id=spec.layer_id(),
        model_name=spec.model_name,
        experiment_metadata=spec.experiment_metadata(),
    )
    print(json.dumps({
        "acc": summary.get("best_validation", {}).get("mean_accuracy", "?"),
        "macro_f1": summary.get("best_validation", {}).get("mean_macro_f1", "?"),
        "attention_type": args.attention_type, "shot": args.shot,
        "seed": seed, "lr": args.learning_rate,
        "loss_type": args.loss_type,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
