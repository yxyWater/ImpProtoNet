"""Evaluate M1 frozen ImageNet features on one fixed split and shot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyelid_fsl.data.episode_dataset import (  # noqa: E402
    EpisodeTensorDataset,
    build_episode_dataloader,
    resolve_device,
)
from eyelid_fsl.evaluation import EvaluationIdentity, evaluate_test_run  # noqa: E402
from eyelid_fsl.frozen_imagenet import (  # noqa: E402
    build_frozen_imagenet_model,
    imagenet_weights_metadata,
    load_frozen_imagenet_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, default=Path("outputs/episodes"))
    parser.add_argument(
        "--split-id",
        choices=[f"split_{index:02d}" for index in range(1, 6)],
        required=True,
    )
    parser.add_argument("--shot", type=int, choices=[3, 5], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("outputs/dataset_audit/patient_manifest.csv"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/frozen_imagenet_baseline.json"),
    )
    parser.add_argument(
        "--episode-protocol",
        type=Path,
        default=Path("configs/episode_protocol.json"),
    )
    parser.add_argument("--save-feature-tensors", action="store_true")
    parser.add_argument("--feature-shard-size", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_frozen_imagenet_protocol(args.model_config)
    if args.shot not in protocol.support_shots:
        raise ValueError("Shot is outside the approved M1 protocol")
    device = resolve_device(args.device)
    episode_manifest = args.episodes_root / args.split_id / "test_episodes.jsonl.gz"
    dataset = EpisodeTensorDataset(
        episode_manifest,
        args.source_root,
        args.shot,
        training=False,
    )
    loader = build_episode_dataloader(
        dataset,
        device=device,
        num_workers=args.num_workers,
    )
    model = build_frozen_imagenet_model(load_pretrained_weights=True)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("M1 must not contain trainable parameters")
    summary = evaluate_test_run(
        model,
        loader,
        device=device,
        output_dir=args.output_dir,
        checkpoint_path=None,
        episode_manifest_path=episode_manifest,
        mixed_precision=device.type == "cuda",
        identity=EvaluationIdentity(
            model_id=protocol.model_id,
            model_name=protocol.model_name,
            model_version=str(1),
            training_seed=None,
        ),
        checkpoint_metadata=imagenet_weights_metadata(),
        dataset_manifest_path=args.dataset_manifest,
        config_paths=(args.model_config, args.episode_protocol),
        project_root=PROJECT_ROOT,
        save_feature_tensors=args.save_feature_tensors,
        feature_shard_size=args.feature_shard_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
