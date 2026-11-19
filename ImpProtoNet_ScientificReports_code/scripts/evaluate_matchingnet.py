"""Evaluate M3 full-context MatchingNet on one fixed split and shot."""

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
from eyelid_fsl.matchingnet_training import (  # noqa: E402
    build_matchingnet_model,
    load_best_matchingnet_checkpoint,
    load_matchingnet_training_protocol,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--learning-rate", type=float, choices=[0.0001, 0.0003], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--training-seed", type=int, default=20261004)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("outputs/dataset_audit/patient_manifest.csv"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/matchingnet_training.json"),
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
    protocol = load_matchingnet_training_protocol(
        args.training_config, args.episode_protocol
    )
    device = resolve_device(args.device)
    manifest_path = args.episodes_root / args.split_id / "test_episodes.jsonl.gz"
    dataset = EpisodeTensorDataset(
        manifest_path,
        args.source_root,
        args.shot,
        training=False,
        support_order_seed=protocol.support_order_seed,
    )
    loader = build_episode_dataloader(
        dataset, device=device, num_workers=args.num_workers
    )
    model = build_matchingnet_model(protocol, pretrained=False)
    checkpoint = load_best_matchingnet_checkpoint(
        model,
        args.checkpoint,
        expected_split_id=args.split_id,
        expected_shot=args.shot,
        expected_training_seed=args.training_seed,
        expected_learning_rate=args.learning_rate,
    )
    checkpoint_metadata = dict(checkpoint)
    checkpoint_metadata.update(
        {
            "classification_rule": "cosine_attention_over_support_labels",
            "matching_processing_steps": protocol.processing_steps,
            "matching_support_order_seed": protocol.support_order_seed,
        }
    )
    summary = evaluate_test_run(
        model,
        loader,
        device=device,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        episode_manifest_path=manifest_path,
        mixed_precision=device.type == "cuda",
        identity=EvaluationIdentity(
            model_id="M3",
            model_name="MatchingNetFCE",
            model_version="1",
            training_seed=args.training_seed,
        ),
        checkpoint_metadata=checkpoint_metadata,
        dataset_manifest_path=args.dataset_manifest,
        config_paths=(args.training_config, args.episode_protocol),
        project_root=PROJECT_ROOT,
        save_feature_tensors=args.save_feature_tensors,
        feature_shard_size=args.feature_shard_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
