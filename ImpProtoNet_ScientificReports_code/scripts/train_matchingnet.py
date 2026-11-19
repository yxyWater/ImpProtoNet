"""Train one M3 full-context MatchingNet run on fixed patient episodes."""

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
from eyelid_fsl.matchingnet_training import (  # noqa: E402
    build_matchingnet_model,
    load_matchingnet_training_protocol,
)
from eyelid_fsl.training import (  # noqa: E402
    resolve_training_seed,
    set_reproducible_seed,
    train_episodic_run,
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
    parser.add_argument(
        "--learning-rate", type=float, choices=[0.0001, 0.0003], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--training-seed", type=int)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    protocol = load_matchingnet_training_protocol(
        args.training_config, args.episode_protocol
    )
    training_seed = resolve_training_seed(
        protocol, args.split_id, args.training_seed
    )
    set_reproducible_seed(training_seed)
    train_manifest = args.episodes_root / args.split_id / "train_episodes.jsonl.gz"
    validation_manifest = (
        args.episodes_root / args.split_id / "validation_episodes.jsonl.gz"
    )
    train_dataset = EpisodeTensorDataset(
        train_manifest,
        args.source_root,
        args.shot,
        training=True,
        augmentation_seed=training_seed,
        support_order_seed=protocol.support_order_seed,
    )
    validation_dataset = EpisodeTensorDataset(
        validation_manifest,
        args.source_root,
        args.shot,
        training=False,
        support_order_seed=protocol.support_order_seed,
    )
    train_loader = build_episode_dataloader(
        train_dataset, device=device, num_workers=args.num_workers
    )
    validation_loader = build_episode_dataloader(
        validation_dataset, device=device, num_workers=args.num_workers
    )
    model = build_matchingnet_model(protocol, pretrained=True)
    summary = train_episodic_run(
        model,
        train_loader,
        validation_loader,
        protocol,
        learning_rate=args.learning_rate,
        shot=args.shot,
        split_id=args.split_id,
        device=device,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
        training_seed=training_seed,
        model_id="M3",
        model_name="MatchingNetFCE",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
