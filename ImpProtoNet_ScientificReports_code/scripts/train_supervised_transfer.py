"""Train one M2 supervised ResNet18 encoder shared by 3-shot and 5-shot."""

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
from eyelid_fsl.data.episodes import load_split_image_manifest  # noqa: E402
from eyelid_fsl.m2_training import train_supervised_transfer_run  # noqa: E402
from eyelid_fsl.models import build_resnet18_encoder  # noqa: E402
from eyelid_fsl.supervised_transfer import (  # noqa: E402
    SupervisedTransferModel,
    load_supervised_transfer_protocol,
    resolve_m2_training_seed,
)
from eyelid_fsl.training import set_reproducible_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/class_splits/split_image_manifest.csv"),
    )
    parser.add_argument("--episodes-root", type=Path, default=Path("outputs/episodes"))
    parser.add_argument(
        "--split-id",
        choices=[f"split_{index:02d}" for index in range(1, 6)],
        required=True,
    )
    parser.add_argument(
        "--learning-rate", type=float, choices=[0.0001, 0.0003], required=True
    )
    parser.add_argument("--selected-c", type=float, choices=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/supervised_transfer.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    protocol = load_supervised_transfer_protocol(args.training_config)
    training_seed = resolve_m2_training_seed(
        protocol, args.split_id, args.training_seed
    )
    set_reproducible_seed(training_seed)
    pools = load_split_image_manifest(args.split_manifest)[
        (args.split_id, "train")
    ]
    validation_manifest = (
        args.episodes_root / args.split_id / "validation_episodes.jsonl.gz"
    )
    validation_loaders = {}
    for shot in (3, 5):
        dataset = EpisodeTensorDataset(
            validation_manifest, args.source_root, shot, training=False
        )
        validation_loaders[shot] = build_episode_dataloader(
            dataset, device=device, num_workers=args.num_workers
        )
    model = SupervisedTransferModel(
        build_resnet18_encoder(pretrained=True), number_of_classes=len(pools)
    )
    summary = train_supervised_transfer_run(
        model,
        pools,
        args.source_root,
        validation_loaders,
        protocol,
        learning_rate=args.learning_rate,
        split_id=args.split_id,
        device=device,
        output_dir=args.output_dir,
        selected_regularization_c=args.selected_c,
        resume_checkpoint=args.resume_checkpoint,
        training_seed=training_seed,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
