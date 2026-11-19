"""Run the final Split 01 ImpProtoNet method with researcher-supplied inputs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL_SEEDS = {3: 20295239, 5: 20262717}


def command(args: argparse.Namespace) -> list[str]:
    common = [
        "--source-root", str(args.images.resolve()), "--episodes-root", str(args.episodes.resolve()),
        "--split-id", "split_01", "--output-dir", str(args.output.resolve()),
        "--device", args.device, "--num-workers", str(args.workers),
    ]
    if args.action == "pretrain":
        return [
            sys.executable, str(ROOT / "scripts/train_supervised_transfer.py"), *common,
            "--split-manifest", str(args.split_manifest.resolve()),
            "--learning-rate", "0.0001", "--training-seed", "20261004",
        ]

    model = [
        "--shot", str(args.shot), "--roi-mode", "none" if args.ablation == "fusion" else "central_soft_crop_85",
        "--attention-type", "none" if args.ablation == "attention" else "residual_cbam",
        "--pretrain-mode", "imagenet" if args.ablation == "pretraining" else "m2_frozen",
        "--reduction-ratio", "32", "--spatial-kernel-size", "7",
        "--loss-type", "focal", "--focal-gamma", "1.5", "--focal-alpha", "0.25",
        "--unfreeze-layer4",
    ]
    if args.ablation != "pretraining":
        if args.pretrained is None:
            raise ValueError("--pretrained is required unless supervised pretraining is ablated")
        model += ["--m2-checkpoint", str(args.pretrained.resolve())]
    if args.shot == 5:
        model += ["--learnable-temp"]
    if args.action == "evaluate":
        manifest = ["--test-manifest", str(args.test_manifest.resolve())] if args.test_manifest else []
        return [
            sys.executable, str(ROOT / "scripts/evaluate_impprotonet.py"),
            *common, *model, *manifest, "--checkpoint", str(args.checkpoint.resolve()), "--num-views", "4",
        ]

    sampling = "standard" if args.ablation == "sampling" else "patient_exposure_balanced"
    overrides = []
    if sampling == "patient_exposure_balanced":
        overrides += ["--train-manifest-override", str(args.episodes.resolve() / "split_01/train_balanced_episodes.jsonl.gz")]
    attention = "none" if args.ablation == "attention" else "residual_cbam"
    return [
        sys.executable, str(ROOT / "scripts/train_impprotonet.py"), *common, *model, *overrides,
        "--learning-rate", "0.0001",
        "--training-config", str(ROOT / "configs/final_training.json"),
        "--episode-protocol", str(ROOT / "configs/episode_protocol_p15.json"),
        "--training-seed", str(FINAL_SEEDS[args.shot]), "--m2-split-id", "split_01",
        "--attention-mode", attention, "--sampling-mode", sampling,
        "--color-jitter", "0", "--rotation-degrees", "0",
    ]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    actions = root.add_subparsers(dest="action", required=True)
    for action in ("pretrain", "train", "evaluate"):
        p = actions.add_parser(action)
        for flag in ("images", "episodes", "output"):
            p.add_argument(f"--{flag}", type=Path, required=True)
        p.add_argument("--device", default="auto")
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--print-command", action="store_true")
        if action == "pretrain":
            p.add_argument("--split-manifest", type=Path, required=True)
            continue
        p.add_argument("--shot", type=int, choices=(3, 5), required=True)
        p.add_argument("--pretrained", type=Path)
        p.add_argument("--ablation", choices=("attention", "pretraining", "sampling", "fusion"),
                       default=None)
        if action == "evaluate":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--test-manifest", type=Path)
    return root


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        argv = command(args)
    except ValueError as exc:
        cli.error(str(exc))
    if args.print_command:
        import json
        print(json.dumps(argv))
        return
    subprocess.run(argv, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
