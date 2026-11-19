#!/usr/bin/env python3
"""Evaluate ImpProtoNet and its component ablations.
Supports custom architecture params and TTA."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, sys, os
from pathlib import Path
import torch
from torch import nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyelid_fsl.data.episode_dataset import EpisodeTensorDataset, build_episode_dataloader, resolve_device
from eyelid_fsl.evaluation import EvaluationIdentity, evaluate_test_run, load_best_protonet_checkpoint
from eyelid_fsl.training import set_reproducible_seed
from eyelid_fsl.models import ProtoNet
from eyelid_fsl.models.focal_protonet import FocalLossProtoNet, FocalTempProtoNet


def build_model(args, m2_checkpoint_path, device):
    from eyelid_fsl.m2_training import load_best_m2_encoder_checkpoint

    def load_m2(encoder):
        load_best_m2_encoder_checkpoint(encoder, m2_checkpoint_path,
            expected_split_id=args.split_id, expected_training_seed=None,
            expected_learning_rate=0.0001)

    from eyelid_fsl.models.enhanced import build_enhanced_encoder
    encoder = build_enhanced_encoder(
        roi_mode=args.roi_mode, pretrain_mode=args.pretrain_mode,
        attention_mode="none" if args.attention_type == "none" else "residual_cbam",
        m2_checkpoint=load_m2 if args.pretrain_mode == "m2_frozen" else None,
        attention_type=args.attention_type, reduction_ratio=args.reduction_ratio)


    model_type = FocalTempProtoNet if args.learnable_temp else FocalLossProtoNet
    model = model_type(encoder, way=3, gamma=args.focal_gamma, alpha=args.focal_alpha)
    return model


def evaluate_tta_detailed(model, loader, device, num_views=2):
    model.to(device)
    model.eval()
    query_rows = []
    episode_rows = []
    global_class_name_map = {}
    global_class_category_map = {}
    with torch.no_grad():
        for batch in loader:
            si = batch["support_images"].to(device)
            qi = batch["query_images"].to(device)
            sl = batch["support_labels"].to(device)
            ql = batch["query_labels"].to(device)
            out0 = model(si, sl, qi, ql)
            probs = out0.probabilities
            if num_views >= 2:
                sf = torch.flip(si, dims=[-1])
                qf = torch.flip(qi, dims=[-1])
                out1 = model(sf, sl, qf, ql)
                probs = probs + out1.probabilities
            if num_views >= 3:
                sf2 = torch.flip(si, dims=[-2])
                qf2 = torch.flip(qi, dims=[-2])
                out2 = model(sf2, sl, qf2, ql)
                probs = probs + out2.probabilities
            if num_views >= 4:
                sf3 = torch.flip(si, dims=[-1,-2])
                qf3 = torch.flip(qi, dims=[-1,-2])
                out3 = model(sf3, sl, qf3, ql)
                probs = probs + out3.probabilities
            if num_views >= 5:
                import torchvision.transforms.functional as TF
                sf4 = TF.rotate(si, 5)
                qf4 = TF.rotate(qi, 5)
                out4 = model(sf4, sl, qf4, ql)
                probs = probs + out4.probabilities
            if num_views >= 6:
                sf5 = TF.rotate(si, -5)
                qf5 = TF.rotate(qi, -5)
                out5 = model(sf5, sl, qf5, ql)
                probs = probs + out5.probabilities
            probs = probs / num_views
            preds = probs.argmax(dim=-1)
            correct_batch = (preds == ql).sum().item()
            total_batch = ql.size(0)
            episode_accuracy = float((preds == ql).float().mean().item())
            episode_f1 = 0.0
            class_ids = list(batch["class_ids"])
            class_names = list(batch["class_names"])
            class_categories = list(batch["class_categories"])
            for class_index, class_id in enumerate(class_ids):
                global_class_name_map[class_id] = class_names[class_index]
                global_class_category_map[class_id] = class_categories[class_index]
            for class_index, class_id in enumerate(class_ids):
                class_mask = ql == class_index
                pred_mask = preds == class_index
                true_positive = int((class_mask & pred_mask).sum().item())
                false_positive = int((~class_mask & pred_mask).sum().item())
                false_negative = int((class_mask & ~pred_mask).sum().item())
                denominator = 2 * true_positive + false_positive + false_negative
                episode_f1 += 0.0 if denominator == 0 else (2 * true_positive / denominator)
            episode_f1 /= len(class_ids)
            episode_rows.append({
                "episode_id": str(batch["episode_id"]),
                "shot": int(batch["shot"]),
                "accuracy": episode_accuracy,
                "macro_f1": episode_f1,
            })
            for query_index in range(total_batch):
                target_label = int(ql[query_index].item())
                predicted_label = int(preds[query_index].item())
                metadata = batch["query_metadata"][query_index]
                query_rows.append({
                    "episode_id": str(batch["episode_id"]),
                    "split_id": str(batch["split_id"]),
                    "partition": str(batch["partition"]),
                    "shot": int(batch["shot"]),
                    "query_index": query_index,
                    "patient_id": str(metadata["patient_id"]),
                    "image_id": str(metadata["image_id"]),
                    "target_class_id": class_ids[target_label],
                    "target_class_name": class_names[target_label],
                    "predicted_class_id": class_ids[predicted_label],
                    "predicted_class_name": class_names[predicted_label],
                    "correct": bool(target_label == predicted_label),
                    "confidence": float(probs[query_index].max().item()),
                    "true_class_probability": float(probs[query_index, target_label].item()),
                    "class_ids": class_ids,
                    "probabilities": probs[query_index].cpu().tolist(),
                })
    all_class_ids = sorted({row["target_class_id"] for row in query_rows} | {row["predicted_class_id"] for row in query_rows})
    class_name_map = global_class_name_map
    class_category_map = global_class_category_map
    confusion = {}
    for row in query_rows:
        key = (row["target_class_id"], row["predicted_class_id"])
        confusion[key] = confusion.get(key, 0) + 1
    per_class_rows = []
    total_queries = len(query_rows)
    for class_id in all_class_ids:
        true_positive = confusion.get((class_id, class_id), 0)
        false_positive = sum(confusion.get((other_id, class_id), 0) for other_id in all_class_ids if other_id != class_id)
        false_negative = sum(confusion.get((class_id, other_id), 0) for other_id in all_class_ids if other_id != class_id)
        true_negative = total_queries - true_positive - false_positive - false_negative
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if 2 * true_positive + false_positive + false_negative else 0.0
        per_class_rows.append({
            "class_id": class_id,
            "class_name": class_name_map[class_id],
            "category": class_category_map.get(class_id, ""),
            "query_count": true_positive + false_negative,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
        })
    accuracy = sum(row["correct"] for row in query_rows) / total_queries if total_queries else 0.0
    macro_precision = sum(row["precision"] for row in per_class_rows) / len(per_class_rows) if per_class_rows else 0.0
    macro_sensitivity = sum(row["sensitivity"] for row in per_class_rows) / len(per_class_rows) if per_class_rows else 0.0
    macro_specificity = sum(row["specificity"] for row in per_class_rows) / len(per_class_rows) if per_class_rows else 0.0
    macro_f1 = sum(row["f1"] for row in per_class_rows) / len(per_class_rows) if per_class_rows else 0.0
    balanced_accuracy = macro_sensitivity
    episode_accuracies = [row["accuracy"] for row in episode_rows]
    episode_mean = float(np.mean(episode_accuracies)) if episode_accuracies else 0.0
    episode_std = float(np.std(episode_accuracies)) if episode_accuracies else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "episode_mean": episode_mean,
        "episode_std": episode_std,
        "query_rows": query_rows,
        "episode_rows": episode_rows,
        "per_class_rows": per_class_rows,
        "confusion": confusion,
        "class_name_map": class_name_map,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-id", default="split_01")
    p.add_argument("--shot", type=int, choices=[3,5], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--m2-checkpoint", type=Path,
        default=N/"runs/M2/split_01/lr_0.0001/best_checkpoint.pt")
    p.add_argument("--source-root", type=Path,
        default=N/"eyelid_source_root",
        help="Root containing the original eyelid image tree")
    p.add_argument("--episodes-root", type=Path,
        default=N/"outputs/episodes",
        help="Root containing split episode manifests")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--test-manifest", type=Path)
    p.add_argument("--training-seed", type=int, default=20261004)
    p.add_argument("--num-views", type=int, default=2, choices=[1,2,3,4,5,6])
    # Architecture params
    p.add_argument("--attention-type", choices=["residual_cbam", "none"], default="residual_cbam")
    p.add_argument("--roi-mode", default="central_soft_crop_85")
    p.add_argument("--pretrain-mode", default="m2_frozen")
    p.add_argument("--loss-type", choices=["focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--learnable-temp", action="store_true", default=False)
    p.add_argument("--reduction-ratio", type=int, default=32)
    p.add_argument("--spatial-kernel-size", type=int, default=7)
    p.add_argument("--unfreeze-layer4", action="store_true", default=False)
    args = p.parse_args()

    device = resolve_device(args.device)
    set_reproducible_seed(args.training_seed)
    model = build_model(args, args.m2_checkpoint, device)
    ckpt_meta = load_best_protonet_checkpoint(model, args.checkpoint,
        expected_split_id=args.split_id, expected_shot=args.shot,
        expected_training_seed=None, expected_model_id=None)

    episodes_root = args.episodes_root
    test_manifest = args.test_manifest or episodes_root / args.split_id / "test_episodes.jsonl.gz"
    source_root = args.source_root
    test_dataset = EpisodeTensorDataset(test_manifest, source_root, args.shot, training=False)
    loader = build_episode_dataloader(test_dataset, device=device, num_workers=args.num_workers)

    detailed = evaluate_tta_detailed(model, loader, device, args.num_views)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result = {
        "shot": args.shot, "split_id": args.split_id,
        "test_acc": detailed["accuracy"],
        "macro_precision": detailed["macro_precision"],
        "macro_sensitivity": detailed["macro_sensitivity"],
        "macro_specificity": detailed["macro_specificity"],
        "macro_f1": detailed["macro_f1"],
        "balanced_accuracy": detailed["balanced_accuracy"],
        "episode_mean": detailed["episode_mean"], "episode_std": detailed["episode_std"],
        "num_views": args.num_views,
        "attention_type": args.attention_type,
        "loss_type": args.loss_type,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "query_count": len(detailed["query_rows"]),
        "episode_count": len(detailed["episode_rows"]),
    }
    print(json.dumps(result, ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "test_summary.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(args.output_dir / "test_episode_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed["episode_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(detailed["episode_rows"])
    with open(args.output_dir / "test_per_class_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed["per_class_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(detailed["per_class_rows"])
    confusion_rows = []
    for actual_id in sorted(detailed["class_name_map"]):
        for predicted_id in sorted(detailed["class_name_map"]):
            confusion_rows.append({
                "actual_class_id": actual_id,
                "actual_class_name": detailed["class_name_map"][actual_id],
                "predicted_class_id": predicted_id,
                "predicted_class_name": detailed["class_name_map"][predicted_id],
                "count": detailed["confusion"].get((actual_id, predicted_id), 0),
            })
    with open(args.output_dir / "test_confusion_matrix.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(confusion_rows[0].keys()))
        writer.writeheader()
        writer.writerows(confusion_rows)
    with gzip.open(args.output_dir / "test_query_predictions.jsonl.gz", "wt") as f:
        for row in detailed["query_rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
