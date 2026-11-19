# ImpProtoNet

Source code for the fixed disease-category split used in the manuscript. The commands below run supervised pretraining, episodic training and evaluation, and the analyses underlying the main and supplementary results.

## Reading the code

Start with `run_impprotonet.py`. It resolves the final 3-shot and 5-shot settings, training seeds and four component ablations.

| Manuscript component | Implementation |
| --- | --- |
| Supervised pretraining | `scripts/train_supervised_transfer.py`, `src/eyelid_fsl/m2_training.py` |
| Patient- and image-balanced sampling | `src/eyelid_fsl/sampling.py` |
| Full-image and central-crop fusion | `src/eyelid_fsl/models/enhanced.py` |
| Channel and spatial attention | `src/eyelid_fsl/models/attention.py` |
| Prototypes and squared Euclidean distances | `src/eyelid_fsl/models/protonet.py` |
| Query focal loss and learned temperature | `src/eyelid_fsl/models/focal_protonet.py` |
| Episodic training | `scripts/train_impprotonet.py`, `src/eyelid_fsl/training.py` |
| Four-view probability averaging | `scripts/evaluate_impprotonet.py` |
| Statistical analyses | `paper_analysis.py` |

## Included numerical results

The `results/` directory contains aggregate source data for Tables 1--5, Supplementary Tables S1--S2 and the numerical figures. See `results/README.md` for file names, units and interpretation.

```sh
python render_results.py --output figures
```

This command redraws the confusion matrices, risk--coverage curves, reliability diagrams, training curves and hyperparameter panels from the included CSV files. Plot appearance can vary with installed fonts. Recalculating patient-level statistics from predictions requires the corresponding study inputs described below.

## Setup

Use Python 3.10 or later. Install a compatible PyTorch and torchvision pair for your device before installing the remaining dependencies.

```sh
python -m pip install -r requirements.txt
```

For access to the study data, see the manuscript's Data Availability statement. Exact replay of the reported numbers requires the original images, patient-linked manifests and validation-selected checkpoints. Independent datasets can use their own filenames and identifiers. Repeated training or inference on a different device can produce numerical differences.

## Inputs

- `IMAGES` is the image root. Episode samples refer to images through `source_relpath`.
- `SPLITS.csv` is a split image manifest for the pretraining and episode generator. The fixed `split_01` uses K00, K01, K02, K03, K04, K05, K09, K10, K13, K14, K15 and K17 for training, K07, K16, K18, K19 and K21 for validation, and K06, K08, K11, K12 and K20 for testing.
- `EPISODES/split_01` holds `train_episodes.jsonl.gz`, `train_balanced_episodes.jsonl.gz`, `validation_episodes.jsonl.gz` and `test_episodes.jsonl.gz`. Evaluation uses the same fixed manifest for all methods within a shot setting.
- The model commands accept PyTorch-compatible checkpoints. The pretraining entry produces a compatible `best_checkpoint.pt`. Episodic evaluation expects the architecture that generated the checkpoint and checks the model state strictly.

Use `python scripts/generate_episodes.py --help` to generate episodes from a supplied split manifest. To construct a balanced training manifest, put `split_image_manifest.csv` in the `class_splits/` directory next to the episodes root and run:

```sh
PYTHONPATH=src python -m eyelid_fsl.sampling \
  --source EPISODES/split_01/train_episodes.jsonl.gz \
  --destination EPISODES/split_01/train_balanced_episodes.jsonl.gz
```

## Train and evaluate ImpProtoNet

```sh
python run_impprotonet.py pretrain --images IMAGES --episodes EPISODES \
  --split-manifest SPLITS.csv --output runs/pretrained
python run_impprotonet.py train --shot 3 --images IMAGES --episodes EPISODES \
  --pretrained runs/pretrained/best_checkpoint.pt --output runs/imp3
python run_impprotonet.py evaluate --shot 3 --images IMAGES --episodes EPISODES \
  --pretrained runs/pretrained/best_checkpoint.pt --checkpoint runs/imp3/best_checkpoint.pt --output eval/imp3
```

Repeat training and evaluation with `--shot 5` and separate output directories. The entry point uses the corresponding recorded training seed and the learned temperature for 5-shot. The episodic stage uses a constant learning rate of 0.0001, weight decay of 0.001 and focal-loss parameters alpha 0.25 and gamma 1.5. Evaluation averages the class probabilities of the original, horizontally flipped, vertically flipped and jointly flipped views. Use matching `--ablation attention|pretraining|sampling|fusion` options at training and evaluation. Add `--print-command` to inspect resolved arguments.

Baseline entry points are the `train_*.py` and `evaluate_*.py` scripts in `scripts/`. FrozenNet uses the ImageNet-pretrained frozen encoder and its evaluator. Use `--help` for their checkpoint and manifest arguments. Use the fixed study checkpoint for each model when replaying the reported comparison.

## Recalculate figures and statistics

`paper_analysis.py` accepts a prediction tree with paths `PREDICTIONS/MODEL/split_01/SHOTshot/test_query_predictions.jsonl.gz`. Use model IDs `M0`, `M1`, `M2`, `M3`, `M5` and `Ours` for ProtoNet, FrozenNet, TransferNet, MatchNet, RelationNet and ImpProtoNet. Use `A`, `P`, `S`, `R` and `Ours` for the ablation analysis. Supply the corresponding fixed manifest for identity checks.

```sh
python paper_analysis.py recognition --predictions PREDICTIONS --manifest TEST.jsonl.gz --output analysis/recognition
python paper_analysis.py recognition --ablation --predictions ABLATION_PREDICTIONS --manifest TEST.jsonl.gz --output analysis/ablation
python paper_analysis.py support --predictions MATCHED_PREDICTIONS --manifest MATCHED.jsonl.gz --output analysis/support
python paper_analysis.py reliability --predictions PREDICTIONS --manifest TEST.jsonl.gz --output analysis/reliability
python plot_selected_hyperparameters.py --input POINTS.csv --output super-parameters.pdf
```

The support analysis uses predictions for the matched-query baseline and all 100 support replacements per disease combination. Baseline predictions can be supplied in a separate `matched_baseline_predictions.jsonl.gz` alongside the corresponding predictions. Prediction rows include query position, patient and image identifiers, true and predicted class IDs, episode ID, split and shot. To evaluate ImpProtoNet on a matched support manifest, pass it to `evaluate --test-manifest`. For baseline evaluators, provide the matched manifest as `EPISODES/split_01/test_episodes.jsonl.gz` in a separate episode root. Use `python scripts/generate_support_sensitivity_episodes.py --help` for manifest generation.

The hyperparameter plotting input uses the four CSV columns `panel,shot,parameter_value,validation_accuracy_percent`. Panel identifiers are `learning_rate`, `weight_decay` and `focal_gamma`. Supply five parameter values in each panel, with a row per value and shot. The displayed settings are a selected comparison with the reference configuration.

`scripts/convergence_plots.py` plots learning curves from a completed training run. `scripts/patient_demographics.py` calculates the cohort summary with authorized clinical metadata.
