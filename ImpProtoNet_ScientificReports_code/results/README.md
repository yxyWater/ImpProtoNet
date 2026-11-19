# Aggregate numerical results

| Manuscript item | Source data |
| --- | --- |
| Table 1, dataset composition | `table1_dataset_composition.csv` |
| Table 2, recognition comparison | `table2_recognition.csv` |
| Table 3, support-set robustness | `table3_support_stability.csv` |
| Table 4, ablation | `table4_ablation.csv` |
| Table 5, reliability | `table5_reliability.csv`, `table5_confidence_intervals.csv` |
| Supplementary Table S1, changes from a matched support baseline | `supp_table1_support_changes.csv` |
| Supplementary Table S2, fixed confidence thresholds | `supp_table2_confidence_thresholds.csv` |
| Figure 2, confusion matrices | `figure2_confusion_counts.csv` |
| Figure 3, risk--coverage curves | `figure3_risk_coverage.csv` |
| Supplementary Figure S1, training curves | `supp_figure1_convergence_3shot.csv`, `supp_figure1_convergence_5shot.csv` |
| Supplementary Figure S2, hyperparameters | `supp_figure2_hyperparameters.csv` |
| Supplementary Figure S3, reliability diagrams | `supp_figure3_reliability_bins.csv` |

Run `python render_results.py --output figures` from the package root to redraw the numerical figures. Figure 1 is the illustrative framework diagram.

The cohort table contains class-level counts transcribed from the submitted manuscript. Confusion matrices contain prediction counts, with true classes as rows. Recognition, support, calibration and risk values are stored as fractions, including ECE and IQR. Multiply by 100 to obtain percentages or percentage points as appropriate. Hyperparameter validation accuracy is already expressed in percent. IDs M0, M1, M2, M3, M5 and Ours correspond to ProtoNet, FrozenNet, TransferNet, MatchNet, RelationNet and ImpProtoNet.

Support-set robustness reports equal-weighted disease-combination summaries. Confidence intervals use the resampling units and procedures specified in the manuscript. AURC is computed from all prediction ranks. The risk--coverage CSV contains the displayed coverage levels.

The ablation table retains its archived full-model reference. Its 5-shot reference differs by one correct prediction from the main-comparison replay over 15,000 predictions. Both accuracy values round to 82.3% at the manuscript's displayed precision.

The hyperparameter CSV contains the 30 displayed observations. The figure presents selected candidate values around the main configuration. It does not establish that the reference configuration outperformed every value tested during development.
