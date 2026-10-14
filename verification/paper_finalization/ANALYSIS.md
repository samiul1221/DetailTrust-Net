# Final paper evidence review — 2026-10-14

The local B0 evaluation and the downloaded B5 evaluation cover the same 49 validation patients. Pairing was verified before averaging. All canonical MRI hashes match the B5 integrity record; reference geometry, mapping, and ET volumes match. Original complete mask-byte hashes were not retained, which limits retrospective mask identity verification.

| Endpoint | B0 Dice | B5 Dice | Difference, percentage points | Pointwise 95% patient bootstrap interval |
|---|---:|---:|---:|---:|
| WT | 0.8328 | 0.8526 | +1.98 | +0.87 to +3.07 |
| TC | 0.6324 | 0.6876 | +5.51 | +1.40 to +9.63 |
| ET | 0.6675 | 0.6736 | +0.61 | −4.64 to +4.26 |
| Mean WT/TC/ET | 0.7109 | 0.7379 | +2.70 | +0.33 to +4.99 |

The point estimates and the 20,000-resample paired bootstrap were independently recomputed from `baseline_vs_b5_per_patient.csv`, seed 20261014. The interval conditions on the selected weights. It is exploratory validation evidence, not an independent test or a training-seed interval.

WT and TC account for most of the observed difference; ET's small difference is uncertain. B5 has higher mean surface-point Dice and sensitivity, lower precision, and numerically higher mean HD95. ET distance aggregates use different valid-case counts (B0 46, B5 45). Four absent-ET references remain in the overlap averages, including B0's one both-empty ET case. No undefined distance was replaced with a fictitious finite value.

B5 finishes 4,000/4,500/1,500 optimizer updates. B0 uses the selected update-2,000 checkpoint from an available 2,999-update history, with no new training. B0 lacks the corruption augmentation used by B5. CPU float32 B0 inference differs from GPU FP16 B5 inference. This is a comparison of available checkpoints; it cannot isolate utility supervision or an architectural effect under matched training budgets.

Within B5, removing all detail contributions reduces mean Dice from 0.7379 to 0.7295. T1ce accounts for most of this conditional effect. Removing T2 produces a tiny mean increase whose patient interval includes zero. These are fixed-output interventions, not independent trained ablations or MRI modality removal.

The final learned gates score 0.7379 versus 0.7403 for the selected unit-gate Stage B model, a 0.24-point decrease. This negative result is in the abstract, Results, Discussion, and conclusion. No claim that utility supervision improves robustness is supported without B4/B5 and shuffled-target controls.

B5's full requested validation suite is complete: 15 modality subsets and eight local FLAIR corruptions, plus five correction-removal conditions, all with 49 patients. T1ce removal strongly reduces ET segmentation. The tested local corruptions have small mean effects but do not establish scanner robustness or a baseline-relative robustness gain. Gate diagnostics were measured on clean inputs only. All 49 sampled-voxel stability enumeration checks pass, with maximum discrepancy 1.91e−6; stability can preserve an incorrect label.

The paper has no result placeholders. Its ten figures show real measured outputs or the verified architecture/workflow. The baseline prediction displayed for subject 205 was rerun on CPU and checked against its saved full-volume Dice; displayed B5 reference and FLAIR planes match local data. All figures retain errors/outliers and use declared scales.

Static LaTeX checks, reference ordering, numerical audit, figure text bounds, embedded fonts, and ZIP CRC checks pass. No TeX engine is installed; network access could not fetch one and WSL access was denied. Manuscript compilation, final page count, and TeX page-layout warnings therefore remain unverified. Overleaf should use main.tex, pdfLaTeX, Normal compilation, and a clean rebuild.

Remaining research: matched corruption/budget baseline, frozen-candidate B4/B5 and A1, multiple training seeds, locked internal test, independent external data, and same-hardware timing. Remaining author checks: exact dataset mirror version/terms and venue requirements. None of those missing measurements is filled with invented evidence.
