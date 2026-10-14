# Review of the 14 October DetailTrust-Net Kaggle output

B5 training is complete. Evaluation is partial. The Kaggle job has ended; it is not still training.

Inspected dump: `C:\D\project\bt\output_dump\samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T004555Z`.

This review preserves the downloaded evidence. It does not change the model, experiment protocol, raw metrics, figures, or paper.

## Training and checkpoint

| Stage | Saved updates | Status | Selected update | Validation macro Dice |
|---|---:|---|---:|---:|
| A | 4000/4000 | completed | 2000 | 0.710497 |
| B | 4500/4500 | completed | 4500 | 0.740310 |
| C | 1500/1500 | completed | 1500 | 0.737904 |

All 10,000 update records are contiguous and their total losses are finite. Stage A was skipped on resume; B resumed at 3,929 and completed its remaining 571 updates; C then completed 1,500 updates.

The exported checkpoint is `detailtrust/best_detailtrust_model.pt` (68,094,847 bytes), selected at C update 1,500. Its SHA-256 matches the report: `c3e3cbeeef00104695546777d1f4033bb4f9694d4412f7adeeb6dc112e957aa5`.

Kaggle recorded a successful strict state-dictionary reload. Local hash verification passed; an independent local PyTorch model reload could not be performed because PyTorch is absent from the checked local Python environments. This audit does not assert a local reload.

## Why Requested experiments complete is False

The experiment status is `evaluation_partial`, not failed training. The validation evaluator reached its session time budget:

- Twelve of the fifteen nonempty modality subsets completed all 49 validation patients.
- T2+FLAIR (mask 12) completed 29/49 patients; 20 remain.
- T1+T2+FLAIR and T1ce+T2+FLAIR (masks 13 and 14) were not evaluated.
- All eight configured corruption conditions remain unrun: two local FLAIR noise severities and two local FLAIR resolution degradations, each with all modalities and with T1ce absent.
- The 101-patient test split remains sealed by configuration (`run_final_test=False`, `protocol_locked=False`). This is an explicit configuration choice, not just a timeout.
- B0-B4 and A1-A7 were not selected in this run. Registry entries are not completed experiments.

The reported 10h 18m includes approximately 1h 17m of preprocessing, checks, and profiling before the configured nine-hour experiment budget, plus export time. The nine-hour limit starts at experiment orchestration, not notebook startup. Saved A/B training durations include work from previous sessions and must not be added as the current session runtime.

## Dataset and interpretation

The saved custom patient split is train=218, validation=49, test=101; split seed=20261012. These are custom, not official challenge splits. All evaluated patient IDs belong to validation, with no duplicate patient-condition-region rows. The startup configuration prints 219 training subjects, but the persisted split actually contains 218; report the manifest count.

All numbers below describe the selected B5 checkpoint on validation patients. They are patient macro averages, not pooled voxel metrics or final test results. Patient standard deviations are not uncertainty across training seeds. Only seed 1729 was run.

## Clean validation: all four modalities

| Region/class | Dice mean +/- SD | IoU | Precision | Sensitivity | Specificity | HD95 mm (valid n) | Surface Dice, 1 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| WT | 0.8526 +/- 0.1052 | 0.7558 | 0.8817 | 0.8483 | 0.9992 | 17.511 (49) | 0.6408 |
| TC | 0.6876 +/- 0.2272 | 0.5637 | 0.7814 | 0.6950 | 0.9995 | 16.136 (49) | 0.5011 |
| ET | 0.6736 +/- 0.2745 | 0.5585 | 0.6737 | 0.7899 | 0.9995 | 14.059 (45) | 0.6510 |
| background | 0.9988 +/- 0.0009 | 0.9977 | 0.9985 | 0.9992 | 0.8483 | 0.160 (49) | 0.9822 |
| NCR_NET | 0.3770 +/- 0.2804 | 0.2732 | 0.6193 | 0.3760 | 0.9997 | 12.622 (48) | 0.3692 |
| edema | 0.7166 +/- 0.1643 | 0.5816 | 0.7100 | 0.7718 | 0.9985 | 16.246 (49) | 0.6387 |
| ET_class | 0.6736 +/- 0.2745 | 0.5585 | 0.6737 | 0.7899 | 0.9995 | 14.059 (45) | 0.6510 |

WT/TC/ET mean Dice: **0.737904**. Whole-tumour segmentation is stronger than core and enhancing-tumour segmentation; exclusive NCR/NET Dice is only 0.3770.

Four patients have no reference ET but receive ET predictions. Their Dice is zero; ET HD95 is undefined for these cases (45 valid, four undefined). These are false-positive ET cases, not evidence that four genuine enhancing tumours were missed. ET-positive patients with very low Dice also exist. The log-volume figure excludes zero-volume reference cases from its axis only; the aggregate metrics retain them.

| Patient with reference ET absent | Predicted ET voxels |
|---|---:|
| BraTS20_Training_278 | 18966 |
| BraTS20_Training_306 | 2492 |
| BraTS20_Training_263 | 3178 |
| BraTS20_Training_286 | 151 |

HD95 is variable, with distant-error outliers around 110 mm. Mean WT/TC/ET HD95 values are 17.51/16.14/14.06 mm; medians are approximately 7.21/7.14/3.74 mm. The qualitative example also shows isolated predicted components away from the reference tumour. Do not suppress these failures in a paper.

## What the detail and gate analyses support

- Disabling all corrections inside the same B5 model gives mean Dice 0.729463. Full B5 increases this by 0.844 percentage points. This is an internal branch-removal comparison, not a separately trained B0 baseline.
- Most of the improvement comes from T1ce corrections. Disabling T2 corrections produces a tiny increase in overall Dice: every branch is not automatically useful.
- Stage B with gates fixed to one selected 0.740310; Stage C selected 0.737904. Gate training reduced clean validation Dice by 0.241 percentage points. These are checkpoint-selection observations, not a controlled B4/B5 comparison or independent test evidence.
- Whole-brain mean gates remain close to 0.5 for all modalities. T1ce gates increase locally in tumour regions, so they are not exactly constant, but gate-versus-utility associations are weak or mixed. The gate is not a calibrated correctness probability.
- Missing modalities markedly reduce performance, especially ET when T1ce is unavailable. Neither superiority over a baseline nor clinical/scanner robustness has been established.
- All 49 numerical conditional-bound checks passed for 16 correction subsets on sampled voxels; maximum margin discrepancy 1.90734863e-06. Conditional stability can coexist with wrong predictions and depends on fixed features/corrections. It is not a medical correctness guarantee.

## Audit coverage and integrity

- Parsed all 56 JSON files with no parsing failure or nonstandard NaN/Infinity constants.
- Checked 6034 per-patient metric rows, 126 summary rows, 980 gate rows, and 980 contribution rows.
- Independently recomputed 1170 metric means and 5544 median/SD/quantile entries; maximum discrepancies 0.0 and 0.0. Valid/undefined counts and ranges were checked.
- Every CSV field matched its corresponding JSON source record; no split-ID or duplicate-row issues were found.
- Visually reviewed all 17 distinct PNG figures using contact sheets and inspected the qualitative segmentation figure separately. Decoded PNG metadata, parsed the 15 SVG counterparts, checked metadata of the 15 PDF counterparts, and checked the saved NPZ display arrays. Vector exports repeat the same plots; they are not 30 additional independent results.
- All 13 saved Kaggle unit checks passed, including anchoring, fusion, utility sign/detachment, gradients, checkpoint round trip, resume, validation-selection equivalence, and NIfTI handling.

The audit verifies consistency of stored evidence; it does not independently recompute segmentation predictions from source NIfTI volumes or retrain on a GPU.

## Figure-by-figure review

| Figure stem | Inspection and paper-use notes |
|---|---|
| B5_s1729_learning_rate | All three schedules reach their planned endpoints. Separate stage axes are appropriate. |
| B5_s1729_training_objectives | Actual raw per-patch losses, no smoothing. Stage C utility BCE is logged but absent from the plot: plotting expects utility while history uses utility_bce. Fix the plotting key before presenting the complete objective breakdown. |
| B5_s1729_validation_boundaries | HD95 and Surface Dice show patient points and undefined counts. Keep the large boundary outliers and ET NA count. |
| B5_s1729_validation_contributions | T1ce has the largest edits; signed contributions can be positive or negative. Magnitude does not establish benefit. |
| B5_s1729_validation_detail_removal | Paired changes include harmful corrections. Describe this as within-B5 removal, not baseline training. |
| B5_s1729_validation_detection | Precision and sensitivity are voxel measures. ET sensitivity has n=45, while precision includes all 49. |
| B5_s1729_validation_et_size | Real ET-positive patients on a log-volume axis. Report four ET-absent false-positive cases separately; no fitted trend or significance is shown. |
| B5_s1729_validation_gate_utility | Brain means cluster near 0.5; utility is measured CE change. Do not infer calibrated quality or strong correlation. |
| B5_s1729_validation_missing_modalities | Real partial availability results with n per cell. Gray is not_run. Mask12 uses only29 patients and is not a matched49-patient comparison. |
| B5_s1729_validation_overlap | Patient Dice/IoU distributions reveal weaker TC/ET and low-score outliers. |
| B5_s1729_validation_stability | High stability includes incorrect voxels and is dominated by background. Repaired and introduced errors must both be discussed. |
| B5_s1729_validation_trajectory | Shows all stages including completed StageC. Selection scores do not establish test generalization. |
| BraTS20_Training_205_et_contribution_maps | Real signed ET-logit contributions on a common diverging scale. Mostly white maps are low contribution, not absent modality. MRI overlay would improve readability. |
| BraTS20_Training_205_segmentation_views | Real prespecified validation patient205, orthogonal FLAIR/reference/base/final panels and explicit legend. Base is the same B5 low-frequency component, not an independently trained B0. Distant false positives remain visible. |
| BraTS20_Training_205_spatial_gates | Real gate maps on0..1scale. Zero/purple margins outside the inference crop are reconstruction fill, not learned low trust. Clarify the caption; an MRI overlay would help. |
| example_corrections | Debug figure: tab10 label palette differs from the publication figure and lacks an explicit class legend. Prefer the dedicated qualitative figure. |
| preprocessing | Actual normalized training example, not test output. Add the class legend and use a consistent segmentation palette if reused in the paper. |

Fifteen publication candidates have PNG/PDF/SVG exports; the two debug figures are PNG only. Real assets can replace training-curve, dataset-example, qualitative, and partial modality-result placeholders with correct captions. No completed baseline-versus-B5, corruption, or held-out-test figures exist in this dump. Architecture diagrams do not depend on training outputs.

## What to do next

Run the existing continuation command to restore this latest dump and finish outstanding validation conditions:

```powershell
.\kaggle.ps1 --mode main --run
```

With the same configuration and restored checkpoints, all A/B/C stages have zero training updates remaining. The evaluator can restore completed patient-condition records and finish remaining cases. Do not use --fresh. Inspect the resume logs to confirm the latest 4,000/4,500/1,500 progress is restored.

This default command does not enable the sealed test set. Final test results require explicitly setting run_final_test=True and protocol_locked=True after fixing the main protocol. Finishing validation may make the requested-experiments flag True even with the test intentionally not_run, because that is how the orchestrator defines completion.

Before submitting a paper, complete evaluation and a fair baseline comparison, retain the negative gating observation, and update manuscript status/figures. The earlier partially trained B0 cannot be called a completed, budget-matched baseline without verifying its final checkpoint and evaluation. A B4 or shuffled-target comparison is needed to establish the specific utility-supervision claim.

## Complete JSON inventory

Every file below was parsed and inventoried; figure sidecars were checked against their plotted/source data where applicable. File-level details, all CSV cross-check counts, training continuity checks, and gate summaries are in audit.json.

- `detailtrust\ablation_results.json` (parsed)
- `detailtrust\artifact_manifest.json` (parsed)
- `detailtrust\artifact_verification.json` (parsed)
- `detailtrust\best_detailtrust_model_config.json` (parsed)
- `detailtrust\compute_profile.json` (parsed)
- `detailtrust\dataset_discovery.json` (parsed)
- `detailtrust\dataset_evidence.json` (parsed)
- `detailtrust\dataset_integrity.json` (parsed)
- `detailtrust\detailtrust_report.json` (parsed)
- `detailtrust\environment.json` (parsed)
- `detailtrust\experiment_config.json` (parsed)
- `detailtrust\experiments\B5_s1729\inference_benchmark.json` (parsed)
- `detailtrust\experiments\B5_s1729\refinement\history.json` (parsed)
- `detailtrust\experiments\B5_s1729\refinement\stage_result.json` (parsed)
- `detailtrust\experiments\B5_s1729\result.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m01.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m02.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m03.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m04.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m05.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m06.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m07.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m08.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m09.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m10.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m11.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m12.json` (parsed)
- `detailtrust\experiments\B5_s1729\validation\clean_m15.json` (parsed)
- `detailtrust\experiments\shared_candidates_s1729\A\history.json` (parsed)
- `detailtrust\experiments\shared_candidates_s1729\A\stage_result.json` (parsed)
- `detailtrust\experiments\shared_candidates_s1729\B\history.json` (parsed)
- `detailtrust\experiments\shared_candidates_s1729\B\stage_result.json` (parsed)
- `detailtrust\figures\B5_s1729_learning_rate.json` (parsed)
- `detailtrust\figures\B5_s1729_training_objectives.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_boundaries.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_contributions.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_detail_removal.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_detection.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_et_size.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_gate_utility.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_missing_modalities.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_overlap.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_stability.json` (parsed)
- `detailtrust\figures\B5_s1729_validation_trajectory.json` (parsed)
- `detailtrust\figures\BraTS20_Training_205_et_contribution_maps.json` (parsed)
- `detailtrust\figures\BraTS20_Training_205_segmentation_views.json` (parsed)
- `detailtrust\figures\BraTS20_Training_205_spatial_gates.json` (parsed)
- `detailtrust\figures\paper_figures_manifest.json` (parsed)
- `detailtrust\figures\qualitative_manifest.json` (parsed)
- `detailtrust\resume_restore.json` (parsed)
- `detailtrust\split_manifest.json` (parsed)
- `detailtrust\stability_analysis.json` (parsed)
- `detailtrust\tiny_overfit_check.json` (parsed)
- `detailtrust\training_history.json` (parsed)
- `detailtrust\unit_checks.json` (parsed)
- `download_manifest.json` (parsed)
