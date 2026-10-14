# Completed B5 validation run: audit and verdict

Source: `C:\D\project\bt\output_dump\samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z`. Raw outputs were not modified.

**B5 training and its configured validation evaluation are complete. Final test evaluation and independent baseline experiments are not complete.**

## Evidence integrity

Read and SHA-256 hashed all 132 files: {'.json': 67, '.pt': 7, '.csv': 4, '.pdf': 16, '.png': 18, '.svg': 16, '.npz': 1, '.md': 2, '.log': 1}.
All seven PyTorch checkpoint containers passed ZIP-payload CRC checks without pickle deserialization. The exported best-model SHA-256 matches the report. Kaggle records a strict model reload; local PyTorch reload was not performed.
All 67 JSON files parsed. All four CSV tables were checked, including 9,604 patient-condition-class/region rows, 196 summary rows, 980 gate rows and 980 correction rows. Independently recomputed 1820 means and 8624 other summary statistics: zero discrepancy. No invalid ranges, duplicate records or split-membership issues were found.
All 18 distinct PNG figures were visually reviewed. Their 16 PDF and 16 SVG counterparts and the NPZ display-source arrays were also checked. All 13 saved unit checks passed. Full inventory: all_file_inventory.json; metric audit: audit.json.

## Completion

A 4,000/4,000; B 4,500/4,500; C 1,500/1,500. This session restored all stages and performed zero remaining research-training updates. The checkpoint hash is unchanged from the preceding dump.
All 15 nonempty modality subsets and all eight validation corruptions completed all 49 patients. Clean component-removal analyses also completed all 49. Test status is not_run because run_final_test=False and protocol_locked=False. B0-B4 and A1-A7 remain not_run because they were not selected.
Requested experiments complete=True follows this definition: selected B5 validation complete, with test allowed to remain intentionally not_run. It does not mean the full research protocol or ablation study is complete.

## Clean segmentation results

All metrics are patient macro values on a custom 49-patient validation split, selected using that validation cohort. Train=218; test=101; split seed=20261012. These are not official challenge splits. One model seed, 1729.

| Region | Dice mean +/- SD | IoU | Precision | Sensitivity | HD95 mm (valid n) | Surface Dice at 1 mm |
|---|---:|---:|---:|---:|---:|---:|
| WT | 0.8526 +/- 0.1052 | 0.7558 | 0.8817 | 0.8483 | 17.51 (49) | 0.6408 |
| TC | 0.6876 +/- 0.2272 | 0.5637 | 0.7814 | 0.6950 | 16.14 (49) | 0.5011 |
| ET | 0.6736 +/- 0.2745 | 0.5585 | 0.6737 | 0.7899 | 14.06 (45) | 0.6510 |

WT/TC/ET mean Dice is 0.737904. Whole tumour is the strongest region; tumour core and enhancing tumour are more variable. Exclusive NCR/NET Dice is 0.3770. Four ET-absent reference cases have false-positive ET predictions (Dice zero, undefined HD95), and some ET-positive cases also score poorly. Keep these cases in reports. Surface Dice here uses unweighted 6-connected boundary voxel centres; it is not a surfel-area-weighted challenge implementation.

## Internal comparison available now

**This is not an independently trained raw-input baseline.** The B5 backbone was jointly trained with corrections, receives smoothed input, and has its branches disabled only at inference.

| Outcome | Corrections disabled | Full B5 | Difference, percentage points | Pointwise paired bootstrap 95% interval | Improved / worsened / tied patients |
|---|---:|---:|---:|---|---|
| WT | 0.8522 | 0.8526 | +0.036 | [+0.010, +0.065] | 31 / 18 / 0 |
| TC | 0.6827 | 0.6876 | +0.486 | [+0.247, +0.756] | 37 / 12 / 0 |
| ET | 0.6535 | 0.6736 | +2.010 | [+1.558, +2.466] | 41 / 4 / 4 |
| WT_TC_ET_mean | 0.7295 | 0.7379 | +0.844 | [+0.656, +1.039] | 45 / 4 / 0 |

The largest mean gain is ET (about 2.01 percentage points); mean WT/TC/ET gains about 0.844 points. T1ce corrections account for most of the branch-removal effect. T2 removal very slightly improves macro Dice. Intervals are exploratory, condition on these weights, do not account for validation selection or seed variation, and cannot replace a controlled B0/B1/B4 comparison. All contrasts and raw paired-difference plots are retained.

## Modality absence

| Available modalities | WT Dice | TC Dice | ET Dice | Mean Dice |
|---|---:|---:|---:|---:|
| T1 | 0.3304 | 0.2341 | 0.1347 | 0.2331 |
| T1ce | 0.3394 | 0.3844 | 0.4258 | 0.3832 |
| T1+T1ce | 0.4484 | 0.5259 | 0.5329 | 0.5024 |
| T2 | 0.6560 | 0.4121 | 0.2349 | 0.4343 |
| T1+T2 | 0.7403 | 0.4567 | 0.2656 | 0.4875 |
| T1ce+T2 | 0.6999 | 0.5966 | 0.6029 | 0.6332 |
| T1+T1ce+T2 | 0.7445 | 0.6597 | 0.6382 | 0.6808 |
| FLAIR | 0.7920 | 0.4380 | 0.2673 | 0.4991 |
| T1+FLAIR | 0.7966 | 0.4439 | 0.2666 | 0.5024 |
| T1ce+FLAIR | 0.8145 | 0.6370 | 0.6195 | 0.6903 |
| T1+T1ce+FLAIR | 0.8399 | 0.6917 | 0.6723 | 0.7346 |
| T2+FLAIR | 0.8302 | 0.4795 | 0.2873 | 0.5323 |
| T1+T2+FLAIR | 0.8385 | 0.4841 | 0.2915 | 0.5380 |
| T1ce+T2+FLAIR | 0.8399 | 0.6584 | 0.6550 | 0.7178 |
| T1+T1ce+T2+FLAIR | 0.8526 | 0.6876 | 0.6736 | 0.7379 |

The unweighted average across all 15 subsets is descriptive, not a model of clinical missingness. Removing T1ce while retaining T1/T2/FLAIR reduces ET Dice from 0.6736 to 0.2915 and mean Dice from 0.7379 to about 0.5380. The model tolerates absence unequally; broad missing-modality robustness is not demonstrated by one model without comparators.

## Spatial corruption: paired with matching clean availability

| Condition | WT change, pp | TC change, pp | ET change, pp | Mean change, pp |
|---|---:|---:|---:|---:|
| noise1_m15_flair | -0.0007 | +0.0013 | -0.0012 | -0.0002 |
| noise1_m13_flair | -0.0024 | -0.0007 | +0.0003 | -0.0010 |
| noise2_m15_flair | +0.0019 | +0.0005 | -0.0017 | +0.0002 |
| noise2_m13_flair | +0.0011 | +0.0007 | +0.0014 | +0.0011 |
| resolution1_m15_flair | -0.0172 | -0.0433 | -0.0055 | -0.0220 |
| resolution1_m13_flair | -0.0438 | -0.0023 | -0.0056 | -0.0172 |
| resolution2_m15_flair | -0.1292 | -0.0908 | -0.0615 | -0.0938 |
| resolution2_m13_flair | -0.0852 | +0.0396 | +0.0126 | -0.0110 |

These eight tests corrupt only FLAIR: local normalized-image Gaussian noise at 0.075/0.15 or axis-2 blur/downsample/upsample factors 2.5/3.5, with all modalities or T1ce absent. The location is seeded independently per patient/condition; it is not tumour-targeted. Zero background is preserved. Noise changes average Dice negligibly; strongest resolution degradation reduces four-modality mean Dice by about 0.094 points. This supports stability under this restricted simulation, not gate-specific benefit, real scanner shifts, arbitrary corruptions or clinical reliability. Gate-versus-utility diagnostics exist only for clean all-modality cases, so adaptive gate responses to corruption were not directly measured.

## Utility-gate verdict

Stage B selected 0.740310 with gates fixed to one. Stage C selected 0.737904 after gate learning: a reduction of 0.241 percentage points on clean validation. Stage C is nearly flat across its validation checkpoints. Whole-brain gates are close to 0.5; T1ce gates increase locally in tumours. Associations with measured utility are weak or mixed. These observations do not establish that utility supervision adds value. B4 segmentation-only and B2 unit-gate evaluations under the same robustness suite are needed to isolate that claim.
All 49 fixed-context stability enumeration checks passed over 16 correction subsets on sampled voxels, maximum numerical margin discrepancy 1.90735e-6. High stability can include incorrect predictions and is dominated by background; it is not a diagnostic guarantee.

## Independent baseline comparison: blocked by missing evidence

Update after the initial audit: the user supplied B0 best.pt, latest.pt and history.json under output_dump/B0_s1729/RAW. Their metadata verifies the same patient split as B5, a selected checkpoint at update 2000 with validation mean Dice 0.7109008, and saved progress at 2999/10000. A B0-only continuation notebook and verified input bundle are prepared; final baseline evaluation remains unavailable. See BASELINE_CONTINUE.md and verification/baseline_recovery/checkpoint_inspection.json.
A partial baseline validation score from an old log is insufficient: the recovered checkpoint is unfinished and no matched patient-level evaluation table is available. Do not compare an old intermediate maximum with selected B5 as a completed experiment. The Stage A low-frequency model is also not B0.
To recover the older run, open its entry in Kaggle Versions and download its saved outputs from that version if they are still available. Do not rely on the current --version launcher option for recovery: the locally installed Kaggle client does not forward version numbers in its output request, and version-specific recovery has not been validated. Check which variant/checkpoint actually exists before comparing. If its B0 training is partial, either resume it to the matched planned budget or explicitly label a compute-limited pilot; evaluate on the same split/conditions. A fair raw-input baseline plus the B4 gate control is still required for strong method claims. No full baseline training or live retrieval was performed by this local audit.

## Figure review

All measured plots retain patient-level points, missing counts and source-data sidecars. The availability heatmap now has n=49 in every cell and the new corruption plot has matched patient differences. Qualitative panels depict one prespecified validation patient, not the test set or a representative cohort. They include persistent distant false positives.
Before paper use: the Stage C utility BCE curve is omitted because plotting requests utility while histories log utility_bce; the actual loss is recorded. Debug figures use a different label palette without a class legend. Zero gate margins outside the image crop are reconstruction fill, not low trust. Contribution maps would be easier to read with MRI overlays. Corruption-axis labels are small at IEEE column width; use a two-column layout or shorter labels.
The plot called Base is B5 with corrections disabled, not separately trained B0. Label it that way. Raw datasets/qualitative assets are real, not fabricated. Existing manuscript placeholders/status must be updated from this completed validation record, while independent baseline/test entries must remain pending.

## Practical next step and overall verdict

No further default B5 training or validation continuation is necessary. Repeating the default command now mainly repeats setup/checks/profiling and reloads completed work; it does not enable the test set or create a baseline.
**Engineering verdict:** successful complete validation allocation, verified artifacts, genuine nonzero detail effects.
**Scientific verdict:** a working model with moderate, uneven validation segmentation and small internal detail gains. The utility-supervision improvement hypothesis is currently unsupported, and clean validation slightly favors unit gates. No claim of superiority, state of the art, external generalization or clinical robustness is warranted.
Fix the evaluation protocol before opening the 101-patient test split. Obtain a genuine baseline checkpoint/evaluation and preferably B4 controls. Report negative outcomes and one-seed limitations. If protocol changes or model tuning are needed, continue using validation only; keep test data untouched.

## File-level inventory

- `detailtrust\ablation_results.json`: read_and_hashed; 22491165 bytes.
- `detailtrust\artifact_manifest.json`: read_and_hashed; 8132 bytes.
- `detailtrust\artifact_verification.json`: read_and_hashed; 975 bytes.
- `detailtrust\best_detailtrust_model.pt`: read_and_hashed; 68094847 bytes.
- `detailtrust\best_detailtrust_model_config.json`: read_and_hashed; 401 bytes.
- `detailtrust\compute_profile.json`: read_and_hashed; 1040 bytes.
- `detailtrust\correction_contributions.csv`: read_and_hashed; 230938 bytes.
- `detailtrust\dataset_discovery.json`: read_and_hashed; 15622 bytes.
- `detailtrust\dataset_evidence.json`: read_and_hashed; 17593 bytes.
- `detailtrust\dataset_integrity.json`: read_and_hashed; 490297 bytes.
- `detailtrust\detailtrust_report.json`: read_and_hashed; 45464445 bytes.
- `detailtrust\environment.json`: read_and_hashed; 382 bytes.
- `detailtrust\experiment_config.json`: read_and_hashed; 1199 bytes.
- `detailtrust\experiments\B5_s1729\inference_benchmark.json`: read_and_hashed; 800 bytes.
- `detailtrust\experiments\B5_s1729\refinement\best.pt`: read_and_hashed; 68094847 bytes.
- `detailtrust\experiments\B5_s1729\refinement\history.json`: read_and_hashed; 585571 bytes.
- `detailtrust\experiments\B5_s1729\refinement\latest.pt`: read_and_hashed; 68095057 bytes.
- `detailtrust\experiments\B5_s1729\refinement\stage_result.json`: read_and_hashed; 625311 bytes.
- `detailtrust\experiments\B5_s1729\result.json`: read_and_hashed; 21367930 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m01.json`: read_and_hashed; 465207 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m02.json`: read_and_hashed; 467462 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m03.json`: read_and_hashed; 467681 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m04.json`: read_and_hashed; 467481 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m05.json`: read_and_hashed; 467721 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m06.json`: read_and_hashed; 467880 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m07.json`: read_and_hashed; 467290 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m08.json`: read_and_hashed; 466769 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m09.json`: read_and_hashed; 468144 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m10.json`: read_and_hashed; 467493 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m11.json`: read_and_hashed; 467492 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m12.json`: read_and_hashed; 467250 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m13.json`: read_and_hashed; 467533 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m14.json`: read_and_hashed; 467326 bytes.
- `detailtrust\experiments\B5_s1729\validation\clean_m15.json`: read_and_hashed; 5048290 bytes.
- `detailtrust\experiments\B5_s1729\validation\noise1_m13_flair.json`: read_and_hashed; 470418 bytes.
- `detailtrust\experiments\B5_s1729\validation\noise1_m15_flair.json`: read_and_hashed; 470020 bytes.
- `detailtrust\experiments\B5_s1729\validation\noise2_m13_flair.json`: read_and_hashed; 470279 bytes.
- `detailtrust\experiments\B5_s1729\validation\noise2_m15_flair.json`: read_and_hashed; 470011 bytes.
- `detailtrust\experiments\B5_s1729\validation\resolution1_m13_flair.json`: read_and_hashed; 472191 bytes.
- `detailtrust\experiments\B5_s1729\validation\resolution1_m15_flair.json`: read_and_hashed; 471988 bytes.
- `detailtrust\experiments\B5_s1729\validation\resolution2_m13_flair.json`: read_and_hashed; 472316 bytes.
- `detailtrust\experiments\B5_s1729\validation\resolution2_m15_flair.json`: read_and_hashed; 472091 bytes.
- `detailtrust\experiments\shared_candidates_s1729\A\best.pt`: read_and_hashed; 203454266 bytes.
- `detailtrust\experiments\shared_candidates_s1729\A\history.json`: read_and_hashed; 1222791 bytes.
- `detailtrust\experiments\shared_candidates_s1729\A\latest.pt`: read_and_hashed; 203801268 bytes.
- `detailtrust\experiments\shared_candidates_s1729\A\stage_result.json`: read_and_hashed; 1311576 bytes.
- `detailtrust\experiments\shared_candidates_s1729\B\best.pt`: read_and_hashed; 204198816 bytes.
- `detailtrust\experiments\shared_candidates_s1729\B\history.json`: read_and_hashed; 1718671 bytes.
- `detailtrust\experiments\shared_candidates_s1729\B\latest.pt`: read_and_hashed; 204207422 bytes.
- `detailtrust\experiments\shared_candidates_s1729\B\stage_result.json`: read_and_hashed; 1836459 bytes.
- `detailtrust\figures\B5_s1729_learning_rate.json`: read_and_hashed; 998520 bytes.
- `detailtrust\figures\B5_s1729_learning_rate.pdf`: read_and_hashed; 14734 bytes.
- `detailtrust\figures\B5_s1729_learning_rate.png`: read_and_hashed; 224814 bytes.
- `detailtrust\figures\B5_s1729_learning_rate.svg`: read_and_hashed; 46732 bytes.
- `detailtrust\figures\B5_s1729_training_objectives.json`: read_and_hashed; 1668629 bytes.
- `detailtrust\figures\B5_s1729_training_objectives.pdf`: read_and_hashed; 225454 bytes.
- `detailtrust\figures\B5_s1729_training_objectives.png`: read_and_hashed; 406989 bytes.
- `detailtrust\figures\B5_s1729_training_objectives.svg`: read_and_hashed; 570484 bytes.
- `detailtrust\figures\B5_s1729_validation_boundaries.json`: read_and_hashed; 183410 bytes.
- `detailtrust\figures\B5_s1729_validation_boundaries.pdf`: read_and_hashed; 19268 bytes.
- `detailtrust\figures\B5_s1729_validation_boundaries.png`: read_and_hashed; 214193 bytes.
- `detailtrust\figures\B5_s1729_validation_boundaries.svg`: read_and_hashed; 71491 bytes.
- `detailtrust\figures\B5_s1729_validation_contributions.json`: read_and_hashed; 125497 bytes.
- `detailtrust\figures\B5_s1729_validation_contributions.pdf`: read_and_hashed; 25423 bytes.
- `detailtrust\figures\B5_s1729_validation_contributions.png`: read_and_hashed; 217852 bytes.
- `detailtrust\figures\B5_s1729_validation_contributions.svg`: read_and_hashed; 122786 bytes.
- `detailtrust\figures\B5_s1729_validation_corruption.json`: read_and_hashed; 1531311 bytes.
- `detailtrust\figures\B5_s1729_validation_corruption.pdf`: read_and_hashed; 31347 bytes.
- `detailtrust\figures\B5_s1729_validation_corruption.png`: read_and_hashed; 272421 bytes.
- `detailtrust\figures\B5_s1729_validation_corruption.svg`: read_and_hashed; 164966 bytes.
- `detailtrust\figures\B5_s1729_validation_detail_removal.json`: read_and_hashed; 882444 bytes.
- `detailtrust\figures\B5_s1729_validation_detail_removal.pdf`: read_and_hashed; 23696 bytes.
- `detailtrust\figures\B5_s1729_validation_detail_removal.png`: read_and_hashed; 217545 bytes.
- `detailtrust\figures\B5_s1729_validation_detail_removal.svg`: read_and_hashed; 107265 bytes.
- `detailtrust\figures\B5_s1729_validation_detection.json`: read_and_hashed; 183407 bytes.
- `detailtrust\figures\B5_s1729_validation_detection.pdf`: read_and_hashed; 18959 bytes.
- `detailtrust\figures\B5_s1729_validation_detection.png`: read_and_hashed; 192010 bytes.
- `detailtrust\figures\B5_s1729_validation_detection.svg`: read_and_hashed; 71228 bytes.
- `detailtrust\figures\B5_s1729_validation_et_size.json`: read_and_hashed; 61229 bytes.
- `detailtrust\figures\B5_s1729_validation_et_size.pdf`: read_and_hashed; 13469 bytes.
- `detailtrust\figures\B5_s1729_validation_et_size.png`: read_and_hashed; 78397 bytes.
- `detailtrust\figures\B5_s1729_validation_et_size.svg`: read_and_hashed; 20101 bytes.
- `detailtrust\figures\B5_s1729_validation_gate_utility.json`: read_and_hashed; 329248 bytes.
- `detailtrust\figures\B5_s1729_validation_gate_utility.pdf`: read_and_hashed; 20815 bytes.
- `detailtrust\figures\B5_s1729_validation_gate_utility.png`: read_and_hashed; 161160 bytes.
- `detailtrust\figures\B5_s1729_validation_gate_utility.svg`: read_and_hashed; 80938 bytes.
- `detailtrust\figures\B5_s1729_validation_missing_modalities.json`: read_and_hashed; 2746960 bytes.
- `detailtrust\figures\B5_s1729_validation_missing_modalities.pdf`: read_and_hashed; 58839 bytes.
- `detailtrust\figures\B5_s1729_validation_missing_modalities.png`: read_and_hashed; 313623 bytes.
- `detailtrust\figures\B5_s1729_validation_missing_modalities.svg`: read_and_hashed; 65785 bytes.
- `detailtrust\figures\B5_s1729_validation_overlap.json`: read_and_hashed; 183401 bytes.
- `detailtrust\figures\B5_s1729_validation_overlap.pdf`: read_and_hashed; 18567 bytes.
- `detailtrust\figures\B5_s1729_validation_overlap.png`: read_and_hashed; 181464 bytes.
- `detailtrust\figures\B5_s1729_validation_overlap.svg`: read_and_hashed; 71866 bytes.
- `detailtrust\figures\B5_s1729_validation_stability.json`: read_and_hashed; 24296 bytes.
- `detailtrust\figures\B5_s1729_validation_stability.pdf`: read_and_hashed; 15566 bytes.
- `detailtrust\figures\B5_s1729_validation_stability.png`: read_and_hashed; 144144 bytes.
- `detailtrust\figures\B5_s1729_validation_stability.svg`: read_and_hashed; 30995 bytes.
- `detailtrust\figures\B5_s1729_validation_trajectory.json`: read_and_hashed; 538131 bytes.
- `detailtrust\figures\B5_s1729_validation_trajectory.pdf`: read_and_hashed; 15427 bytes.
- `detailtrust\figures\B5_s1729_validation_trajectory.png`: read_and_hashed; 123614 bytes.
- `detailtrust\figures\B5_s1729_validation_trajectory.svg`: read_and_hashed; 46114 bytes.
- `detailtrust\figures\BraTS20_Training_205_display_source.npz`: read_and_hashed; 548648 bytes.
- `detailtrust\figures\BraTS20_Training_205_et_contribution_maps.json`: read_and_hashed; 1175 bytes.
- `detailtrust\figures\BraTS20_Training_205_et_contribution_maps.pdf`: read_and_hashed; 56964 bytes.
- `detailtrust\figures\BraTS20_Training_205_et_contribution_maps.png`: read_and_hashed; 138164 bytes.
- `detailtrust\figures\BraTS20_Training_205_et_contribution_maps.svg`: read_and_hashed; 124900 bytes.
- `detailtrust\figures\BraTS20_Training_205_segmentation_views.json`: read_and_hashed; 1198 bytes.
- `detailtrust\figures\BraTS20_Training_205_segmentation_views.pdf`: read_and_hashed; 111372 bytes.
- `detailtrust\figures\BraTS20_Training_205_segmentation_views.png`: read_and_hashed; 313085 bytes.
- `detailtrust\figures\BraTS20_Training_205_segmentation_views.svg`: read_and_hashed; 244413 bytes.
- `detailtrust\figures\BraTS20_Training_205_spatial_gates.json`: read_and_hashed; 1090 bytes.
- `detailtrust\figures\BraTS20_Training_205_spatial_gates.pdf`: read_and_hashed; 59351 bytes.
- `detailtrust\figures\BraTS20_Training_205_spatial_gates.png`: read_and_hashed; 139516 bytes.
- `detailtrust\figures\BraTS20_Training_205_spatial_gates.svg`: read_and_hashed; 117558 bytes.
- `detailtrust\figures\example_corrections.png`: read_and_hashed; 281752 bytes.
- `detailtrust\figures\paper_figures_manifest.json`: read_and_hashed; 7312 bytes.
- `detailtrust\figures\preprocessing.png`: read_and_hashed; 228776 bytes.
- `detailtrust\figures\qualitative_manifest.json`: read_and_hashed; 1686 bytes.
- `detailtrust\figures\README.md`: read_and_hashed; 354 bytes.
- `detailtrust\gate_analysis.csv`: read_and_hashed; 816796 bytes.
- `detailtrust\metrics_per_subject.csv`: read_and_hashed; 4966706 bytes.
- `detailtrust\metrics_summary.csv`: read_and_hashed; 648844 bytes.
- `detailtrust\README.md`: read_and_hashed; 1378 bytes.
- `detailtrust\resume_restore.json`: read_and_hashed; 527 bytes.
- `detailtrust\split_manifest.json`: read_and_hashed; 10603 bytes.
- `detailtrust\stability_analysis.json`: read_and_hashed; 127027 bytes.
- `detailtrust\tiny_overfit_check.json`: read_and_hashed; 633 bytes.
- `detailtrust\training_history.json`: read_and_hashed; 4028529 bytes.
- `detailtrust\unit_checks.json`: read_and_hashed; 1447 bytes.
- `detailtrust-net-reproducible-brats-experiments.log`: read_and_hashed; 73183 bytes.
- `download_manifest.json`: read_and_hashed; 405 bytes.
