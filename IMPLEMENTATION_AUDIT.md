# DetailTrust-Net implementation audit

Audit date: 12 October 2026. Architecture authority: `arch.md`. The unrelated
research brief was not used. `detailtrust_kaggle.ipynb` is standalone; Python
modules are editable local source, not Kaggle dependencies. Exact cell names,
IDs and positions are in `notebook_cell_map.json`.

## Requirement mapping

| Architecture requirement | Notebook implementation / cell name | Verification |
|---|---|---|
| Sections 7.3–7.4: modalities, availability, physical Gaussian mixture | `model.availability_maps`, `model.GaussianSplit` | Masking, normalized kernels, scale conversion, initialization and reconstruction tests |
| Section 7.5: full five-level residual U-Net | `model.ResidualUNet` | Widths 32/64/128/256/320; full B5 has 16,934,644 parameters; shape and gradient tests |
| Section 7.6: shared anchored correction | `model.DetailCorrection` | 37 input channels; zero-detail correction exactly zero |
| Sections 7.7–7.8: gate and additive output | `model.SpatialUtilityGate`, `model.fuse_logits` | 44 input channels; sigmoid range; unavailable contribution zero; unchanged background score |
| Section 8: objectives and staged training | `losses.stage_objective`, `training.configure_stage`, `training.run_training_stage` | A/B/C updates executed; utility sign/detachment and frozen candidates checked |
| Section 9: contribution/stability accounting | `evaluation.contribution_analysis`, `metrics.stability_bound` | All 16 correction subsets checked against bound; sliding-window identity checked |
| Section 10: BraTS preprocessing and split | `data.load_subject`, `data.make_split`, `subject_split` | Label mapping, disjoint splits, deterministic indexed patches tested; NIfTI runtime test locally blocked |
| Sections 10.4–10.5: missing/corrupted MRI | `data.training_augmentation`, `evaluation.evaluation_conditions` | All 15 nonempty subsets; corruption implementations checked on synthetic volumes |
| Section 11: compute and checkpointing | `gpu_detection`, `gpu_profile_execution`, `training.save_checkpoint` | CPU profiling only; strict model reload and exact next-update optimizer/RNG resumption passed |
| Section 12.1: B0–B5, A1–A7 | `config.experiment_registry`, `experiments.run_experiment_sweep` | All 13 variants exercised with tiny synthetic training; five core variants also ran through generated notebook cells |
| Sections 12.4–12.6: evaluation/statistics | `metrics.subject_metrics`, `metrics.paired_patient_comparison`, `reporting.build_report` | Patient macro metrics, empty cases, physical distances, JSON parsing and CSV export tested |
| Artifacts and reproducibility | `report_export`, `best_checkpoint_export`, `verify_artifacts` | Synthetic checkpoint, configuration, report, histories, splits, four CSVs, stability JSON and figures saved |

## Actual execution evidence

- `verification/notebook_execution.json`: all 142 code cells executed with explicit
  local substitutions: CPU, synthetic data, 16³ patches, one update per stage,
  local output path, and no dependency download. Backbone widths stayed unchanged.
- `detailtrust_outputs/notebook_verification/unit_checks.json`: 11 test groups
  passed; the NIfTI adapter test is blocked because local NiBabel is absent.
- `verification/integration_status.json`: earlier module integration exercised all
  13 variants and produced 1,911 patient/condition/class records on synthetic data.
- `verification/sync_checks.json`: offline CLI response parsing and output
  classification tests passed. No authenticated API transaction is claimed.

The cell-order test found and fixed delayed Matplotlib backend selection. The
upload helper now requires a positive version acknowledgment and pins a subsequent
download to that version. A zero CLI exit status alone is not treated as success.

## Explicit implementation decisions

These are also serialized by `reporting.implementation_decisions()` in each report.

- B3 uses four global sigmoid scalars. A1 spatially permutes detached targets.
- A4 uses one-level Haar averaging/repetition with an even origin.
- A6 unfreezes all components in Stage C following the same A/B initialization.
- B2 gets additional B-style updates to match total update count. Gate controls
  fork the shared candidate before these extra updates.
- Utility pooling excludes padded positions. A7 uses central-patch coarse-grid
  occupancy supervision on globally degraded samples; it does not claim exact
  equivalence to the full MRI blur/resampling operator.
- The adapter losslessly reorients to RAS and rejects non-1mm/misaligned data.
  Evaluation reconstructs the full canonical grid. Native-orientation NIfTI
  prediction export is not implemented or claimed.
- Custom patient IDs are persisted; the original authors' split was not recovered.
- Surface Dice uses unweighted boundary voxel centres rather than area-weighted
  surfels. HD95 combines directed distances. Empty cases remain explicit.
- Epochs are logging intervals; stage transitions restore best validation weights.
- Small-ET threshold is the positive training-volume lower quartile; held-out labels
  do not determine it. Patient rows retain volumes and size groups.
- Flips are the implemented spatial augmentation. No rotation claim is made.

## Unverified work and limitations

No actual BraTS training, Kaggle upload, CUDA execution, T4 memory measurement,
clinical validation, or research ablation result has been obtained here. Thus
best medical validation/test scores are unavailable. Synthetic values are not
segmentation-performance evidence, and synthetic checkpoints are not useful
trained tumour models.

Live dataset licence/version, mounted case count, and real NIfTI geometry still
require Kaggle verification. The notebook checks them at runtime where possible;
the user must verify the dataset card's terms/version. No external authentication
or dataset download is performed inside the notebook.

Full sweeps may exceed one session or quota. Failed/paused/not-selected variants
remain explicit. Actual GPU fit and convergence are not guaranteed by CPU tests.
Published M²FTrans, evidential fusion and SwitchNet reproductions are not included;
internal controls alone cannot establish state of the art or scientific novelty.
Learning-curve runs use separate configurations/directories with 50/100/219
training subjects; they have not been executed on medical data.

## Cloud launch and output collection

Target saved locally: `samseikhnsut/detailtrust-net`. This session blocks outbound
local connections and offers no connected Kaggle upload tool. Cloud training has
**not started**. In a normal PowerShell terminal with network access, run:

```powershell
cd C:\D\project\bt
.\kaggle.ps1 --mode pilot --run --download --wait 43200
```

The launcher rebuilds the notebook, reads the existing local `.env` token without
printing it, attaches the NIfTI dataset, requests a private T4 GPU run and waits.
The downloader saves to `output_dump/` only when Kaggle reports completion, then
checks requested experiment statuses and checkpoint hash. A completed notebook
with partial experiments is not marked as a successful experiment sweep.
