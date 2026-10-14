# DetailTrust-Net implementation audit

## Follow-up: 13 October 2026

**Current selection: B5 only.** At the user's request, baseline/ablation execution
is deferred. The required A/B/C stages and 4,000/4,500/1,500 main updates are
unchanged. The default main training allowance is now 8.1 hours within a 9-hour
sweep; preflight time is additional. Extended mode retains the 70% allowance.
Complete-input evaluation runs first. Evaluation saves after each patient and
resumes pending subjects without duplicate rows; a condition may remain partial
when its session budget expires. Best B5 weights are exported before long evaluation.

`verification/b5_only_checks.json` records a successful temporary CPU synthetic
run of A/B/C, strict checkpoint reload, all 15 availability subsets plus a smoke
corruption, report export, and interrupted/resumed evaluation equivalence. No B0
training was launched by that test. Large temporary synthetic artifacts were
removed. The updated source has not yet been executed on Kaggle.

The metrics remain Dice, IoU, precision, recall, specificity, HD95 in millimetres,
surface Dice at 1 mm, volume error, and ET lesion matching, with patient aggregation
and explicit empty cases. Compute reports include measured latency, memory and
parameter counts. Lesion matching now constructs overlap pairs only where both
masks are positive, avoiding the former full-background sort without changing
the matching definition. Test patients remain sealed during model development;
final testing requires the notebook's explicit locked-protocol settings.

User-supplied Kaggle logs now confirm a real T4 run with BraTS data, all 12
then-listed checks passing, 96³ A/B/C profiling without OOM, and B0 reaching
2,800/10,000 updates. Best reported validation macro Dice was 0.7109. This is
an intermediate baseline result, not a completed B5 experiment or test result.
The cloud-status statements below describe the original delivery on 12 October.

Routine validation previously computed surface/lesion metrics although selection
uses only WT/TC/ET patient macro Dice. The local revision computes the same Dice
and loss directly, retaining the complete metrics in final evaluation. A synthetic
regression check verifies exact score/loss equivalence and rejection of training
subjects. Training IDs, validation subjects, checkpoint criterion, architecture
and update budget are unchanged. Small-ET threshold setup now reuses measurements
from the integrity audit instead of loading the training collection a second time.
Actual T4 speedup has not been measured. These edits do not alter an already
running Kaggle version. Existing checkpoints can be resumed with the same settings.

The observed validation-heavy B0 run is not on pace to finish 10,000 updates in
12 hours. That older version's shared training deadline is 6.3 hours after sweep start,
with a save/pause between updates; an in-progress validation may overrun it.

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


## 13 October 2026: automatic continuation, cache exclusion, and paper figures

- `kaggle.ps1 --mode main --run` now resumes an existing notebook by default.
  The uploader fetches its current report and reuses a matching downloaded copy
  or downloads checkpoint records. It creates/reuses a content-addressed private
  Kaggle dataset, waits for input readiness, attaches it, and injects the exact
  bundle hash. `--fresh` is the explicit reset override.
- Runtime verifies every payload hash and size, saved configuration and stage
  progress before copying. Stage fingerprints are checked again when restoring
  optimizer/scheduler/scaler/RNG state. Missing or incompatible input stops execution.
- Real downloaded checkpoint bundle was restored locally and verified at A=4000,
  B=3929 (11 files). This did not train or upload a cloud version.
- A synthetic CPU interrupted-stage test proved continuation executes zero A
  updates, only the remaining B update and the C update. Mock upload checks verify
  the private input attachment and injected bundle ID. See continuation_checks.json
  and downloaded_checkpoint_resume.json under verification/.
- `--download` passes a negative cache-directory regex to the installed CLI's
  `--file-pattern`; filtering precedes transfer. Unsupported filtering stops rather
  than downloading the cache. New runtime cache is `/tmp/detailtrust_cache`.
- Figure generation includes separate stage objectives, validation trajectories,
  learning rate, WT/TC/ET overlap/boundary/detection distributions, ET size analysis,
  missing-modality heatmap, paired corruption differences, gates versus measured
  utility, signed contributions, detail-removal differences, conditional stability,
  and measured multi-variant comparisons when available. Three additional
  qualitative figures use the first validation subject: orthogonal segmentations,
  gate maps, ET contribution maps. No new architecture or loss changes were made.
- PNG (300 DPI), PDF, SVG and caption/source-data JSON exports are implemented.
  Undefined cases and partial patient counts stay visible. No completed ablation
  is implied when only B5 is run. Actual downloaded histories were replotted;
  no held-out metrics were invented. Journal formatting remains to be checked.
- Live dataset creation, attachment, and cloud continuation cannot be verified
  from this network-restricted environment. The user's normal terminal performs
  those operations. Output status still distinguishes a clean notebook exit from
  a complete research experiment. Full robustness evaluation may span sessions.
