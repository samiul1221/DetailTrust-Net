# DetailTrust-Net: one standalone Kaggle notebook

Upload **`detailtrust_kaggle.ipynb`** to Kaggle. All implementation code is already
inside it. You do **not** need the local `detailtrust/` Python package on Kaggle.
The Python modules are the tested editable source used to build the notebook.

The default experiment selection is now **B5 only**, per the user's revised
priority: train and evaluate the complete DetailTrust-Net first. All A/B/C stages
remain enabled. Baselines/ablations are implemented but recorded as `not_run`
unless added to `EXPERIMENTS`. B5-only results do not establish comparative gains.

## Run on Kaggle

1. Attach `awsaf49/brats20-dataset-training-validation`, the BraTS 2020 **NIfTI**
   dataset: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation.
   Do not substitute the similarly named HDF5 slice dataset. Confirm the dataset
   card's version and licence and follow the original BraTS usage/citation terms:
   https://www.med.upenn.edu/cbica/brats2020/data.html.
2. Enable a GPU in notebook settings. Internet is needed only if a dependency such
   as NiBabel is absent. The notebook does not use an API token.
3. Start with `MODE = "smoke"`. Check the inventory, tests, repeated-patch loss and
   GPU profile. Select `pilot`, `main`, or `extended` for longer experiments. All
   modes use the full 16,934,644-parameter B5 model. Real-data patches start at 96³.
4. Select **Save Version / Run All**. Artifacts go under
   `/kaggle/working/detailtrust/`. Download or retain the committed output.

Automatic discovery checks mounted paths first. Set `DATASET_ROOT` only if multiple
compatible mounts exist. Set `DATASET_VERSION` to the displayed Kaggle version.
The code validates modalities, labels, geometry, nonzero foreground and duplicate
image contents. It rejects incorrect files rather than guessing a repair. In
particular, an anomalous segmentation filename appears in some historical copies;
the discovery report exposes unmatched files, and affected subjects are excluded
with a reason rather than silently renamed. Subject counts are **measured**.

## Modes and compute

| Mode | Train subjects (maximum) | A / B / C updates | Patch | Selection cohort |
|---|---:|---:|---|---|
| smoke | 4 | 2 / 2 / 2 | 96³ | 1 validation subject |
| pilot | 100 | 400 / 450 / 150 | 96³ | 5 validation subjects |
| main | 219 | 4000 / 4500 / 1500 | 96³ | all validation subjects |
| extended | 219 | 4000 / 4500 / 1500 | 96³ | all validation subjects |

Each update uses one sampled patch. An epoch is a configurable reporting interval,
not one pass through the subjects. Extended mode defines three seeds; enable its
secondary variants in `EXPERIMENTS`. B0/B1 get A+B+C updates. B2 gets A+B followed
by C additional B-style updates. B4/B5 fork the same **pre-polish** Stage B weights.
Equal update counts do not imply equal GPU cost or convergence.

CUDA uses mixed precision, gradient scaling and activation checkpointing. CPU-only
real-data training is rejected. For local pipeline verification only, the explicit
`SYNTHETIC_TEST_ONLY` mode uses tiny fabricated volumes and 16³ patches; its metrics
have no medical interpretation. The backbone widths do not change.

The profile measures each stage before the sweep; OOM never silently shrinks the
model. If needed, explicitly set `cfg.patch=(80,80,80)` using `replace`, use a new
output directory and apply the same change to every comparator. Keep this deviation
in the report. The default sweep uses 90% of its configurable time allowance for
training (8.1 hours in main) and reserves the remainder for evaluation. Extended
mode uses 70%. Preflight/profile time is additional. Complete-input evaluation
runs first; unfinished robustness conditions remain explicitly pending.
The full research study can require multiple Kaggle sessions and weekly quotas.

## Resume

Attach the previous notebook output as an input, and set `RESUME_DIRECTORY` to its
`detailtrust` folder. Keep mode, seed, patient lists, training settings and patch size
unchanged. Latest checkpoints contain model, optimizer, scheduler, AMP scaler,
random states and completed update count. A deterministic indexed patch stream
continues from the same sample. The best-validation checkpoint remains separate.
Use the same output directory name; mismatched fingerprints fail explicitly.

Completed evaluation conditions are reused only if their checkpoint hash, subjects,
condition and evaluation settings match. Paused/failed/not-selected variants remain
visible in `ablation_results.json`; they are not reported as successful experiments.

## Test-set discipline

The default is a **custom** patient split of the labelled BraTS training collection.
At 369 eligible cases the counts are 219/50/100. Different eligible counts produce
declared proportional counts. These are not the original hidden challenge test
patients or the published M²FTrans subject lists. All splits and actual subsets are
saved. Test evaluation is disabled by default. After fixing the complete protocol,
set `RUN_FINAL_TEST=True` and `PROTOCOL_LOCKED=True` in `main`/`extended` mode.
Do not change the design after inspecting test results.

## Interpret artifacts

- `detailtrust_report.json`: configuration, actual results, status and limitations.
- `best_detailtrust_model.pt`: B5 selected using validation only; produced only if
  a B5 stage finishes. Its configuration and hash are recorded separately.
- `experiments/<variant>_<seed>/`: stage checkpoints, histories and condition results.
- `metrics_per_subject.csv`, `metrics_summary.csv`: patient-level scores and macro summaries.
- `gate_analysis.csv`, `correction_contributions.csv`, `stability_analysis.json`: explanations.
- `figures/`: plots of real logged values, including failures when observed.
- `artifact_verification.json`: exists/missing status of every expected artifact.

Undefined metrics use JSON `null`, with conventions and counts. Surface Dice uses
boundary voxel centres, not area-weighted surfels. Gates are not calibrated quality
probabilities. Stability refers only to attenuating computed detail contributions;
it is not correctness or whole-modality robustness. Full-volume contributions are
blended directly; the product of averaged gates and averaged corrections is not used.

## Load a checkpoint

Run the notebook definition cells first, then:

```python
state = torch.load('/kaggle/working/detailtrust/best_detailtrust_model.pt',
                   map_location=device, weights_only=False)
model = DetailTrustNet(**state['architecture']).to(device)
model.load_state_dict(state['model_state'], strict=True)
model.eval()
```

Only load checkpoints you created or trust: full resume checkpoints contain pickle
state. Inference receives images/availability only, never a segmentation mask.

## Edit locally and sync changes

On this Windows machine, `uv` is already installed. From a **normal PowerShell**
terminal, use the launcher to obtain Python 3.11 and the current sync dependencies:

```powershell
cd C:\D\project\bt
.\kaggle.ps1 --mode main --run
```

This command resumes the existing notebook from saved checkpoints, uploads a new version, requests a GPU run,
and exits after acknowledgment. It does not wait for training to finish.
It does not modify Codex permissions or `.env`.
For a later output download alone: `.\kaggle.ps1 --download`.
The cloud launch depends on valid credentials, dataset access and available GPU quota.

Edit `detailtrust/*.py` and the teaching/runtime cells in `build_notebook.py`. Then:

```powershell
.\kaggle.ps1 --mode main --run
```

Use `--fresh` only for an intentional restart. By default the script verifies the current cloud report,
reuses matching downloaded checkpoints, and attaches a private resume dataset. Missing or incompatible
checkpoints stop the upload/run rather than silently starting new weights. The cache is excluded before
download; runtime cache lives in `/tmp/detailtrust_cache`. See `KAGGLE_COMMANDS.md`.

The notebook ID is saved in `kaggle-sync.json`. Each upload checks the account's
notebook inventory, preserves the existing title, and records Kaggle's returned
URL. Local edits only reach Kaggle after a successful upload. Use `--status` to
check a run without uploading, or `--download` to retrieve completed outputs.
For rebuilding alone: `python sync_kaggle.py --build-only`.

Sync needs Python 3.11+ and a current official CLI:
`python -m pip install --upgrade kaggle python-dotenv nbformat`.
It reads `KAGGLE_API_TOKEN` from your local `.env` or environment without printing
it. The staging folder includes only the generated notebook and metadata. `.env`
is never uploaded. API error diagnostics are credential-redacted before display
and saved to `kaggle_last_error.json`. GPU uploads work with Kaggle CLI 2.2.4.
Save-only uploads require the newer `--no-run` option; if absent, the script stops
that operation rather than starting unwanted training. Do not edit the generated notebook locally and then rebuild:
source changes belong in the builder/modules, or a rebuild will replace them.

## Current evidence limits

No Kaggle upload, GPU measurement or real BraTS training is claimed in this source
delivery. Consult `IMPLEMENTATION_AUDIT.md` and `verification/` for actually executed
local tests. Synthetic runs test engineering only. Close published comparators are
not reproduced here; internal controls cannot justify a state-of-the-art claim.

## Paper figures

Measured figure candidates are saved in PNG (300 DPI), PDF, and SVG with captions and source-data JSON.
See `figures/paper_figures_manifest.json` for produced and unavailable plots. Validation and test are separate,
patient distributions retain missing/empty-case counts, and partial evaluations show their sample sizes.
No figures imply completed ablations or clinical validation. Target-journal formatting still needs review.

Figure preparation used the Scientific Agent Skills visualization workflow:
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
This is tooling attribution, not evidence for DetailTrust-Net performance.
