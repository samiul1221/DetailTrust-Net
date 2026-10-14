# Evaluate B0 locally without more training

Run this in your normal PowerShell terminal from `C:\D\project\bt`:

```powershell
.\baseline_cpu.ps1
```

It installs CPU dependencies in a uv-managed environment, reads the existing
`.env` token securely, downloads the 245 MRI/mask files for the **49 validation
patients only**, loads `output_dump/B0_s1729/RAW/best.pt`, and runs CPU inference.
It does not launch a Kaggle notebook, spend GPU hours, train a model, or download
the old Kaggle preprocessing cache. Raw validation images are required to
calculate segmentation results; checkpoints alone cannot produce Dice scores.

Keep an internet connection for setup and data download. You need uv installed.
The original BraTS terms and the Kaggle dataset's access restrictions still apply.
No API token is embedded in a notebook, report, or command. If authorization or
access fails, the downloader stops and prints credential-redacted diagnostics.

Downloads are under `validation_data/`. Results are under `local_baseline_results/`:

- `local_baseline_report.json`: checkpoint, training configuration, data checks,
  actual validation metrics, runtime, precision, and limitations.
- `metrics_per_subject.csv` and `metrics_summary.csv`: B0 region/class metrics.
- `baseline_vs_b5_per_patient.csv` and `baseline_vs_b5_summary.csv`: matched patient
  differences against the saved B5 validation results.
- `baseline_vs_b5.png`, `.pdf`, `.svg`: measured clean-validation comparison.
- `patients/`: saved results for each patient/condition; these permit continuation.

The figure legend says **B0 baseline**. The methods/report retain its actual
selected update (2,000) and planned budget (10,000); B5 completed its own budget.
This comparison can describe the selected checkpoints' validation performance,
but unequal training budgets do not isolate an architectural advantage. Test data
remain sealed. No runtime comparison between CPU B0 and GPU B5 is valid.

For a first-patient timing check:

```powershell
.\baseline_cpu.ps1 --subjects 1
```

This still prepares the full validation input set, but evaluates one patient only.
That output is a timing/subset check, not a complete validation result. Run the
default command afterward to evaluate the remaining patients; completed results
are reused. Rerun the same command after interruption to resume patient evaluation.

If you already have the NIfTI data elsewhere:

```powershell
.\baseline_cpu.ps1 --data-root 'C:\path\to\BraTS20' --no-download
```

Optional CPU threads: add `--threads 8`. Optional full validation robustness suite:
add `--robustness` (15 modality subsets plus eight FLAIR corruptions). This is much
more work than the default clean comparison and may be slow on CPU. No runtime
estimate is reported until a real patient has been evaluated.

Validation MRI hashes are compared with the B5 dataset integrity record before
inference. Spatial geometry, label mapping and reference ET volumes are checked.
Original label-array hashes were not saved by B5, so exact label-byte identity
cannot be independently established. CPU uses float32 whereas the saved B5 run
used GPU FP16; this execution difference is recorded in the report.

Local verification covered syntax, CLI help, file selection, pagination parsing,
and paired comparison arithmetic using labelled synthetic fixtures. This agent
environment blocked dependency downloads with socket error 10013, so real-data
CPU execution and dataset authentication have not been claimed as verified.
