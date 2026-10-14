# Continue the recovered B0 baseline

The original baseline saved 2,999/10,000 updates. Its selected best checkpoint is
update 2,000, validation mean Dice 0.7109008. That selection observation is not a
final budget-matched comparison with B5. No result.json is required to restore:
the supplied checkpoints contain the exact config, split, optimizer, scheduler,
AMP scaler, RNG state and history.

1. In Kaggle, create a **private dataset** by uploading the files inside
   `baseline_resume_input/`. These are checkpoint/metadata files only, about
   408 MB; no MRI cache. Preserve all payload filenames and resume_manifest.json.
2. Create a **new notebook**, import `detailtrust_b0_resume.ipynb`, and name it
   `DetailTrust B0 Baseline`. Keep the completed B5 notebook separate.
3. Attach the private resume dataset and the same dataset used for B5:
   `awsaf49/brats20-dataset-training-validation`.
4. Enable GPU T4, then **Save Version / Run All**. Do not remove the bundle ID.
   Expect `Resuming RAW: 2999/10000 ... 7001 remaining.` Missing input or a
   fingerprint mismatch stops execution; it never silently starts fresh.
5. If another session is needed, attach the latest baseline output as a dataset.
   In a copy of this baseline notebook, set RESUME_BUNDLE_ID=None and set
   RESUME_DIRECTORY to that input's actual `detailtrust` directory. Keep B0 and
   all training settings unchanged. Run All to restore latest progress.

When the baseline finishes, download detailtrust_report.json, split_manifest.json,
experiment_config.json, metrics_per_subject.csv, metrics_summary.csv, and the
experiments/B0_s1729 result and RAW checkpoint files. Skip cache. We can then make
the actual patient-matched B0/B5 comparison; final test evaluation remains disabled
until the protocol is locked. B0 does not use B5's corruption augmentation, so
its comparison tests the combined architecture/training recipe. B1 would isolate
architecture more fairly with matched corruption training.

The existing kaggle.ps1 default still targets B5 and does not upload this separate
baseline notebook. Use the Kaggle import steps above for this prepared artifact.
Local checks verified archive CRCs, checkpoint metadata, the split, the original
training fingerprint, input hashes, notebook JSON and cell syntax. No GPU run or
strict local PyTorch model reload was performed. Original downloaded files and
the paper were not changed.
