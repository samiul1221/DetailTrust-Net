# Kaggle commands

Run in PowerShell from the project folder:

```powershell
cd C:\D\project\bt
```

## Continue the existing run on Kaggle GPU

```powershell
.\kaggle.ps1 --mode main --run
```

This command **continues the existing B5 run**, uploads the updated notebook,
starts GPU execution, and exits. No `--continue` or `--wait` is needed.

It checks the latest cloud report, reuses your matching downloaded checkpoints
(or downloads them without cache), uploads a private checkpoint input, and attaches
it automatically. It may take a few minutes to transfer/process checkpoints before
launching. It never waits for training to finish.

For the latest run downloaded on 14 October: **all B5 training is complete**
(A: 4,000/4,000, B: 4,500/4,500, C: 1,500/1,500). With those checkpoints restored
and the settings unchanged, this command skips further training updates and
continues unfinished validation modality/corruption evaluations. Setup, checks,
and profiling still run. Saved patient evaluations are reused.

Missing checkpoints, changed training settings, or a missing attached input stop
the run with an error; there is no fallback to fresh weights. Keep the same mode
and split. The first ever upload of a new notebook starts fresh automatically.

The notebook trains B5 only. Baselines/ablations are not run. The test set remains
sealed until the protocol is explicitly locked. Long robustness evaluations may
span sessions; repeat the same command after a completed session to continue.

## Intentionally start again from new weights

```powershell
.\kaggle.ps1 --mode main --run --fresh
```

Use this only when you want to discard the current training trajectory for the
new run. It does not delete older Kaggle versions or local downloads. A fresh pilot
uses `--mode pilot --run --fresh`.

## Check the run's status

```powershell
.\kaggle.ps1 --status
```

## Download outputs after training finishes

```powershell
.\kaggle.ps1 --download
```

Outputs are saved in a dated folder inside `output_dump/`. **Cache files are
excluded before download.** Older cache files already downloaded are untouched.
New notebooks store their preprocessing cache in `/tmp/detailtrust_cache`, outside
committed outputs.

A normally exited notebook can have paused training: `Requested experiments
complete: False` means work remains, not that the download failed. Use the continue
command above after checking the report.

Paper figure candidates are in `detailtrust/figures/`: PNG (300 DPI), PDF and SVG,
with caption/source-data JSON files. Accuracy, boundary, modality, corruption,
gate/contribution and stability figures require actual evaluation records.
Qualitative MRI/prediction/gate panels use a prespecified validation subject.
Missing results are listed in `paper_figures_manifest.json`, never invented.

## Upload changes without starting training

```powershell
.\kaggle.ps1 --mode main
```

This save-only command requires a CLI with `--no-run`. Kaggle CLI 2.2.4 does
not have that option; use Kaggle's web editor for save-only updates on that version.
The GPU training command above works without `--no-run`.

## Open the notebook

Use the link printed after upload. Its actual identifier is saved in
`kaggle-sync.json`; an earlier uploader could save the wrong identifier when
Kaggle derived a longer URL from the notebook title. The script now looks up
that existing notebook and corrects the saved identifier when the match is unique.

Local edits and cancelling a run do not update the cloud code. Run the upload
command above to create a new version; refresh Kaggle and open that version.

Your username is already saved. The script reads your existing `.env` token.

There is no need for `--wait`. That option only keeps your terminal waiting to
collect outputs automatically; it does not control cloud training. The commands
above upload, check, and download separately.

## If an upload fails

Check the notebook's **Versions** first: an uncertain response can mean Kaggle
accepted the upload but the client did not receive confirmation. If there is no
new version, retry the upload command. The script now prints a credential-redacted
diagnostic and saves it to `kaggle_last_error.json`. That file helps distinguish
quota, permission, cancellation, network and server errors; it does not contain
the raw API token. No automatic upload retry is performed.

If notebook lookup reports an ambiguous identifier, copy the `OWNER/SLUG` part
of your existing notebook's browser URL and use it once:

```powershell
.\kaggle.ps1 --kernel OWNER/SLUG --mode main --run
```

The script remembers it for later commands. A 409 alone does not establish the
cause. If it persists with the confirmed identifier, check that cancellation has
finished in Kaggle and inspect Versions before trying again.

The continuation uploader checks your owned dataset list for the exact private
resume bundle before creating it. If Kaggle denies the status endpoint, it checks
for the bundle manifest in the dataset's file list. It never interprets a 403 as
permission to overwrite or start training from scratch.

If an earlier attempt stopped with `IndexError` while reading the dataset file
list, rerun the same command after applying the parser fix. That failure occurred
before notebook upload; the script checks for and reuses the exact private resume
dataset if it was already created. Do not add `--fresh`.
