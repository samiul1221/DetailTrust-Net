# Kaggle commands

Run in PowerShell from the project folder:

```powershell
cd C:\D\project\bt
```

## Upload a new version and start GPU training

```powershell
.\kaggle.ps1 --mode main --run
```

This rebuilds your notebook from local changes, uploads a new private version to
`samseikhnsut/detailtrust-net`, requests GPU training, prints the link, and exits
once Kaggle acknowledges the upload. Training continues on Kaggle. You can close
your terminal. Run this same command whenever you want a new version to train.

`main` selects the full training schedule. For a shorter pilot, replace `main`
with `pilot`.

## Check the run's status

```powershell
.\kaggle.ps1 --status
```

## Download outputs after training finishes

```powershell
.\kaggle.ps1 --download
```

Outputs are saved in a dated folder inside `output_dump/`. If training has not
finished successfully, the command tells you and exits.

## Upload changes without starting training

```powershell
.\kaggle.ps1 --mode main
```

This save-only command requires a CLI with `--no-run`. Kaggle CLI 2.2.4 does
not have that option; use Kaggle's web editor for save-only updates on that version.
The GPU training command above works without `--no-run`.

## Open the notebook

https://www.kaggle.com/code/samseikhnsut/detailtrust-net

Your username is already saved. The script reads your existing `.env` token.

There is no need for `--wait`. That option only keeps your terminal waiting to
collect outputs automatically; it does not control cloud training. The commands
above upload, check, and download separately.
