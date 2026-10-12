# Downloaded Kaggle outputs

No cloud outputs have been downloaded yet.

After configuring the notebook ID, run:

```powershell
python sync_kaggle.py --download
```

The script checks Kaggle status first. Running, queued, failed or unknown jobs are
not downloaded as successes. To wait for a running job:

```powershell
python sync_kaggle.py --download --wait 43200
```

To rebuild, upload, request GPU execution and wait/download in one command:

```powershell
python sync_kaggle.py --kernel YOUR_USERNAME/detailtrust-net --run --download --wait 43200
```

Each download uses a new UTC timestamped folder. `download_manifest.json` distinguishes
Kaggle execution completion from completion of the requested experiments, and checks
the selected B5 checkpoint hash when present. Incomplete downloads/experiments remain
marked unsuccessful. Use `--version NUMBER` to pin a later download; avoid pushing
another version while downloading the latest one. No existing outputs are overwritten.
