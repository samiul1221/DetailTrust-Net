"""Rebuild and upload a private Kaggle notebook without exposing credentials.

First use: python sync_kaggle.py --kernel YOUR_USERNAME/detailtrust-net
Later:     python sync_kaggle.py
Run too:   python sync_kaggle.py --run

Requires a current official Kaggle CLI and python-dotenv. The notebook itself
does not read .env. Only the CLI child process receives KAGGLE_API_TOKEN.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import shutil
from build_notebook import build_notebook, ROOT


def run_kaggle(arguments, environment):
    """Never relay raw CLI errors, HTTP details or credential-bearing output."""
    return subprocess.run([sys.executable, "-m", "kaggle", *arguments], env=environment,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", help="Existing or new owner/notebook-slug; saved locally for later syncs")
    parser.add_argument("--run", action="store_true", help="Also execute and consume Kaggle compute")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--status", action="store_true", help="Check execution status without uploading")
    args = parser.parse_args()
    config_path = ROOT/"kaggle-sync.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if args.kernel:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", args.kernel):
            parser.error("--kernel must be owner/notebook-slug, not a URL")
        config["kernel"] = args.kernel
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    notebook = build_notebook()
    if args.build_only:
        return
    if not config.get("kernel"):
        parser.error("First run needs --kernel YOUR_USERNAME/detailtrust-net to identify the upload target")
    if not importlib.util.find_spec("kaggle") or not importlib.util.find_spec("dotenv"):
        raise SystemExit("Install sync dependencies in Python 3.11+: python -m pip install --upgrade kaggle python-dotenv nbformat")
    from dotenv import dotenv_values
    token = os.environ.get("KAGGLE_API_TOKEN") or dotenv_values(ROOT/".env").get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("KAGGLE_API_TOKEN is missing from the local .env/environment. It is never needed inside the notebook.")
    environment = os.environ.copy()
    environment["KAGGLE_API_TOKEN"] = token
    # A fresh configuration directory prevents accidental fallback to another account.
    with tempfile.TemporaryDirectory(prefix="detailtrust-kaggle-") as temporary:
        temporary = Path(temporary)
        environment["KAGGLE_CONFIG_DIR"] = str(temporary/"config")
        (temporary/"config").mkdir()
        try:
            if args.status:
                response = run_kaggle(["kernels", "status", config["kernel"]], environment)
                if response.returncode:
                    raise SystemExit("Status request failed. Check network access, target ownership and token validity. Raw output suppressed.")
                output = response.stdout.lower()
                status = next((word for word in ("complete", "running", "queued", "error", "cancelled") if word in output), "unknown")
                print("Kaggle execution status:", status)
                return
            help_result = run_kaggle(["kernels", "push", "--help"], environment)
            if help_result.returncode or "--no-run" not in help_result.stdout:
                raise SystemExit("This Kaggle CLI lacks verified --no-run/token support. Upgrade with Python 3.11+: python -m pip install --upgrade kaggle")
            notebook_target = temporary/notebook.name
            shutil.copy2(notebook, notebook_target)
            metadata = dict(id=config["kernel"], title="DetailTrust-Net reproducible BraTS experiments",
                            code_file=notebook.name, language="python", kernel_type="notebook",
                            is_private=True, enable_gpu=True, enable_internet=True,
                            dataset_sources=["awsaf49/brats20-dataset-training-validation"],
                            competition_sources=[], kernel_sources=[])
            (temporary/"kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
            arguments = ["kernels", "push", "-p", str(temporary)]
            if not args.run:
                arguments.append("--no-run")
            response = run_kaggle(arguments, environment)
            if response.returncode:
                raise SystemExit("Upload failed or outcome is uncertain. Check the Kaggle notebook before retrying. Raw API output suppressed to protect credentials.")
            print("Uploaded a new version:", "https://www.kaggle.com/code/" + config["kernel"])
            print("Execution requested." if args.run else "Saved without execution; use Kaggle Save Version / Run All when ready.")
        except (subprocess.TimeoutExpired, OSError):
            raise SystemExit("Kaggle request timed out or could not connect. Check the target notebook before retrying; no automatic duplicate upload.") from None


if __name__ == "__main__":
    main()
