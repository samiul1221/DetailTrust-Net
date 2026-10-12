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
import time
from datetime import datetime, timezone
from build_notebook import build_notebook, ROOT


def run_kaggle(arguments, environment, timeout=180):
    """Never relay raw CLI errors, HTTP details or credential-bearing output."""
    return subprocess.run([sys.executable, "-c", "from kaggle.cli import main; main()", *arguments], env=environment,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def parse_kernel_status(output):
    match = re.search(r'(?:has status|"status"\s*:|status\s*[:=])\s*["\']?(?:kernelworkerstatus\.)?(complete|running|queued|error|cancelled)', output.lower())
    return match.group(1) if match else "unknown"


def parse_pushed_version(output):
    """Require the CLI's positive acknowledgment; exit code zero alone is insufficient."""
    match = re.search(r"Kernel version (\d+) successfully (?:pushed|saved without running)\.", output)
    return int(match.group(1)) if match else None


def push_arguments(folder, run, help_result):
    """Check only the options needed for the requested operation."""
    if help_result.returncode:
        raise SystemExit("Could not inspect Kaggle's upload command. This is not a Python-version check. Raw CLI output suppressed to protect credentials.")
    help_text = help_result.stdout + help_result.stderr
    arguments = ["kernels", "push", "-p", str(folder)]
    if run:
        if "--accelerator" not in help_text:
            raise SystemExit("This Kaggle CLI does not expose --accelerator for GPU uploads. Check its kernels push --help output.")
        return arguments + ["--accelerator", "NvidiaTeslaT4"]
    if "--no-run" not in help_text:
        raise SystemExit("This Kaggle CLI can upload and run, but cannot save without running. Use --run to start GPU training, or import the notebook in Kaggle's web editor to save without execution.")
    return arguments + ["--no-run"]


def kernel_status(target, environment):
    response = run_kaggle(["kernels", "status", target], environment)
    if response.returncode:
        raise SystemExit("Status request failed. Check network, notebook ownership and token validity. Raw output suppressed.")
    return parse_kernel_status(response.stdout)


def classify_download(directory):
    """Kernel success alone does not establish successful research experiments."""
    reports = list(Path(directory).rglob("detailtrust_report.json"))
    if len(reports) != 1:
        return dict(success=False, reason="Expected one complete experiment report")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    cfg = report.get("training_configuration", {})
    statuses = report.get("experiment_status", {})
    if not isinstance(statuses, dict):
        return dict(success=False, reason="Report is an initialization/failure record")
    selected = [f"{v}_s{s}" for s in cfg.get("seeds", []) for v in cfg.get("variants", [])]
    complete = bool(selected) and all(statuses.get(k, {}).get("status") == "completed" for k in selected)
    checkpoint = report.get("best_checkpoint", {}) or {}
    hash_check = None
    if checkpoint.get("status") == "saved":
        paths = list(Path(directory).rglob("best_detailtrust_model.pt"))
        if len(paths) == 1:
            import hashlib
            digest = hashlib.sha256()
            with paths[0].open("rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    digest.update(block)
            hash_check = digest.hexdigest() == checkpoint.get("sha256")
            complete = complete and hash_check
        else:
            complete = False
    return dict(success=complete, selected_experiments=selected, best_checkpoint_hash_verified=hash_check,
                evidence_level=report.get("project", {}).get("evidence_level"),
                synthetic=report.get("project", {}).get("synthetic"),
                reason=None if complete else "Some requested experiments/artifacts are incomplete; inspect report")


def download_completed(target, environment, wait_seconds=0):
    deadline = time.monotonic() + wait_seconds
    while True:
        status = kernel_status(target, environment)
        print("Kaggle status:", status, flush=True)
        if status == "complete":
            break
        if status in ("error", "cancelled", "unknown") or time.monotonic() >= deadline:
            print("No successful run available for download. Nothing marked as successful.")
            return
        time.sleep(min(30, max(1, deadline-time.monotonic())))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump_root = ROOT/"output_dump"
    dump_root.mkdir(exist_ok=True)
    folder = dump_root/(target.replace("/", "--") + "--" + stamp)
    folder.mkdir(exist_ok=False)
    response = run_kaggle(["kernels", "output", target, "-p", str(folder), "--quiet"], environment, timeout=3600)
    if response.returncode:
        result = dict(success=False, reason="Download failed or incomplete; rerun into a new timestamped directory")
    else:
        result = classify_download(folder)
    result.update(kernel=target, downloaded_at_utc=stamp, kaggle_status=status,
                  download_command_succeeded=response.returncode == 0)
    (folder/"download_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Downloaded output:", folder)
    print("Requested experiments complete:", result["success"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", help="Existing or new owner/notebook-slug; saved locally for later syncs")
    parser.add_argument("--run", action="store_true", help="Also execute and consume Kaggle compute")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--status", action="store_true", help="Check execution status without uploading")
    parser.add_argument("--download", action="store_true", help="Download only after Kaggle reports complete; does not upload unless --run is also set")
    parser.add_argument("--wait", type=int, default=0, help="Seconds to wait for a running/queued job before downloading")
    parser.add_argument("--version", type=int, help="Pin status/download to a particular Kaggle version")
    parser.add_argument("--mode", choices=("smoke", "pilot", "main", "extended"), help="Override generated notebook mode for this upload")
    args = parser.parse_args()
    if args.wait < 0 or args.version is not None and args.version < 1:
        parser.error("--wait must be nonnegative and --version must be positive")
    if args.run and args.version is not None:
        parser.error("--version selects an existing run; omit it when launching a new version")
    config_path = ROOT/"kaggle-sync.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if args.kernel:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", args.kernel):
            parser.error("--kernel must be owner/notebook-slug, not a URL")
        config["kernel"] = args.kernel
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    notebook = build_notebook() if args.build_only or args.run or not (args.status or args.download) else ROOT/"detailtrust_kaggle.ipynb"
    if args.mode:
        import nbformat
        document = nbformat.read(notebook, as_version=4)
        for cell in document.cells:
            if cell.metadata.get("detailtrust_cell") == "configuration":
                cell.source = re.sub(r'MODE = "[a-z]+"', f'MODE = "{args.mode}"', cell.source)
        nbformat.write(document, notebook)
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
            target = config["kernel"] + (f"/{args.version}" if args.version else "")
            if args.status:
                print("Kaggle execution status:", kernel_status(target, environment))
                if not args.download:
                    return
            if args.download and not args.run:
                download_completed(target, environment, args.wait)
                return
            help_result = run_kaggle(["kernels", "push", "--help"], environment)
            arguments = push_arguments(temporary, args.run, help_result)
            notebook_target = temporary/notebook.name
            shutil.copy2(notebook, notebook_target)
            metadata = dict(id=config["kernel"], title="DetailTrust-Net reproducible BraTS experiments",
                            code_file=notebook.name, language="python", kernel_type="notebook",
                            is_private=True, enable_gpu=True, enable_internet=True,
                            dataset_sources=["awsaf49/brats20-dataset-training-validation"],
                            competition_sources=[], kernel_sources=[])
            (temporary/"kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
            response = run_kaggle(arguments, environment)
            pushed_version = parse_pushed_version(response.stdout)
            if response.returncode or pushed_version is None:
                raise SystemExit("Upload failed or outcome is uncertain. Check the Kaggle notebook before retrying. Raw API output suppressed to protect credentials.")
            if "not valid dataset sources" in response.stdout:
                raise SystemExit("The notebook was uploaded but Kaggle rejected its dataset attachment. Check the notebook Input panel; this run cannot train until the NIfTI dataset is attached.")
            config["last_pushed_version"] = pushed_version
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            print("Uploaded a new version:", "https://www.kaggle.com/code/" + config["kernel"])
            print("Version:", pushed_version)
            print("Execution requested." if args.run else "Saved without execution; use Kaggle Save Version / Run All when ready.")
            if args.download:
                download_completed(f"{config['kernel']}/{pushed_version}", environment, args.wait)
        except (subprocess.TimeoutExpired, OSError):
            raise SystemExit("Kaggle request timed out or could not connect. Check the target notebook before retrying; no automatic duplicate upload.") from None


if __name__ == "__main__":
    main()
