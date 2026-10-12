"""Offline tests for sync parsing and download classification; no API requests."""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
from unittest.mock import patch
import sync_kaggle as sync


def main():
    old_help = subprocess.CompletedProcess([], 0, stdout="--path --timeout --accelerator", stderr="")
    assert sync.push_arguments("stage", True, old_help) == ["kernels", "push", "-p", "stage", "--accelerator", "NvidiaTeslaT4"]
    try:
        sync.push_arguments("stage", False, old_help)
        raise AssertionError("Save-only must not silently start training")
    except SystemExit as error:
        assert "cannot save without running" in str(error)
    new_help = subprocess.CompletedProcess([], 0, stdout="--accelerator --no-run", stderr="")
    assert sync.push_arguments("stage", False, new_help)[-1] == "--no-run"
    assert "--no-run" not in sync.push_arguments("stage", True, new_help)
    try:
        sync.push_arguments("stage", True, subprocess.CompletedProcess([], 1, stdout="", stderr="private error"))
        raise AssertionError("Failed help command must be distinguished from missing options")
    except SystemExit as error:
        assert "Could not inspect" in str(error) and "private error" not in str(error)
    assert sync.parse_pushed_version("Kernel version 12 successfully pushed. Please check progress at URL") == 12
    assert sync.parse_pushed_version("Kernel version 13 successfully saved without running. See URL") == 13
    assert sync.parse_pushed_version("Kernel push error: example") is None
    for text, expected in [
        ('owner/notebook has status "complete"', "complete"),
        ('owner/notebook has status "KernelWorkerStatus.RUNNING"', "running"),
        ('{"status": "queued"}', "queued"),
        ('status: error', "error"), ('nonsense', "unknown")]:
        assert sync.parse_kernel_status(text) == expected
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report = dict(training_configuration=dict(seeds=[9], variants=["B5"]),
            experiment_status={"B5_s9": {"status": "completed"}},
            project=dict(evidence_level="smoke", synthetic=True), best_checkpoint={
                "status": "saved", "sha256": hashlib.sha256(b"fixture").hexdigest()})
        path = root/"detailtrust_report.json"
        path.write_text(json.dumps(report))
        assert not sync.classify_download(root)["success"]
        (root/"best_detailtrust_model.pt").write_bytes(b"fixture")
        assert sync.classify_download(root)["success"]
        report["experiment_status"]["B5_s9"]["status"] = "paused"
        path.write_text(json.dumps(report))
        assert not sync.classify_download(root)["success"]
        report["experiment_status"] = "initializing"
        path.write_text(json.dumps(report))
        assert not sync.classify_download(root)["success"]
        with patch.object(sync, "ROOT", root), patch.object(sync, "kernel_status", return_value="running"), patch.object(sync, "run_kaggle") as request:
            sync.download_completed("user/test", {}, wait_seconds=0)
            request.assert_not_called()
            assert not (root/"output_dump").exists()
    Path("verification/sync_checks.json").write_text(json.dumps({
        "status": "passed", "checks": ["GPU launch without no-run support", "save-only never silently executes", "failed help diagnostics", "status parsing", "checkpoint hash verification", "incomplete experiment rejection",
        "initialization report rejection", "no output download while running"],
        "live_api_executed": False}, indent=2))
    print("Offline sync checks passed; no API request made.")


if __name__ == "__main__":
    main()
