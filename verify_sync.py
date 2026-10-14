"""Offline tests for sync parsing and download classification; no API requests."""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import base64
import contextlib
import io
from urllib.parse import quote
from unittest.mock import patch
import sync_kaggle as sync


def main():
    header = 'ref,title,author,lastRunTime,totalVotes\n'
    actual = 'samseikhnsut/detailtrust-net-reproducible-brats-experiments'
    inventory = header + actual + ',DetailTrust-Net reproducible BraTS experiments,samseikhnsut,,0\n'
    saved = dict(kernel='samseikhnsut/detailtrust-net', last_pushed_version=1)
    listed = subprocess.CompletedProcess([], 0, stdout=inventory, stderr='')
    with patch.object(sync, 'run_kaggle', return_value=listed) as request:
        assert sync.resolve_kernel(saved, {}) == (actual, True)
        assert request.call_args.args[0][:2] == ['kernels', 'list']
    exact_inventory = inventory + 'samseikhnsut/detailtrust-net,Custom title,samseikhnsut,,0\n'
    with patch.object(sync, 'run_kaggle', return_value=subprocess.CompletedProcess([], 0, stdout=exact_inventory, stderr='')):
        assert sync.resolve_kernel(saved, {}) == (saved['kernel'], True)
    for unsafe_inventory in (inventory + inventory.splitlines()[1] + '\n', 'Not found',
                             'Next Page Token = more\n' + inventory):
        with patch.object(sync, 'run_kaggle', return_value=subprocess.CompletedProcess([], 0, stdout=unsafe_inventory, stderr='')):
            try:
                sync.resolve_kernel(saved, {})
                raise AssertionError('Ambiguous or missing previously uploaded notebook must not be recreated')
            except SystemExit:
                pass
    with patch.object(sync, 'run_kaggle', return_value=subprocess.CompletedProcess([], 0, stdout='Not found', stderr='')):
        assert sync.resolve_kernel({'kernel': 'user/new-notebook'}, {}) == ('user/new-notebook', False)
    existing_metadata = sync.upload_metadata(actual, 'notebook.ipynb', existing=True)
    assert 'title' not in existing_metadata and existing_metadata['id'] == actual
    assert existing_metadata['enable_gpu']
    assert sync.upload_metadata('user/new-notebook', 'notebook.ipynb', False)['title'] == 'new notebook'
    assert sync.parse_uploaded_kernel('Kernel version 2 successfully pushed. See https://www.kaggle.com/code/' + actual) == actual
    assert sync.parse_uploaded_kernel('https://www.kaggle.com.evil.invalid/code/user/notebook') is None
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
    assert sync.upload_acknowledged("Kernel version successfully pushed. Please check progress at URL")
    assert not sync.upload_acknowledged("Kernel push error: quota exhausted")
    assert sync.parse_pushed_version("Kernel version 14 successfully pushed.\n") == 14
    fake_secret = "test-token/with+punctuation=012345"
    environment = {"KAGGLE_API_TOKEN": fake_secret}
    raw = ("Kernel push error: GPU quota exceeded\n"
           f"Authorization: Bearer {fake_secret}\n"
           f"Exception: encoded={quote(fake_secret, safe='')}\n"
           f"Exception: encoded={base64.b64encode(fake_secret.encode()).decode()}\n"
           "ConnectionError: https://example.invalid/?secret=do-not-show-this\n"
           "Exception: Bearer unrelated-hidden-value\n")
    diagnostic = sync.safe_cli_diagnostic(raw, "", environment)
    for value in (fake_secret, quote(fake_secret, safe=""), base64.b64encode(fake_secret.encode()).decode(),
                  "do-not-show-this", "unrelated-hidden-value"):
        assert value not in diagnostic
    assert "GPU quota exceeded" in diagnostic
    for text, expected in [
        ('owner/notebook has status "complete"', "complete"),
        ('owner/notebook has status "KernelWorkerStatus.RUNNING"', "running"),
        ('{"status": "queued"}', "queued"),
        ('status: error', "error"), ('nonsense', "unknown")]:
        assert sync.parse_kernel_status(text) == expected
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root/'kaggle-sync.json').write_text(json.dumps(saved))
        notebook = root/'notebook.ipynb'
        notebook.write_text('{}')
        def fake_cli(arguments, environment, timeout=180):
            if arguments[1] == 'list':
                return listed
            if '--help' in arguments:
                return old_help
            assert arguments[:2] == ['kernels', 'push']
            metadata = json.loads((Path(arguments[3])/'kernel-metadata.json').read_text())
            assert metadata['id'] == actual and 'title' not in metadata
            # Exercise server-returned identity persistence, not just requested identity.
            return subprocess.CompletedProcess([], 0, stdout='Kernel version 2 successfully pushed. Please check progress at https://www.kaggle.com/code/samseikhnsut/server-returned-slug', stderr='')
        with patch.object(sync, 'ROOT', root), patch.object(sync, 'build_notebook', return_value=notebook), \
                patch.object(sync.sys, 'argv', ['sync_kaggle.py', '--run', '--fresh']), \
                patch.dict(sync.os.environ, {'KAGGLE_API_TOKEN': 'fake-offline-test-token'}), \
                patch.object(sync, 'run_kaggle', side_effect=fake_cli), contextlib.redirect_stdout(io.StringIO()):
            sync.main()
        persisted = json.loads((root/'kaggle-sync.json').read_text())
        assert persisted['kernel'] == 'samseikhnsut/server-returned-slug'
        assert persisted['last_pushed_version'] == 2
        assert sync.parse_dataset_listing('No datasets found') == ([], None)
        assert sync.parse_dataset_listing('ref,title,totalBytes\nuser/detailtrust-resume-abc,My resume,10\n')[0][0]['ref'] == 'user/detailtrust-resume-abc'
        dataset_responses = [subprocess.CompletedProcess([], 0, 'No datasets found', ''),
            subprocess.CompletedProcess([], 0, 'Your private Dataset is being created. Please check progress at URL', ''),
            subprocess.CompletedProcess([], 0, 'ready', '')]
        with patch.object(sync, 'run_kaggle', side_effect=dataset_responses) as cli:
            dataset = sync.publish_resume_dataset(root, {'bundle_id': 'a'*64}, 'user', {})
            assert dataset == 'user/detailtrust-resume-' + 'a'*16
            create_args = cli.call_args_list[1].args[0]
            assert create_args[:2] == ['datasets', 'create'] and '--public' not in create_args
            assert '--keep-tabular' in create_args
        with patch.object(sync, 'run_kaggle', side_effect=[
            subprocess.CompletedProcess([], 0, 'ref,title,totalBytes\nuser/detailtrust-resume-'+'a'*16+',Resume,20\n', ''),
            subprocess.CompletedProcess([], 0, 'DatasetStatus.READY', '')]):
            assert sync.publish_resume_dataset(root, {'bundle_id': 'a'*64}, 'user', {}) == 'user/detailtrust-resume-'+'a'*16
        with patch.object(sync, 'run_kaggle', side_effect=[
            subprocess.CompletedProcess([], 0, 'ref,title,totalBytes\nuser/detailtrust-resume-'+'a'*16+',Resume,20\n', ''),
            subprocess.CompletedProcess([], 1, '', '403 Client Error: Forbidden'),
            subprocess.CompletedProcess([], 0, '\n\r\n\ufeffname,size\r\n\r\n"resume_manifest.json",100\r\n\n', '')]) as cli:
            assert sync.publish_resume_dataset(root, {'bundle_id': 'a'*64}, 'user', {}) == 'user/detailtrust-resume-'+'a'*16
            assert not any(call.args[0][:2] == ['datasets', 'create'] for call in cli.call_args_list)
        # An empty listing or similar filename must not acknowledge readiness.
        for incomplete_listing in ('\n\r\n', 'name,size\nresume_manifest.json.bak,100\n'):
            with patch.object(sync, 'run_kaggle', side_effect=[
                subprocess.CompletedProcess([], 0, 'No datasets found', ''),
                subprocess.CompletedProcess([], 0, 'Your private Dataset is being created', ''),
                subprocess.CompletedProcess([], 1, '', '403 Client Error: Forbidden'),
                subprocess.CompletedProcess([], 0, incomplete_listing, ''),
                subprocess.CompletedProcess([], 0, 'DatasetStatus.READY', '')]) as cli, \
                    patch.object(sync.time, 'sleep') as sleep:
                assert sync.publish_resume_dataset(root, {'bundle_id': 'a'*64}, 'user', {}) == 'user/detailtrust-resume-'+'a'*16
                assert sum(call.args[0][:2] == ['datasets', 'create'] for call in cli.call_args_list) == 1
                sleep.assert_called_once_with(10)
        for path in ('detailtrust/cache/example.npy', 'cache/a.json', 'detailtrust/cache/sub/file', 'detailtrust\\cache\\x.npy'):
            assert not sync.re.search(sync.OUTPUT_PATTERN, path)
        for path in ('detailtrust/experiments/B/latest.pt', 'detailtrust/figures/x.pdf', 'detailtrust/detailtrust_report.json'):
            assert sync.re.search(sync.OUTPUT_PATTERN, path)
        with patch.object(sync, 'run_kaggle', side_effect=[
            subprocess.CompletedProcess([], 0, stdout='--file-pattern', stderr=''),
            subprocess.CompletedProcess([], 0, stdout='', stderr='')]) as cli:
            sync.filtered_output('user/notebook', root, {})
            assert cli.call_args.args[0][-2:] == ['--file-pattern', sync.OUTPUT_PATTERN]
        with patch.object(sync, "ROOT", root):
            try:
                sync.fail_cli("upload", subprocess.CompletedProcess([], 0, stdout=raw, stderr=""), environment)
                raise AssertionError("A server error with exit zero must not count as success")
            except SystemExit as error:
                assert "GPU quota exceeded" in str(error) and fake_secret not in str(error)
            error_record = json.loads((root/"kaggle_last_error.json").read_text())
            assert error_record["credentials_redacted"] and fake_secret not in json.dumps(error_record)
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
        "status": "passed", "checks": ["existing notebook identity recovery", "exact identity preferred", "ambiguous/missing/truncated inventory rejected", "existing title preserved", "actual acknowledged URL persisted", "credential-redacted diagnostics", "server error despite zero exit", "versionless upload acknowledgment", "GPU launch without no-run support", "save-only never silently executes", "failed help diagnostics", "status parsing", "checkpoint hash verification", "incomplete experiment rejection",
        "initialization report rejection", "no output download while running", "cache excluded before transfer",
        "resume dataset found by exact owned inventory", "private resume dataset creation/readiness",
        "403 status fallback verified through owned dataset manifest",
        "blank/BOM/CRLF/quoted CSV rows handled without indexing errors",
        "empty/similar-name file listing waits for readiness; exact dataset retry never recreates"],
        "live_api_executed": False}, indent=2))
    print("Offline sync checks passed; no API request made.")


if __name__ == "__main__":
    main()
