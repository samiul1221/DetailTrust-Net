"""Rebuild and upload a private Kaggle notebook without exposing credentials.

First use: python sync_kaggle.py --kernel YOUR_USERNAME/detailtrust-net
Later:     python sync_kaggle.py
Run too:   python sync_kaggle.py --run

Requires a current official Kaggle CLI and python-dotenv. The notebook itself
does not read .env. Only the CLI child process receives KAGGLE_API_TOKEN.
"""
import argparse
import base64
import csv
import io
import hashlib
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
from urllib.parse import quote, quote_plus
from build_notebook import build_notebook, ROOT

OUTPUT_PATTERN = r'^(?!.*(?:^|[/\\])cache(?:[/\\]|$)).*$'
RESUME_PATTERN = r'^(?!.*(?:^|[/\\])cache(?:[/\\]|$)).*(?:\.pt|\.json)$'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filtered_output(target, folder, environment, pattern=OUTPUT_PATTERN):
    """Filter remote filenames before any file transfer (never download then delete)."""
    help_result = run_kaggle(['kernels', 'output', '--help'], environment)
    if help_result.returncode or '--file-pattern' not in help_result.stdout + help_result.stderr:
        raise SystemExit('This CLI cannot filter outputs safely; no unfiltered download was attempted.')
    return run_kaggle(['kernels', 'output', target, '-p', str(folder), '--quiet',
                       '--file-pattern', pattern], environment, timeout=3600)


def resume_source(target, environment, folder, mode):
    """Read the current cloud report; reuse a matching downloaded copy when available."""
    if kernel_status(target, environment) != 'complete':
        raise SystemExit('The previous notebook must finish before resuming its saved outputs. No new run was started.')
    response = filtered_output(target, folder, environment, r'(?:^|/)detailtrust_report\.json$')
    if response.returncode:
        fail_cli('resume report download', response, environment)
    reports = list(Path(folder).rglob('detailtrust_report.json'))
    if len(reports) != 1:
        raise SystemExit('No unique resume report found. Refusing to restart; use --fresh only for an intentional new run.')
    report = json.loads(reports[0].read_text(encoding='utf-8'))
    saved_mode = report.get('training_configuration', {}).get('mode')
    if saved_mode != mode:
        raise SystemExit(f'Resume mode is {saved_mode}; requested {mode}. Keep the original mode or explicitly use --fresh.')
    digest = sha256_file(reports[0])
    for manifest_path in sorted((ROOT/'output_dump').glob('*/download_manifest.json'), reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('kernel', '').split('/')[:2] != target.split('/')[:2] or not manifest.get('download_command_succeeded'):
            continue
        candidates = list(manifest_path.parent.rglob('detailtrust_report.json'))
        if len(candidates) == 1 and sha256_file(candidates[0]) == digest:
            root = candidates[0].parent
            if list((root/'experiments').rglob('latest.pt')):
                print('Using downloaded checkpoints matching the current cloud report.', flush=True)
                return root
    print('Downloading saved checkpoints and evaluation records; cache excluded.', flush=True)
    response = filtered_output(target, folder, environment, RESUME_PATTERN)
    if response.returncode:
        fail_cli('resume checkpoint download', response, environment)
    return reports[0].parent


def package_resume(source, destination):
    """Flatten only checkpoint/evaluation artifacts into a hash-verified private input."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    required = [source/name for name in ('split_manifest.json', 'experiment_config.json', 'detailtrust_report.json')]
    if not all(p.is_file() for p in required) or not list((source/'experiments').rglob('latest.pt')):
        raise SystemExit('Resume checkpoints or split/configuration are missing. Refusing a fresh start.')
    files = required + sorted(p for p in (source/'experiments').rglob('*') if p.is_file() and p.suffix in ('.pt', '.json'))
    progress = {}
    for history in (source/'experiments').rglob('history.json'):
        rows = json.loads(history.read_text())
        if rows:
            latest, best = history.parent/'latest.pt', history.parent/'best.pt'
            if not latest.exists() or any('validation_dice' in r for r in rows) and not best.exists():
                raise SystemExit(f'Resume checkpoint missing beside {history.relative_to(source)}. No upload attempted.')
            progress[latest.relative_to(source).as_posix()] = dict(stage=rows[-1]['stage'], update=rows[-1]['update'])
    if not progress:
        raise SystemExit('No saved training progress found; refusing a fresh start.')
    report = json.loads((source/'detailtrust_report.json').read_text(encoding='utf-8'))
    for row in report.get('training_history', []):
        if row.get('source'):
            expected_path = 'experiments/' + str(Path(row['source']).parent/'latest.pt').replace('\\', '/')
            if expected_path not in progress or progress[expected_path]['update'] < row['update']:
                raise SystemExit('Downloaded files are missing progress recorded in the cloud report. Refusing a restart.')
    reported_stages = {row['stage'] for row in report.get('training_history', [])}
    if not reported_stages <= {item['stage'] for item in progress.values()}:
        raise SystemExit('A previously trained stage is missing from the resume bundle.')
    entries = []
    for index, path in enumerate(files):
        payload = f'payload_{index:04d}{path.suffix}'
        shutil.copy2(path, destination/payload)
        entries.append(dict(path=path.relative_to(source).as_posix(), payload=payload,
                            bytes=path.stat().st_size, sha256=sha256_file(destination/payload)))
    manifest = dict(schema_version=1, files=entries, progress=progress)
    manifest['bundle_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (destination/'resume_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def publish_resume_dataset(bundle, manifest, owner, environment):
    """Content-addressed private dataset; reuse on retry, never overwrite another bundle."""
    slug = 'detailtrust-resume-' + manifest['bundle_id'][:16]
    target = owner + '/' + slug
    existing = find_owned_dataset(target, environment)
    if not existing:
        metadata = dict(id=target, title=slug.replace('-', ' '), licenses=[{'name': 'other'}],
                        description='Private training checkpoints and experiment records. No MRI cache or credentials.')
        (Path(bundle)/'dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('Uploading private resume checkpoints (one-time transfer for this checkpoint).', flush=True)
        response = run_kaggle(['datasets', 'create', '-p', str(bundle), '--quiet', '--keep-tabular'], environment, timeout=3600)
        if response.returncode or 'Your private Dataset is being created' not in response.stdout + response.stderr:
            # A retry after a lost acknowledgement can race with a successful create.
            # Accept 409 only if the exact content-addressed dataset is now ours.
            if re.search(r'\b409\b', response.stdout + response.stderr) and find_owned_dataset(target, environment):
                pass
            else:
                fail_cli('private resume dataset creation', response, environment)
    deadline = time.monotonic() + 300
    while True:
        response = run_kaggle(['datasets', 'status', target], environment)
        if response.returncode:
            # Some Kaggle accounts receive 403 from the status route even for
            # an owned private dataset. Listing its uploaded manifest verifies
            # readiness without treating a forbidden lookup as “does not exist”.
            if not re.search(r'\b403\b', response.stdout + response.stderr):
                fail_cli('resume dataset readiness', response, environment)
            files = run_kaggle(['datasets', 'files', target, '--csv', '--page-size', '200'], environment)
            if files.returncode:
                fail_cli('resume dataset file verification', files, environment)
            # Blank CLI lines parse as [], not ['']; guard before indexing.
            # Parse the whole CSV so quoted filenames and CRLF work too.
            is_ready = any(row and row[0].lstrip('\ufeff').strip() == 'resume_manifest.json'
                           for row in csv.reader(io.StringIO(files.stdout)))
        else:
            is_ready = bool(re.search(r'\bready\b', response.stdout, re.I))
        if is_ready:
            return target
        if response.returncode == 0 and re.search(r'\b(error|failed)\b', response.stdout, re.I):
            fail_cli('resume dataset processing', response, environment)
        if time.monotonic() >= deadline:
            raise SystemExit('Resume dataset is still processing. Rerun the same command later; no notebook run was started.')
        print('Waiting for checkpoint input to become ready (not waiting for training).', flush=True)
        time.sleep(10)


def parse_dataset_listing(output):
    """Read the Kaggle CLI CSV table and expose any pagination token."""
    lines = output.splitlines()
    next_page = re.search(r'Next Page Token\s*=\s*(\S+)', output)
    for index, line in enumerate(lines):
        header = next(csv.reader([line.lstrip('\ufeff')]), [])
        if 'ref' in header and 'title' in header:
            table = []
            for row in csv.DictReader(io.StringIO('\n'.join([line.lstrip('\ufeff'), *lines[index+1:]]))):
                if not row.get('ref') or row['ref'].startswith('Next Page Token'):
                    continue
                table.append(row)
            return table, next_page.group(1) if next_page else None
    if 'No datasets found' in output:
        return [], None
    raise SystemExit('Could not read your Kaggle dataset inventory. No resume dataset upload was attempted.')


def find_owned_dataset(target, environment):
    """Resolve exact private resume ownership through the account listing, not a 403-prone lookup."""
    rows, page_token = [], None
    while True:
        args = ['datasets', 'list', '--mine', '--csv', '--page-size', '100']
        if page_token:
            args.extend(['--page-token', page_token])
        response = run_kaggle(args, environment)
        if response.returncode:
            fail_cli('resume dataset inventory', response, environment)
        page, page_token = parse_dataset_listing(response.stdout)
        rows.extend(row for row in page if row.get('ref', '').lower() == target.lower())
        if len(rows) > 1:
            raise SystemExit('Kaggle returned duplicate exact resume dataset references. No upload attempted.')
        if not page_token:
            return bool(rows)


def configure_resume_notebook(notebook, bundle_id, mode):
    import nbformat
    document = nbformat.read(notebook, as_version=4)
    for cell in document.cells:
        if cell.metadata.get('detailtrust_cell') == 'configuration':
            cell.source = re.sub(r'MODE = "[a-z]+"', f'MODE = "{mode}"', cell.source)
            cell.source = re.sub(r'RESUME_BUNDLE_ID = None', f'RESUME_BUNDLE_ID = {json.dumps(bundle_id)}', cell.source)
            if bundle_id and bundle_id not in cell.source:
                raise SystemExit('Notebook has no resume configuration cell; refusing upload.')
    nbformat.write(document, notebook)


def run_kaggle(arguments, environment, timeout=180):
    """Never relay raw CLI errors, HTTP details or credential-bearing output."""
    return subprocess.run([sys.executable, "-c", "from kaggle.cli import main; main()", *arguments], env=environment,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def parse_kernel_status(output):
    match = re.search(r'(?:has status|"status"\s*:|status\s*[:=])\s*["\']?(?:kernelworkerstatus\.)?(complete|running|queued|error|cancelled)', output.lower())
    return match.group(1) if match else "unknown"


def parse_pushed_version(output):
    """Require the CLI's positive acknowledgment; exit code zero alone is insufficient."""
    match = re.search(r"Kernel\s+version\s+(\d+)\s+successfully\s+(?:pushed|saved without running)\.", output, re.I)
    return int(match.group(1)) if match else None


def upload_acknowledged(output):
    """Some CLI releases acknowledge a save without including its version number."""
    return bool(re.search(r"Kernel\s+version\s+(?:\d+\s+)?successfully\s+(?:pushed|saved without running)\.", output, re.I))


def parse_uploaded_kernel(output):
    """Use Kaggle's acknowledged URL, not the requested slug, as the saved identity."""
    match = re.search(r"https://(?:www\.)?kaggle\.com/(?:code|kernels)/([\w-]+/[\w-]+)(?=[/?\s.#]|$)", output)
    return match.group(1) if match else None


def parse_kernel_listing(output):
    """Accept the official CSV format, including a possible pagination notice."""
    lines = output.splitlines()
    for index, line in enumerate(lines):
        fields = next(csv.reader([line.lstrip('\ufeff')]), [])
        if 'ref' in fields and 'title' in fields:
            return list(csv.DictReader(io.StringIO('\n'.join([line.lstrip('\ufeff'), *lines[index+1:]]))))
    if output.strip() == 'Not found':
        return []
    raise SystemExit('Could not read Kaggle notebook inventory. No upload was attempted.')


def resolve_kernel(config, environment, allow_legacy_recovery=True):
    """Find an existing target before push, recovering only the known old title mismatch."""
    response = run_kaggle(['kernels', 'list', '--mine', '--csv', '--page-size', '100'], environment)
    if response.returncode:
        fail_cli('notebook lookup', response, environment)
    rows = parse_kernel_listing(response.stdout)
    target = config['kernel']
    owner, slug = target.split('/')
    owned = [row for row in rows if re.fullmatch(r'[\w-]+/[\w-]+', row.get('ref') or '')
             and row['ref'].split('/')[0].lower() == owner.lower()]
    exact = [row for row in owned if row['ref'].lower() == target.lower()]
    if len(exact) == 1:
        return exact[0]['ref'], True
    # Do not infer absence from a truncated account inventory.
    if 'Next Page Token' in response.stdout or len(rows) >= 100:
        raise SystemExit('Notebook inventory is paginated and the target was not found. Set --kernel to its exact owner/slug; no upload was attempted.')
    legacy = [row for row in owned if row.get('title') == 'DetailTrust-Net reproducible BraTS experiments']
    if allow_legacy_recovery and slug == 'detailtrust-net' and len(legacy) == 1:
        return legacy[0]['ref'], True
    if legacy or config.get('last_pushed_version') is not None:
        raise SystemExit('The saved notebook identifier was not found unambiguously. Use --kernel OWNER/SLUG from the existing notebook URL. No upload was attempted.')
    return target, False


def upload_metadata(target, notebook_name, existing):
    """Omit title on updates so Kaggle retains the notebook's current title and slug."""
    metadata = dict(id=target, code_file=notebook_name, language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=True, enable_internet=True,
                    dataset_sources=['awsaf49/brats20-dataset-training-validation'],
                    competition_sources=[], kernel_sources=[])
    if not existing:
        metadata['title'] = target.split('/')[1].replace('-', ' ')
    return metadata


def safe_cli_diagnostic(stdout, stderr, environment):
    """Redact credentials before selecting or saving useful CLI error lines."""
    def decoded(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
    message = decoded(stdout) + "\n" + decoded(stderr)
    secrets = set()
    for key, value in environment.items():
        if value and re.search(r"TOKEN|PASSWORD|SECRET|AUTHORIZATION|(?:^|_)KEY$", key, re.I):
            secrets.update((value, quote(value, safe=""), quote_plus(value),
                            base64.b64encode(value.encode()).decode()))
    for secret in sorted(secrets, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"\x1b\[[0-9;]*m", "", message)
    message = re.sub(r"https?://[^\s\"'<>]+", "[URL omitted]", message, flags=re.I)
    message = re.sub(r"(?im)^.*(?:authorization|cookie|api[_ -]?token|api[_ -]?key|password|client[_ -]?secret)\s*[=:].*$",
                     "[Credential-bearing line omitted]", message)
    message = re.sub(r"(?i)\b(?:bearer|basic)\s+[^\s,;]+", "[Authorization omitted]", message)
    message = re.sub(r"\bKGAT_[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[REDACTED]", message)
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    useful = [line for line in lines if re.search(
        r"error|exception|denied|quota|limit|invalid|failed|running|pending|cancel|forbidden|unauthori|timeout|connection|\b[45]\d\d\b", line, re.I)]
    return "\n".join((useful or lines)[-12:])[-6000:] or "The CLI returned no diagnostic text."


def fail_cli(operation, response, environment):
    """Persist only sanitized diagnostics; never automatically retry a mutation."""
    detail = safe_cli_diagnostic(response.stdout, response.stderr, environment)
    try:
        version = importlib.metadata.version("kaggle")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    record = dict(operation=operation, exit_code=response.returncode, kaggle_cli_version=version,
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), diagnostic=detail,
                  credentials_redacted=True)
    path = ROOT/"kaggle_last_error.json"
    try:
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        saved = "\nDiagnostic saved to kaggle_last_error.json."
    except OSError:
        saved = ""
    raise SystemExit(f"Kaggle {operation} failed or could not be confirmed (exit {response.returncode}).\n"
                     f"Credential-redacted diagnostic:\n{detail}{saved}\n"
                     "For an upload, check the notebook's Versions before retrying; no automatic retry was made.")


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
        fail_cli("status", response, environment)
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
    response = filtered_output(target, folder, environment)
    if response.returncode:
        result = dict(success=False, reason="Download failed or incomplete; rerun into a new timestamped directory")
    else:
        result = classify_download(folder)
    result.update(kernel=target, downloaded_at_utc=stamp, kaggle_status=status,
                  download_command_succeeded=response.returncode == 0, cache_excluded=True)
    (folder/"download_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Downloaded output:", folder)
    print("Requested experiments complete:", result["success"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", help="Existing or new owner/notebook-slug; saved locally for later syncs")
    parser.add_argument("--run", action="store_true", help="Also execute and consume Kaggle compute")
    parser.add_argument('--fresh', action='store_true', help='Explicitly start new weights; default resumes an existing notebook')
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
        if config.get('kernel') != args.kernel:
            config.pop('last_pushed_version', None)
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
            resolved, existing = resolve_kernel(config, environment, allow_legacy_recovery=not bool(args.kernel))
            if resolved != config['kernel']:
                print('Found the existing notebook:', 'https://www.kaggle.com/code/' + resolved)
                config['kernel'] = resolved
                config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
            target = config["kernel"] + (f"/{args.version}" if args.version else "")
            if args.status:
                print("Kaggle execution status:", kernel_status(target, environment))
                if not args.download:
                    return
            if args.download and not args.run:
                download_completed(target, environment, args.wait)
                return
            help_result = run_kaggle(["kernels", "push", "--help"], environment)
            if help_result.returncode:
                fail_cli("upload command inspection", help_result, environment)
            arguments = push_arguments(temporary, args.run, help_result)
            notebook_target = temporary/notebook.name
            shutil.copy2(notebook, notebook_target)
            metadata = upload_metadata(config['kernel'], notebook.name, existing)
            if existing and not args.fresh:
                mode = args.mode or 'main'
                source = resume_source(config['kernel'], environment, temporary/'resume_download', mode)
                bundle = temporary/'resume_upload'
                manifest = package_resume(source, bundle)
                dataset = publish_resume_dataset(bundle, manifest, config['kernel'].split('/')[0], environment)
                metadata['dataset_sources'].append(dataset)
                configure_resume_notebook(notebook_target, manifest['bundle_id'], mode)
                print('Resume progress:', manifest['progress'], flush=True)
            else:
                print('Starting fresh weights: explicit --fresh or first notebook creation.', flush=True)
            (temporary/"kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
            response = run_kaggle(arguments, environment)
            output = response.stdout + "\n" + response.stderr
            pushed_version = parse_pushed_version(output)
            if response.returncode or not upload_acknowledged(output):
                fail_cli("upload", response, environment)
            actual_kernel = parse_uploaded_kernel(output)
            if actual_kernel:
                if actual_kernel.split('/')[0].lower() != config['kernel'].split('/')[0].lower():
                    raise SystemExit('Kaggle acknowledged an unexpected notebook owner. Check Versions before retrying; the saved target was not changed.')
                config['kernel'] = actual_kernel
            config["last_pushed_version"] = pushed_version
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            if "not valid dataset sources" in output:
                raise SystemExit("The notebook was uploaded but Kaggle rejected its dataset attachment. Check the notebook Input panel; this run cannot train until the NIfTI dataset is attached.")
            print("Uploaded a new version:", "https://www.kaggle.com/code/" + config["kernel"])
            print("Version:", pushed_version if pushed_version is not None else "accepted; number not returned by CLI")
            print("Execution requested." if args.run else "Saved without execution; use Kaggle Save Version / Run All when ready.")
            if args.download:
                if pushed_version is None:
                    print("Automatic download skipped: inspect Versions and use --download --version NUMBER.")
                else:
                    download_completed(f"{config['kernel']}/{pushed_version}", environment, args.wait)
        except subprocess.TimeoutExpired as error:
            response = subprocess.CompletedProcess([], -1, stdout=error.stdout or "",
                stderr=(error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or "") + "\nRequest timed out; server acceptance is unknown.")
            fail_cli("request", response, environment)
        except OSError as error:
            response = subprocess.CompletedProcess([], -1, stdout="", stderr=str(error))
            fail_cli("local request execution", response, environment)


if __name__ == "__main__":
    main()
