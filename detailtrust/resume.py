"""Restore a specific, verified checkpoint bundle before any training begins."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import torch
from .config import file_hash, write_json


def restore_resume_bundle(bundle_id, output_dir, cfg, input_root='/kaggle/input'):
    """Missing, corrupted, ambiguous or incompatible inputs stop execution."""
    matches = []
    for path in Path(input_root).rglob('resume_manifest.json'):
        manifest = json.loads(path.read_text())
        if manifest.get('bundle_id') == bundle_id:
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one attached resume bundle {bundle_id}; found {len(matches)}. Refusing fresh training.')
    manifest_path, manifest = matches[0]
    canonical = {k: v for k, v in manifest.items() if k != 'bundle_id'}
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest() == bundle_id, 'Resume manifest hash mismatch'
    destination = Path(output_dir).resolve()
    entries = {}
    for entry in manifest['files']:
        relative = PurePosixPath(entry['path'])
        assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in entry['path'] and ':' not in entry['path']
        payload = entry['payload']
        assert Path(payload).name == payload and '/' not in payload and '\\' not in payload
        source = manifest_path.parent/payload
        assert source.is_file() and source.stat().st_size == entry['bytes'], f'Missing/incomplete resume file: {relative}'
        assert file_hash(source) == entry['sha256'], f'Resume hash mismatch: {relative}'
        assert entry['path'] not in entries, 'Duplicate resume path'
        entries[entry['path']] = source
    old = json.loads(entries['experiment_config.json'].read_text())
    # Other checks, including stage fingerprints, run again before optimizer updates.
    keys = ('mode', 'seed', 'seeds', 'split_seed', 'patch', 'batch_size', 'train_subjects',
            'val_subjects', 'updates_a', 'updates_b', 'updates_c', 'validation_every', 'precision',
            'lr', 'gate_lr', 'warmup_updates', 'weight_decay', 'branch_dropout', 'utility_tau',
            'utility_weight', 'corruption_probability', 'synthetic', 'variants')
    for key in keys:
        assert json.dumps(old[key]) == json.dumps(getattr(cfg, key)), f'Resume setting changed: {key}; refusing fresh training'
    for path, expected in manifest['progress'].items():
        assert path in entries, f'Required checkpoint missing: {path}'
        state = torch.load(entries[path], map_location='cpu', weights_only=False)
        assert state['update'] == expected['update'] and state['stage'] == expected['stage'], 'Checkpoint progress differs from history'
        assert all(key in state for key in ('optimizer_state', 'scheduler_state', 'scaler_state', 'rng', 'fingerprint'))
        del state
    for relative, source in entries.items():
        target = (destination/relative).resolve()
        assert target.is_relative_to(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # Rerunning this cell must never overwrite newer in-session progress.
            raise RuntimeError(f'Resume destination already contains {relative}. Start a clean session rather than overwrite progress.')
        shutil.copy2(source, target)
    restored = dict(bundle_id=bundle_id, source=str(manifest_path.parent), progress=manifest['progress'],
                    verified_files=len(entries), status='restored')
    write_json(destination/'resume_restore.json', restored)
    print('Restored saved training progress:', manifest['progress'], flush=True)
    return restored
