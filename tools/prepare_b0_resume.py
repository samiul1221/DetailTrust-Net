"""Recover genuine B0 metadata and prepare a hash-verified Kaggle continuation.

Source checkpoints remain unchanged. Only the experiment selection is narrowed
from the original five variants to B0; its training fingerprint stays identical.
No torch.load, training, cloud upload, or fabricated evaluation is performed.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import zipfile
import nbformat
from checkpoint_metadata import read_checkpoint_metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sync_kaggle import package_resume, sha256_file

SOURCE = ROOT/'output_dump/B0_s1729/RAW'
AUDIT = ROOT/'verification/baseline_recovery'
BUNDLE = ROOT/'baseline_resume_input'
NOTEBOOK = ROOT/'detailtrust_b0_resume.ipynb'


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def checkpoint_audit(name):
    path=SOURCE/name
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, 'Checkpoint payload CRC failed'
    state=read_checkpoint_metadata(path)
    assert state['stage']=='RAW' and state['architecture']['gate']=='raw'
    assert state['variant']['corruption'] is False
    assert all(k in state for k in ('optimizer_state','scheduler_state','scaler_state','rng','fingerprint'))
    return state, dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path),
        payload_crc='passed', stage=state['stage'], update=state['update'],
        validation_score=state['validation_score'], best_validation_score=state['best_validation_score'],
        fingerprint=state['fingerprint'], tensor_count=len(state['model_state']),
        parameter_count=sum(math.prod(t['shape']) for t in state['model_state'].values()),
        strict_pytorch_load='not_run locally; enforced by Kaggle restore and training loader')


def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    best,best_audit=checkpoint_audit('best.pt')
    latest,latest_audit=checkpoint_audit('latest.pt')
    history=json.loads((SOURCE/'history.json').read_text())
    assert history==latest['history']
    assert [r['update'] for r in history]==list(range(1,latest['update']+1))
    assert all(math.isfinite(r['total']) for r in history)
    assert best['history']==history[:best['update']]
    assert best['fingerprint']==latest['fingerprint']
    split=latest['split_manifest']
    b5=json.loads((ROOT/'output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z/detailtrust/split_manifest.json').read_text())
    assert split==b5, 'Patient split differs from B5'
    cfg=dict(latest['config'])
    keys=('patch','batch_size','precision','seed','lr','gate_lr','weight_decay','branch_dropout',
          'utility_tau','utility_weight','corruption_probability','warmup_updates',
          'validation_every','val_subjects','synthetic')
    planned=cfg['updates_a']+cfg['updates_b']+cfg['updates_c']
    subjects=split['train'][:cfg['train_subjects']]
    payload=dict(config={k:cfg[k] for k in keys},variant=latest['variant'],split=split['sha256'],
                 stage='RAW',updates=planned,subjects=subjects)
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    assert fingerprint==latest['fingerprint'], 'Current training-fingerprint format differs'
    original_variants=list(cfg['variants'])
    cfg['variants']=['B0']
    recovery=dict(source='original B0 checkpoint metadata and supplied history.json',
                  original_selected_variants=original_variants,resume_selected_variants=['B0'],
                  selection_change='Only B0 is scheduled; its stage/config/variant fingerprint is unchanged.',
                  split_matches_b5=True,updates_planned=planned,updates_saved=latest['update'],
                  updates_remaining=planned-latest['update'],checkpoints=[best_audit,latest_audit],
                  reconstruction='Missing root config/split report recovered from checkpoint fields; no result.json or evaluation values invented.',
                  actual_gpu_execution=False)
    write_json(AUDIT/'checkpoint_inspection.json',recovery)
    # Temporary preparation stays inside the workspace; package copies out before cleanup.
    with tempfile.TemporaryDirectory(prefix='stage-',dir=str(AUDIT)) as staging:
        source=Path(staging).resolve()
        assert source.is_relative_to(ROOT.resolve())
        stage=source/'experiments/B0_s1729/RAW'
        stage.mkdir(parents=True)
        for filename in ('best.pt','latest.pt','history.json'):
            shutil.copy2(SOURCE/filename,stage/filename)
        write_json(source/'experiment_config.json',cfg)
        write_json(source/'split_manifest.json',split)
        write_json(source/'detailtrust_report.json',dict(schema_version='recovered_metadata_1',
            project=dict(name='Recovered B0 continuation metadata',synthetic=False),
            training_configuration=cfg,data_splits=split,
            training_history=[dict(row,source='B0_s1729/RAW/history.json') for row in history],
            experiment_status={'B0_s1729':dict(status='paused',reason='Recovered checkpoint at 2999/10000; no final evaluation available')},
            recovery_metadata=recovery,validation_metrics=None,test_metrics=None))
        manifest=package_resume(source,BUNDLE)
    assert list(manifest['progress'])==['experiments/B0_s1729/RAW/latest.pt']
    assert manifest['progress']['experiments/B0_s1729/RAW/latest.pt']['update']==2999
    for file in manifest['files']:
        assert sha256_file(BUNDLE/file['payload'])==file['sha256']
    nb=nbformat.read(ROOT/'detailtrust_kaggle.ipynb',as_version=4)
    found=False
    for cell in nb.cells:
        if cell.metadata.get('detailtrust_cell')=='configuration':
            cell.source=re.sub(r'^MODE = .*$', 'MODE = "main"',cell.source,flags=re.M)
            cell.source=re.sub(r'^EXPERIMENTS = .*$', 'EXPERIMENTS = ("B0",)  # recovered raw-input baseline only',cell.source,flags=re.M)
            cell.source=re.sub(r'^RESUME_DIRECTORY = .*$', 'RESUME_DIRECTORY = None',cell.source,flags=re.M)
            cell.source=re.sub(r'^RESUME_BUNDLE_ID = .*$',f'RESUME_BUNDLE_ID = {json.dumps(manifest["bundle_id"])}',cell.source,flags=re.M)
            found=True
    assert found
    nb.cells[0].source='''# B0 raw-input baseline: resume from update 2,999

This notebook continues the original raw-input residual U-Net for the remaining
7,001 of its planned 10,000 updates. It uses the same 218/49/101 patient split,
seed 1729 and 96-cubed patches as B5. B0 uses modality dropout without corruption
augmentation. Upload baseline_resume_input as a private Kaggle dataset and attach
it alongside the same BraTS20 dataset. Missing or mismatched input stops execution.

The shared tutorial retains DetailTrust component tests/profiling on disposable
models; the research experiment list contains B0 only. No B5 stage is retrained.
The test set stays sealed. B0 validation and robustness results are exported when
the selected checkpoint is evaluated. Its full selected checkpoint is under
experiments/B0_s1729/RAW/best.pt; B5-only export filenames are not produced.
'''
    nb.metadata['baseline_recovery']=dict(bundle_id=manifest['bundle_id'],progress=2999,
                                         planned_updates=planned,split_sha256=split['sha256'])
    for cell in nb.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            cell.outputs=[];cell.execution_count=None
    nbformat.validate(nb)
    nbformat.write(nb,NOTEBOOK)
    write_json(AUDIT/'preparation_checks.json',dict(status='passed',notebook=str(NOTEBOOK),
        cells=len(nb.cells),code_cells=sum(c.cell_type=='code' for c in nb.cells),
        source_unchanged=True,history_matches_latest=True,split_matches_b5=True,
        fingerprint_matches_current_code=True,bundle_hashes_verified=True,
        bundle_id=manifest['bundle_id'],bundle_bytes=sum(p.stat().st_size for p in BUNDLE.iterdir() if p.is_file()),
        cloud_execution='not_run',baseline_training_status='partial'))
    (ROOT/'BASELINE_CONTINUE.md').write_text('''# Continue the recovered B0 baseline

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
''',encoding='utf-8')
    print(json.dumps(recovery,indent=2))
    print('Prepared',NOTEBOOK,'and',BUNDLE)


if __name__=='__main__':
    main()
