"""Real CPU stop/restore/continue test; offline upload and cache-filter checks."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import torch
import nbformat
import subprocess
import sync_kaggle as sync
from detailtrust.config import configuration, experiment_registry, write_json
from detailtrust.data import SyntheticStore, make_split
from detailtrust.model import build_model
from detailtrust.resume import restore_resume_bundle
from detailtrust import training
from detailtrust.experiments import run_experiment_sweep


def expect_failure(action):
    try:
        action()
    except (AssertionError, RuntimeError, SystemExit):
        return
    raise AssertionError('Expected a fail-closed error')


def main():
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='detailtrust-resume-test-') as temporary:
        root = Path(temporary)
        original, restored = root/'original', root/'restored'
        original.mkdir(); restored.mkdir()
        cfg = configuration('smoke', output_dir=str(original), synthetic=True, patch=(16,16,16),
                            precision='float32', updates_a=1, updates_b=2, updates_c=1,
                            validation_every=1, epoch_updates=1, max_hours=1)
        store = SyntheticStore(count=20, size=16)
        split = make_split([dict(id=s) for s in store.records], cfg.split_seed)
        write_json(original/'experiment_config.json', asdict(cfg))
        write_json(original/'split_manifest.json', split)
        write_json(original/'detailtrust_report.json', {'training_configuration': asdict(cfg)})
        variant = experiment_registry()['B5']
        model = build_model(variant, False)
        shared = original/'experiments/shared_candidates_s1729'
        train_ids, val_ids = split['train'][:cfg.train_subjects], split['validation'][:1]
        training.run_training_stage(model, store, train_ids, val_ids, split, cfg, variant,
                                    'A', 1, shared/'A', torch.device('cpu'), float('inf'))
        clock = [0.]
        real_update = training.train_update
        def pause_after_update(*args, **kwargs):
            terms = real_update(*args, **kwargs)
            clock[0] = 2.
            return terms
        with patch.object(training.time, 'perf_counter', side_effect=lambda: clock[0]), \
                patch.object(training, 'train_update', side_effect=pause_after_update):
            try:
                training.run_training_stage(model, store, train_ids, val_ids, split, cfg, variant,
                    'B', 2, shared/'B', torch.device('cpu'), 1., stream_offset=1)
                raise AssertionError('Expected pause')
            except training.SessionBudgetReached:
                pass
        manifest = sync.package_resume(original, root/'input/bundle')
        assert manifest['progress']['experiments/shared_candidates_s1729/B/latest.pt']['update'] == 1
        expect_failure(lambda: restore_resume_bundle('missing', restored, cfg, root/'input'))
        expect_failure(lambda: restore_resume_bundle(manifest['bundle_id'], restored, replace(cfg, updates_b=3), root/'input'))
        assert not list(restored.rglob('*.pt'))
        restored_cfg = replace(cfg, output_dir=str(restored))
        restore_resume_bundle(manifest['bundle_id'], restored, restored_cfg, root/'input')
        expect_failure(lambda: restore_resume_bundle(manifest['bundle_id'], restored, restored_cfg, root/'input'))
        calls = []
        def tracked_update(*args, **kwargs):
            calls.append(args[4])  # stage
            return real_update(*args, **kwargs)
        with patch.object(training, 'train_update', side_effect=tracked_update):
            statuses = run_experiment_sweep(store, split, restored_cfg, torch.device('cpu'))
        assert calls == ['B', 'C'], calls
        assert statuses['B5_s1729']['status'] == 'completed', statuses['B5_s1729'].get('reason')
        assert len(json.loads((restored/'experiments/shared_candidates_s1729/B/history.json').read_text())) == 2
        # Corrupt input and verify hashes reject it before restoration.
        payload = root/'input/bundle'/manifest['files'][0]['payload']
        payload.write_bytes(b'corrupt')
        expect_failure(lambda: restore_resume_bundle(manifest['bundle_id'], root/'bad', cfg, root/'input'))
        # No live API call: exercise the complete default upload flow and attachment.
        notebook = root/'notebook.ipynb'
        cell = nbformat.v4.new_code_cell('MODE = "smoke"\nRESUME_BUNDLE_ID = None')
        cell.metadata['detailtrust_cell'] = 'configuration'
        nbformat.write(nbformat.v4.new_notebook(cells=[cell]), notebook)
        (root/'kaggle-sync.json').write_text(json.dumps({'kernel':'user/existing','last_pushed_version':2}))
        def fake_cli(args, env, timeout=180):
            if '--help' in args:
                return subprocess.CompletedProcess(args,0,'--accelerator','')
            assert args[:2] == ['kernels','push']
            folder=Path(args[3]); metadata=json.loads((folder/'kernel-metadata.json').read_text())
            assert 'user/resume-dataset' in metadata['dataset_sources']
            generated=nbformat.read(folder/notebook.name,as_version=4)
            assert manifest['bundle_id'] in generated.cells[0].source
            return subprocess.CompletedProcess(args,0,'Kernel version 3 successfully pushed. https://www.kaggle.com/code/user/existing','')
        with patch.object(sync,'ROOT',root), patch.object(sync,'build_notebook',return_value=notebook), \
             patch.object(sync.sys,'argv',['sync_kaggle.py','--mode','smoke','--run']), \
             patch.dict(sync.os.environ,{'KAGGLE_API_TOKEN':'fake-test-token'}), \
             patch.object(sync,'resolve_kernel',return_value=('user/existing',True)), \
             patch.object(sync,'resume_source',return_value=original), \
             patch.object(sync,'publish_resume_dataset',return_value='user/resume-dataset'), \
             patch.object(sync,'run_kaggle',side_effect=fake_cli):
            sync.main()
    write_json('verification/continuation_checks.json', dict(status='passed', live_api_executed=False,
        checks=['paused B checkpoint packaged/restored','no A optimizer updates on continuation',
                'only remaining B/C updates executed','evaluation completed on synthetic fixture',
                'missing/mismatched/corrupted bundle rejected','rerun cannot overwrite session progress',
                'default upload attaches resume dataset and injects bundle ID']))
    print('Continuation checks passed; synthetic CPU only, no cloud upload.')


if __name__ == '__main__':
    main()
