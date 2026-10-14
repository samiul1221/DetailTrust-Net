"""Audit the completed validation allocation and compare recorded patient outcomes.

No network access, PyTorch deserialization, training, or raw-output modification.
Run audit_downloaded_run.py for this dump before this script.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT/'output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z'
DATA = DUMP/'detailtrust'
OUT = ROOT/'verification/run_20261014T062437Z_audit'
REGIONS = ('WT', 'TC', 'ET')
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_REPLICATES = 20000


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def inspect_all_files():
    """Read/hash every file; check checkpoint ZIP payload CRCs without unpickling."""
    records = []
    for path in sorted(p for p in DUMP.rglob('*') if p.is_file()):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        record = dict(path=str(path.relative_to(DUMP)), bytes=path.stat().st_size,
                      sha256=digest.hexdigest(), status='read_and_hashed')
        if path.suffix in ('.json', '.log'):
            value = read_json(path)
            record.update(parsed_type=type(value).__name__, parsed_items=len(value))
            if path.suffix == '.log':
                record['last_timestamp_seconds'] = max(x['time'] for x in value)
                record['fatal_error_markers'] = [x['time'] for x in value
                    if any(s in x.get('data','') for s in ('Traceback (most recent call last)', 'CUDA out of memory'))]
        elif path.suffix == '.md':
            record['text_lines'] = len(path.read_text(encoding='utf-8').splitlines())
        elif path.suffix == '.pt':
            with zipfile.ZipFile(path) as archive:
                failed = archive.testzip()
                assert failed is None, (path, failed)
                record.update(zip_members=len(archive.namelist()), payload_crc='passed',
                              model_deserialization='not_run: local PyTorch unavailable')
        records.append(record)
    write_json(OUT/'all_file_inventory.json', records)
    return records


def patient_dice(rows, condition):
    values = {}
    for row in rows:
        if row['condition'] == condition and row['region'] in REGIONS:
            key = (row['subject'], row['region'])
            assert key not in values
            values[key] = row['dice']
    return values


def paired_comparison(rows, without_condition, subjects, bootstrap_indices):
    """Estimand: mean within-patient Dice difference, full minus branch removal."""
    full = patient_dice(rows, 'clean_m15')
    without = patient_dice(rows, without_condition)
    assert set(full) == set(without) == {(p,r) for p in subjects for r in REGIONS}
    x = np.asarray([[full[p,r] for r in REGIONS] for p in subjects])
    y = np.asarray([[without[p,r] for r in REGIONS] for p in subjects])
    x = np.column_stack((x, x.mean(1)))
    y = np.column_stack((y, y.mean(1)))
    delta = x-y
    bootstrap = delta[bootstrap_indices].mean(1)
    results = []
    for index, region in enumerate((*REGIONS, 'WT_TC_ET_mean')):
        values = delta[:,index]
        low, high = np.quantile(bootstrap[:,index], [.025,.975])
        results.append(dict(reference_condition=without_condition, region=region,
            patients=len(subjects), full_mean=float(x[:,index].mean()),
            reference_mean=float(y[:,index].mean()), delta_mean=float(values.mean()),
            delta_median=float(np.median(values)), delta_std=float(values.std(ddof=1)),
            ci95_low=float(low), ci95_high=float(high),
            improved_patients=int((values>1e-12).sum()),
            worsened_patients=int((values < -1e-12).sum()),
            tied_patients=int((np.abs(values)<=1e-12).sum())))
    return results, delta


def comparison_plot(results, delta):
    """Truthful internal comparison with patient pairs, not an independent baseline."""
    selected = [r for r in results if r['reference_condition']=='clean_m15_base_only']
    fig, axes = plt.subplots(1,2,figsize=(7.0,3.8), layout='constrained')
    fig.get_layout_engine().set(rect=(0,.15,1,.85))
    positions = np.arange(4)
    labels = ['WT','TC','ET','Mean']
    axes[0].plot(positions, [r['reference_mean'] for r in selected], 's', color='#0072B2', label='B5: corrections disabled')
    axes[0].plot(positions, [r['full_mean'] for r in selected], 'o', color='#D55E00', label='B5: full model')
    axes[0].set(xticks=positions, xticklabels=labels, ylim=(0,1), ylabel='Validation patient macro Dice')
    axes[0].legend(fontsize=7,loc='lower left')
    for i,r in enumerate(selected):
        axes[1].scatter(i+np.linspace(-.13,.13,len(delta)), delta[:,i]*100,
                        s=7,alpha=.65,color='#0072B2')
        mean=r['delta_mean']*100
        axes[1].errorbar(i,mean,yerr=[[mean-r['ci95_low']*100],[r['ci95_high']*100-mean]],
                         fmt='D',color='#D55E00',capsize=3)
    axes[1].axhline(0,color='black',lw=.7)
    axes[1].set(xticks=positions,xticklabels=labels,ylabel='Full minus disabled\nDice change (percentage points)')
    for ax in axes: ax.grid(axis='y',alpha=.2)
    fig.suptitle('Internal correction removal: same B5 weights, 49 validation patients',fontsize=10)
    fig.text(.5,.025,'Dots: patient differences; diamonds: means; bars: 95% paired bootstrap intervals.\nExploratory validation; fixed B5 weights; no independent B0 baseline.',ha='center',fontsize=7)
    for extension in ('png','pdf','svg'):
        fig.savefig(OUT/f'internal_correction_comparison.{extension}',dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    report = read_json(DATA/'detailtrust_report.json')
    experiment = read_json(DATA/'experiments/B5_s1729/result.json')
    split = read_json(DATA/'split_manifest.json')
    audit = read_json(OUT/'audit.json')
    files = inspect_all_files()
    subjects = sorted(split['validation'])
    assert len(subjects)==49 and not audit['metrics']['issues']
    assert experiment['status']=='completed' and experiment['validation']['status']=='completed'
    assert len(experiment['validation']['conditions'])==23
    assert all(c['status']=='completed' for c in experiment['validation']['conditions'])
    for result in experiment['validation']['results']:
        assert set(result['completed_subjects'])==set(subjects)
    assert audit['checkpoint_hash_matches']
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(subjects),size=(BOOTSTRAP_REPLICATES,len(subjects)))
    comparisons=[]
    for condition in ('clean_m15_base_only', *(f'clean_m15_remove_detail_{m}' for m in ('t1','t1ce','t2','flair'))):
        rows,delta=paired_comparison(report['per_subject_metrics'],condition,subjects,indices)
        comparisons.extend(rows)
        if condition=='clean_m15_base_only': plot_delta=delta
    comparison_plot(comparisons,plot_delta)
    with (OUT/'internal_comparisons.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    baseline_candidates=[]
    for path in (ROOT/'output_dump').rglob('result.json'):
        if path.parent.name.startswith(('B0_','B1_')):
            baseline_candidates.append(str(path))
    baseline_status=dict(status='blocked',reason='No local B0/B1 checkpoint, result.json or patient-level outcomes found; older 20261013T065126Z folder is empty.',
                         actual_baseline_candidates=baseline_candidates,
                         performed_comparison='B5 full versus the same jointly trained B5 with corrections disabled',
                         independent_baseline_equivalence=False)
    metadata=dict(bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=BOOTSTRAP_REPLICATES,
        interval='pointwise 95% percentile interval for paired patient mean difference',
        independent_unit='patient; WT/TC/ET resampled together',
        assumptions='Validation subjects are treated as independent and representative of this internal cohort; single fixed seed/checkpoint.',
        limitations=['Post-selection exploratory validation analysis; not independent confirmatory inference.',
                    'Intervals condition on trained weights and exclude training-seed variability.',
                    'Pointwise intervals are not multiplicity-adjusted; no hypothesis-test or superiority claim.',
                    'Disabled corrections do not turn B5 into raw-input B0 or independently trained B1.'])
    write_json(OUT/'comparison_results.json',dict(internal_comparisons=comparisons,statistical_method=metadata,baseline=baseline_status))
    summaries=report['region_metrics']
    lookup={(s['condition'],s['region']):s for s in summaries}
    lines=['# Completed B5 validation run: audit and verdict','',f'Source: `{DUMP}`. Raw outputs were not modified.','',
           '**B5 training and its configured validation evaluation are complete. Final test evaluation and independent baseline experiments are not complete.**','',
           '## Evidence integrity','',
           f"Read and SHA-256 hashed all {len(files)} files: {dict(Counter(Path(f['path']).suffix for f in files))}.",
           'All seven PyTorch checkpoint containers passed ZIP-payload CRC checks without pickle deserialization. The exported best-model SHA-256 matches the report. Kaggle records a strict model reload; local PyTorch reload was not performed.',
           f"All {len(audit['json'])} JSON files parsed. All four CSV tables were checked, including 9,604 patient-condition-class/region rows, 196 summary rows, 980 gate rows and 980 correction rows. Independently recomputed {audit['metrics']['recomputed_means']} means and {audit['metrics']['recomputed_other_statistics']} other summary statistics: zero discrepancy. No invalid ranges, duplicate records or split-membership issues were found.",
           'All 18 distinct PNG figures were visually reviewed. Their 16 PDF and 16 SVG counterparts and the NPZ display-source arrays were also checked. All 13 saved unit checks passed. Full inventory: all_file_inventory.json; metric audit: audit.json.','',
           '## Completion','',
           'A 4,000/4,000; B 4,500/4,500; C 1,500/1,500. This session restored all stages and performed zero remaining research-training updates. The checkpoint hash is unchanged from the preceding dump.',
           'All 15 nonempty modality subsets and all eight validation corruptions completed all 49 patients. Clean component-removal analyses also completed all 49. Test status is not_run because run_final_test=False and protocol_locked=False. B0-B4 and A1-A7 remain not_run because they were not selected.',
           'Requested experiments complete=True follows this definition: selected B5 validation complete, with test allowed to remain intentionally not_run. It does not mean the full research protocol or ablation study is complete.','',
           '## Clean segmentation results','',
           'All metrics are patient macro values on a custom 49-patient validation split, selected using that validation cohort. Train=218; test=101; split seed=20261012. These are not official challenge splits. One model seed, 1729.','',
           '| Region | Dice mean +/- SD | IoU | Precision | Sensitivity | HD95 mm (valid n) | Surface Dice at 1 mm |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for region in REGIONS:
        s=lookup['clean_m15',region]
        lines.append(f"| {region} | {s['dice']['mean']:.4f} +/- {s['dice']['std']:.4f} | {s['iou']['mean']:.4f} | {s['precision']['mean']:.4f} | {s['recall']['mean']:.4f} | {s['hd95_mm']['mean']:.2f} ({s['hd95_mm']['valid_count']}) | {s['surface_dice']['mean']:.4f} |")
    lines += ['', 'WT/TC/ET mean Dice is 0.737904. Whole tumour is the strongest region; tumour core and enhancing tumour are more variable. Exclusive NCR/NET Dice is 0.3770. Four ET-absent reference cases have false-positive ET predictions (Dice zero, undefined HD95), and some ET-positive cases also score poorly. Keep these cases in reports. Surface Dice here uses unweighted 6-connected boundary voxel centres; it is not a surfel-area-weighted challenge implementation.','',
              '## Internal comparison available now','',
              '**This is not an independently trained raw-input baseline.** The B5 backbone was jointly trained with corrections, receives smoothed input, and has its branches disabled only at inference.','',
              '| Outcome | Corrections disabled | Full B5 | Difference, percentage points | Pointwise paired bootstrap 95% interval | Improved / worsened / tied patients |',
              '|---|---:|---:|---:|---|---|']
    for s in comparisons[:4]:
        lines.append(f"| {s['region']} | {s['reference_mean']:.4f} | {s['full_mean']:.4f} | {s['delta_mean']*100:+.3f} | [{s['ci95_low']*100:+.3f}, {s['ci95_high']*100:+.3f}] | {s['improved_patients']} / {s['worsened_patients']} / {s['tied_patients']} |")
    lines += ['', 'The largest mean gain is ET (about 2.01 percentage points); mean WT/TC/ET gains about 0.844 points. T1ce corrections account for most of the branch-removal effect. T2 removal very slightly improves macro Dice. Intervals are exploratory, condition on these weights, do not account for validation selection or seed variation, and cannot replace a controlled B0/B1/B4 comparison. All contrasts and raw paired-difference plots are retained.','',
              '## Modality absence','', '| Available modalities | WT Dice | TC Dice | ET Dice | Mean Dice |', '|---|---:|---:|---:|---:|']
    for bits in range(1,16):
        values=[lookup[f'clean_m{bits:02}',r]['dice']['mean'] for r in REGIONS]
        label='+'.join(m for i,m in enumerate(('T1','T1ce','T2','FLAIR')) if bits&(1<<i))
        lines.append('| '+label+' | '+' | '.join(f'{x:.4f}' for x in values)+f' | {np.mean(values):.4f} |')
    lines += ['', 'The unweighted average across all 15 subsets is descriptive, not a model of clinical missingness. Removing T1ce while retaining T1/T2/FLAIR reduces ET Dice from 0.6736 to 0.2915 and mean Dice from 0.7379 to about 0.5380. The model tolerates absence unequally; broad missing-modality robustness is not demonstrated by one model without comparators.','',
              '## Spatial corruption: paired with matching clean availability','',
              '| Condition | WT change, pp | TC change, pp | ET change, pp | Mean change, pp |', '|---|---:|---:|---:|---:|']
    for c in experiment['validation']['conditions']:
        condition=c['condition'];name=condition['id']
        if condition['corruption'] is None: continue
        bits=sum(a<<i for i,a in enumerate(condition['available']))
        values=[(lookup[name,r]['dice']['mean']-lookup[f'clean_m{bits:02}',r]['dice']['mean'])*100 for r in REGIONS]
        lines.append('| '+name+' | '+' | '.join(f'{x:+.4f}' for x in values)+f' | {np.mean(values):+.4f} |')
    lines += ['', 'These eight tests corrupt only FLAIR: local normalized-image Gaussian noise at 0.075/0.15 or axis-2 blur/downsample/upsample factors 2.5/3.5, with all modalities or T1ce absent. The location is seeded independently per patient/condition; it is not tumour-targeted. Zero background is preserved. Noise changes average Dice negligibly; strongest resolution degradation reduces four-modality mean Dice by about 0.094 points. This supports stability under this restricted simulation, not gate-specific benefit, real scanner shifts, arbitrary corruptions or clinical reliability. Gate-versus-utility diagnostics exist only for clean all-modality cases, so adaptive gate responses to corruption were not directly measured.','',
              '## Utility-gate verdict','',
              'Stage B selected 0.740310 with gates fixed to one. Stage C selected 0.737904 after gate learning: a reduction of 0.241 percentage points on clean validation. Stage C is nearly flat across its validation checkpoints. Whole-brain gates are close to 0.5; T1ce gates increase locally in tumours. Associations with measured utility are weak or mixed. These observations do not establish that utility supervision adds value. B4 segmentation-only and B2 unit-gate evaluations under the same robustness suite are needed to isolate that claim.',
              'All 49 fixed-context stability enumeration checks passed over 16 correction subsets on sampled voxels, maximum numerical margin discrepancy 1.90735e-6. High stability can include incorrect predictions and is dominated by background; it is not a diagnostic guarantee.','',
              '## Independent baseline comparison: blocked by missing evidence','',
              baseline_status['reason'],
              'A partial baseline validation score from an old log is insufficient: there is no locally available B0 checkpoint, final selected score, or matched patient-level evaluation table. Do not compare an old intermediate maximum with selected B5 as a completed experiment. The Stage A low-frequency model is also not B0.',
              'To recover the older run, open its entry in Kaggle Versions and download its saved outputs from that version if they are still available. Do not rely on the current --version launcher option for recovery: the locally installed Kaggle client does not forward version numbers in its output request, and version-specific recovery has not been validated. Check which variant/checkpoint actually exists before comparing. If B0 training is partial, either resume it to the matched planned budget or explicitly label a compute-limited pilot; evaluate on the same split/conditions. A fair raw-input baseline plus the B4 gate control is still required for strong method claims. No full baseline training or live retrieval was performed by this local audit.','',
              '## Figure review','',
              'All measured plots retain patient-level points, missing counts and source-data sidecars. The availability heatmap now has n=49 in every cell and the new corruption plot has matched patient differences. Qualitative panels depict one prespecified validation patient, not the test set or a representative cohort. They include persistent distant false positives.',
              'Before paper use: the Stage C utility BCE curve is omitted because plotting requests utility while histories log utility_bce; the actual loss is recorded. Debug figures use a different label palette without a class legend. Zero gate margins outside the image crop are reconstruction fill, not low trust. Contribution maps would be easier to read with MRI overlays. Corruption-axis labels are small at IEEE column width; use a two-column layout or shorter labels.',
              'The plot called Base is B5 with corrections disabled, not separately trained B0. Label it that way. Raw datasets/qualitative assets are real, not fabricated. Existing manuscript placeholders/status must be updated from this completed validation record, while independent baseline/test entries must remain pending.','',
              '## Practical next step and overall verdict','',
              'No further default B5 training or validation continuation is necessary. Repeating the default command now mainly repeats setup/checks/profiling and reloads completed work; it does not enable the test set or create a baseline.',
              '**Engineering verdict:** successful complete validation allocation, verified artifacts, genuine nonzero detail effects.',
              '**Scientific verdict:** a working model with moderate, uneven validation segmentation and small internal detail gains. The utility-supervision improvement hypothesis is currently unsupported, and clean validation slightly favors unit gates. No claim of superiority, state of the art, external generalization or clinical robustness is warranted.',
              'Fix the evaluation protocol before opening the 101-patient test split. Obtain a genuine baseline checkpoint/evaluation and preferably B4 controls. Report negative outcomes and one-seed limitations. If protocol changes or model tuning are needed, continue using validation only; keep test data untouched.','',
              '## File-level inventory','']
    lines += [f"- `{f['path']}`: {f['status']}; {f['bytes']} bytes." for f in files]
    (OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(files),baseline=baseline_status,internal_comparison=comparisons[:4],review=str(OUT/'REVIEW.md')),indent=2))


if __name__=='__main__':
    main()
