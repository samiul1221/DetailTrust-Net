"""Render measured paper figures; keep analysis code outside the Overleaf folder.

Uses the two recorded evaluations. One additional CPU inference supplies the
baseline display planes; its complete-volume Dice must match saved patient rows.
No training, test-set access, synthetic MRIs, or invented curves are used.
"""
from pathlib import Path
import sys
import csv
import json
import hashlib
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PAPER = ROOT / 'DetailTrustNet_IEEE_Paper'
FIG = PAPER / 'figures'
AUDIT = ROOT / 'verification/paper_finalization'
B5 = ROOT / 'output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z/detailtrust'
B0 = ROOT / 'local_baseline_results'
SUBJECT = 'BraTS20_Training_205'
REGIONS = ('WT', 'TC', 'ET')
MODALITIES = ('T1', 'T1ce', 'T2', 'FLAIR')
CLASS_COLORS = ['black', '#CC79A7', '#E69F00', '#009E73']
manifest = {}
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def save(fig, name, sources, meaning):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300)
    manifest[name] = {'source_artifacts': sources, 'meaning': meaning,
                      'figure_size_inches': list(fig.get_size_inches()), 'fabricated_values': False}
    plt.close(fig)
    print('Saved measured figure:', name, flush=True)


def patient_planes():
    """Recompute one baseline volume and validate against its saved metrics."""
    import torch
    from detailtrust.config import RunConfig
    from detailtrust.data import discover_subjects, load_subject, prepare_subject
    from detailtrust.model import DetailTrustNet
    from detailtrust.evaluation import sliding_window
    from detailtrust.metrics import subject_metrics
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    destination = AUDIT / 'qualitative_source.npz'
    if destination.exists():
        return dict(np.load(destination))
    state = torch.load(ROOT / 'output_dump/B0_s1729/RAW/best.pt', map_location='cpu', weights_only=False)
    cfg = replace(RunConfig(**state['config']), precision='float32', run_final_test=False)
    model = DetailTrustNet(**state['architecture'])
    model.load_state_dict(state['model_state'], strict=True)
    model.eval().requires_grad_(False)
    records, _ = discover_subjects(ROOT / 'validation_data')
    images, labels, metadata = load_subject(next(r for r in records if r['id'] == SUBJECT))
    report = json.loads((B0 / 'local_baseline_report.json').read_text())
    proof = next(r for r in report['input_integrity'] if r['subject'] == SUBJECT)
    assert hashlib.sha256(images.tobytes()).hexdigest() == proof['canonical_mri_sha256']
    images, labels, metadata = prepare_subject(images, labels, metadata)
    print('CPU inference for the real baseline illustration:', SUBJECT, flush=True)
    with torch.inference_mode():
        result = sliding_window(model, images, np.ones(4, np.float32), cfg, torch.device('cpu'), stage='A')
        prediction = result['logits'].argmax(0).numpy()
    rows = subject_metrics(prediction, labels, metadata['spacing'], SUBJECT, 'B0_s1729', 'clean_m15', 'validation', 1.)
    prior = next(r for r in read_csv(B0 / 'metrics_per_subject.csv') if r['subject'] == SUBJECT and r['region'] == 'WT')
    assert abs(rows[0]['dice'] - float(prior['dice'])) < 1e-10
    source = dict(np.load(B5 / 'figures/BraTS20_Training_205_display_source.npz'))
    centre = (83, 77, 90)
    for axis, plane in ((2, 'Axial'), (1, 'Coronal'), (0, 'Sagittal')):
        truth_plane = np.take(labels, centre[axis], axis=axis).T
        assert np.array_equal(truth_plane, source[f'{plane}_Reference'])
        assert np.allclose(np.take(images[3], centre[axis], axis=axis).T, source[f'{plane}_FLAIR'], atol=1e-6)
        source[f'{plane}_B0'] = np.take(prediction, centre[axis], axis=axis).T.astype(np.uint8)
    for m, name in enumerate(MODALITIES):
        source[f'Axial_{name}'] = images[m, :, :, 90].T
    np.savez_compressed(destination, **source)
    (AUDIT / 'qualitative_provenance.json').write_text(json.dumps({
        'subject': SUBJECT, 'partition': 'validation', 'selection': 'first validation ID; not selected by score',
        'ras_indices': centre, 'slice_selection': 'median reference tumour coordinates',
        'baseline_full_volume_dice_verified_against_saved_result': True,
        'b5_reference_and_flair_planes_match_local_data': True,
        'labels_only_used_for_display_slice_selection': True}, indent=2))
    return source


def segmentation_overlay(ax, image, labels):
    ax.imshow(np.ma.masked_equal(image, 0), cmap=plt.get_cmap('gray').with_extremes(bad='black'), origin='lower', vmin=-1, vmax=3)
    ax.imshow(np.ma.masked_equal(labels, 0), cmap=ListedColormap(CLASS_COLORS),
              origin='lower', vmin=0, vmax=3, alpha=.72, interpolation='nearest')
    ax.set_axis_off()


def image_figures(source):
    legend = [Patch(color=CLASS_COLORS[i], label=s) for i, s in enumerate(('Background', 'NCR/NET', 'Oedema', 'ET')) if i]
    fig, axes = plt.subplots(1, 5, figsize=(7, 2.1), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .15, 1, .79))
    for ax, name in zip(axes[:4], MODALITIES):
        ax.imshow(np.ma.masked_equal(source[f'Axial_{name}'], 0), cmap=plt.get_cmap('gray').with_extremes(bad='black'), origin='lower', vmin=-1, vmax=3)
        ax.set_title(name); ax.set_axis_off()
    segmentation_overlay(axes[4], source['Axial_FLAIR'], source['Axial_Reference'])
    axes[4].set_title('Reference')
    fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False)
    save(fig, 'dataset_examples', ['validation_data', 'qualitative_provenance.json'], 'Real aligned modality inputs and reference; shared normalized intensity display range [-1,3].')
    fig, axes = plt.subplots(3, 5, figsize=(7, 4.3), layout='constrained',
                             gridspec_kw={'height_ratios': [240, 155, 155]})
    fig.get_layout_engine().set(rect=(0, .12, 1, .84))
    for r, plane in enumerate(('Axial', 'Coronal', 'Sagittal')):
        image = source[f'{plane}_FLAIR']; truth = source[f'{plane}_Reference']
        baseline, final = source[f'{plane}_B0'], source[f'{plane}_Final']
        axes[r, 0].imshow(np.ma.masked_equal(image, 0), cmap=plt.get_cmap('gray').with_extremes(bad='black'), origin='lower', vmin=-1, vmax=3)
        axes[r, 0].set_axis_off()
        for c, labels in enumerate((truth, baseline, final), 1):
            segmentation_overlay(axes[r, c], image, labels)
        old_wrong, new_wrong = baseline != truth, final != truth
        error = np.zeros(truth.shape, np.uint8)
        error[old_wrong & ~new_wrong] = 1
        error[~old_wrong & new_wrong] = 2
        error[old_wrong & new_wrong] = 3
        axes[r, 4].imshow(np.ma.masked_equal(image, 0), cmap=plt.get_cmap('gray').with_extremes(bad='black'), origin='lower', vmin=-1, vmax=3)
        axes[r, 4].imshow(np.ma.masked_equal(error, 0), cmap=ListedColormap(['black', '#0072B2', '#D55E00', '#999999']),
                          origin='lower', vmin=0, vmax=3, alpha=.85, interpolation='nearest')
        axes[r, 4].set_axis_off()
        axes[r, 0].text(-.05, .5, plane, transform=axes[r, 0].transAxes, rotation=90, va='center', ha='right')
    for ax, title in zip(axes[0], ('FLAIR', 'Reference', 'B0 baseline', 'DetailTrust-Net', 'Error change')):
        ax.set_title(title, fontsize=8.5)
    errors = [Patch(color=c, label=s) for c, s in (('#0072B2', 'B0 error only'), ('#D55E00', 'B5 error only'), ('#999999', 'Both wrong'))]
    fig.legend(handles=legend + errors, loc='lower center', ncol=3, frameon=False)
    save(fig, 'qualitative_results', ['qualitative_source.npz', 'B0 best.pt', 'B5 display_source.npz'], 'Three orthogonal reference-selected slices; errors include background and class errors; not representative cohort evidence.')
    fig, axes = plt.subplots(2, 4, figsize=(7, 4.1), layout='constrained')
    image = source['Axial_FLAIR']; gates = source['spatial_gates']; contributions = source['et_contribution_maps']
    limit = float(np.abs(contributions).max())
    for m, name in enumerate(MODALITIES):
        for r in (0, 1):
            axes[r, m].imshow(np.ma.masked_equal(image, 0), cmap=plt.get_cmap('gray').with_extremes(bad='black'), origin='lower', vmin=-1, vmax=3)
            axes[r, m].set_axis_off()
        gate = np.ma.masked_where((gates[m].T == 0) | (image == 0), gates[m].T)
        gm = axes[0, m].imshow(gate, cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=.8)
        cm = axes[1, m].imshow(np.ma.masked_where(image == 0, contributions[m].T), cmap='RdBu_r',
                            origin='lower', vmin=-limit, vmax=limit, alpha=.8)
        axes[0, m].set_title(name)
    fig.colorbar(gm, ax=list(axes[0]), shrink=.72, label='Gate coefficient q (not confidence)', pad=.01)
    fig.colorbar(cm, ax=list(axes[1]), shrink=.72, label='Signed ET-logit contribution', pad=.01)
    axes[0, 0].text(-.07, .5, 'Spatial gates', transform=axes[0, 0].transAxes, rotation=90, va='center')
    axes[1, 0].text(-.07, .5, 'ET contributions', transform=axes[1, 0].transAxes, rotation=90, va='center')
    save(fig, 'gate_contribution_maps', ['B5 display_source.npz'], 'Window-averaged axial maps; shared scales; image-zero and uncomputed crop fill masked; no percentile clipping.')


def comparison_figures():
    summary = read_csv(B0 / 'baseline_vs_b5_summary.csv')
    paired = read_csv(B0 / 'baseline_vs_b5_per_patient.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), layout='constrained')
    for field, marker, color, label in (('B0_dice', 's', '#0072B2', 'B0 baseline'), ('B5_dice', 'o', '#D55E00', 'DetailTrust-Net')):
        axes[0].plot(range(4), [float(r[field]) for r in summary], marker, color=color, label=label)
    axes[0].set(xticks=range(4), xticklabels=['WT', 'TC', 'ET', 'Mean'], ylim=(0, 1), ylabel='Validation patient macro Dice')
    axes[0].legend(loc='lower left', frameon=False)
    for i, region in enumerate(REGIONS + ('mean_WT_TC_ET',)):
        if i < 3:
            values = [100 * float(r['B5_minus_B0_dice']) for r in paired if r['region'] == region]
        else:
            lookup = {}
            for r in paired: lookup.setdefault(r['subject'], []).append(float(r['B5_minus_B0_dice']))
            values = [100 * np.mean(v) for v in lookup.values()]
        axes[1].scatter(i + np.linspace(-.12, .12, 49), values, s=7, alpha=.5, color='#0072B2', rasterized=True)
        row = summary[i]; mean = 100 * float(row['B5_minus_B0_dice'])
        lo, hi = [100 * float(row['pointwise_bootstrap_ci95_' + k]) for k in ('low', 'high')]
        axes[1].errorbar(i, mean, yerr=[[mean-lo], [hi-mean]], fmt='D', color='#D55E00', capsize=3)
    axes[1].axhline(0, color='black', lw=.7)
    axes[1].set(xticks=range(4), xticklabels=['WT', 'TC', 'ET', 'Mean'], ylabel='B5 minus B0 Dice\n(percentage points)')
    for ax in axes: ax.grid(axis='y', alpha=.2)
    axes[0].set_title('(a) Selected checkpoint scores')
    axes[1].set_title('(b) Paired patient differences')
    save(fig, 'baseline_comparison', ['baseline_vs_b5_summary.csv', 'baseline_vs_b5_per_patient.csv'], '49 matched validation patients; pointwise 95% paired percentile bootstrap with 20,000 draws, seed 20261014; unmatched training budgets.')
    internal = read_csv(ROOT / 'verification/run_20261014T062437Z_audit/internal_comparisons.csv')
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.7), layout='constrained')
    names = ('base_only', 'remove_detail_t1', 'remove_detail_t1ce', 'remove_detail_t2', 'remove_detail_flair')
    for ax, region in zip(axes, REGIONS):
        rows = [next(r for r in internal if r['region'] == region and r['reference_condition'] == 'clean_m15_' + n) for n in names]
        means = np.array([float(r['delta_mean']) for r in rows])*100
        lower = np.array([float(r['ci95_low']) for r in rows])*100
        upper = np.array([float(r['ci95_high']) for r in rows])*100
        ax.errorbar(means, range(5), xerr=np.stack((means-lower, upper-means)), fmt='o', capsize=2, color='#0072B2')
        ax.axvline(0, color='black', lw=.6)
        ax.set(yticks=range(5), yticklabels=['All', 'T1', 'T1ce', 'T2', 'FLAIR'], title=region,
               xlabel='Full minus removed\nDice (percentage points)')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    axes[0].set_ylabel('Detail contribution removed')
    save(fig, 'internal_corrections', ['internal_comparisons.csv'], 'Fixed B5 base and computed proposals; 49 paired patients; all removes every correction; not separately trained ablations or B0.')


def robustness_figures(rows):
    lookup = {(r['subject'], r['condition'], r['region']): float(r['dice']) for r in rows if r['region'] in REGIONS}
    patients = sorted({r['subject'] for r in rows})
    conditions = [f'clean_m{i:02d}' for i in range(1, 16)]
    values = np.array([[np.mean([lookup[p, c, region] for p in patients]) for c in conditions] for region in REGIONS])
    labels = ['+'.join(n for i,n in enumerate(('T1','C','T2','F')) if mask & (1<<i)) for mask in range(1,16)]
    fig, ax = plt.subplots(figsize=(7, 2.7), layout='constrained')
    mesh = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set(xticks=range(15), xticklabels=labels, yticks=range(3), yticklabels=REGIONS)
    plt.setp(ax.get_xticklabels(), rotation=55, ha='right', rotation_mode='anchor')
    for i in range(3):
        for j in range(15): ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if values[i,j]<.58 else 'black')
    fig.colorbar(mesh, ax=ax, label='Patient macro Dice', shrink=.85, pad=.015)
    save(fig, 'missing_modalities', ['B5 metrics_per_subject.csv'], 'All 15 nonempty modality subsets, 49 patients each; C=T1ce, F=FLAIR; B5 only, not a baseline robustness comparison.')
    corrupted = sorted({r['condition'] for r in rows if r['condition'].startswith(('noise', 'resolution'))})
    assert len(corrupted) == 8, corrupted
    labels = []
    for c in corrupted:
        parts = c.split('_')
        labels.append(('N' if 'noise' in c else 'R') + parts[0][-1] + ('/3' if '_m13_' in c else '/4'))
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.2), layout='constrained')
    for ax, region in zip(axes, REGIONS):
        for i, c in enumerate(corrupted):
            clean = 'clean_m13' if '_m13_' in c else 'clean_m15'
            delta = np.array([100*(lookup[p,c,region]-lookup[p,clean,region]) for p in patients])
            ax.scatter(i+np.linspace(-.12,.12,49),delta,s=6,alpha=.5,color='#0072B2',rasterized=True)
            ax.plot(i,delta.mean(),'D',color='#D55E00',ms=4)
        ax.axhline(0,color='black',lw=.6)
        ax.set(xticks=range(8),xticklabels=labels,title=region)
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right',rotation_mode='anchor')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Corrupted minus matching clean\nDice (percentage points)')
    save(fig,'corruption_robustness',['B5 metrics_per_subject.csv'],'49 patient differences per condition; N1/N2 noise .075/.15; R1/R2 resolution 2.5/3.5; /4 all inputs, /3 T1ce absent; local FLAIR corruption only.')


def training_curves():
    history = json.loads((B5 / 'training_history.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), layout='constrained')
    for ax, stage in zip(axes, 'ABC'):
        entries = [r for r in history if r['stage'] == stage and 'validation_dice' in r]
        # Read the actual logger key rather than interpreting per-step loss as Dice.
        if not entries: entries = [r for r in history if r['stage'] == stage and 'validation_score' in r]
        key = 'validation_dice' if entries and 'validation_dice' in entries[0] else 'validation_score'
        assert entries, f'Validation trajectory missing for {stage}'
        ax.plot([r['update'] for r in entries], [r[key] for r in entries], 'o-', color='#0072B2', lw=1, ms=3)
        ax.set(title=f'Stage {stage}', xlabel='Stage optimizer update', ylim=(0,1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Selection validation Dice\n(mean WT, TC, ET)')
    save(fig,'training_curves',['training_history.json'],'All three completed stage trajectories; stage-specific optimizer counters, different active models/objectives, no held-out test evidence.')


def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    source = patient_planes()
    image_figures(source)
    comparison_figures()
    robustness_figures(read_csv(B5 / 'metrics_per_subject.csv'))
    training_curves()
    (AUDIT / 'generated_figures.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
