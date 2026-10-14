"""Publication figure candidates: measured data, patient units, explicit missingness."""
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import MODALITIES, REGIONS, write_json


def save_paper_figure(fig, directory, name, caption, source):
    """Export 300-DPI PNG plus vector PDF/SVG and a caption/source-data sidecar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 9}):
        for suffix in ('png', 'pdf', 'svg'):
            path = directory/f'{name}.{suffix}'
            fig.savefig(path, dpi=300, facecolor='white')
            paths.append(str(path))
    plt.close(fig)
    write_json(directory/f'{name}.json', dict(caption=caption, source=source,
               formats=paths, journal_requirements='Not specified; verify target journal before submission'))
    return dict(status='saved', files=paths, caption=caption)


def distribution_panel(ax, rows, metric, categories, category_key, ylabel, bounded=False):
    """Each dot is a measured patient record; undefined values are counted explicitly."""
    for index, category in enumerate(categories, 1):
        group = [r for r in rows if r[category_key] == category]
        values = [r[metric] for r in group if r.get(metric) is not None and np.isfinite(r[metric])]
        if values:
            ax.boxplot([values], positions=[index], widths=.45, showfliers=False,
                       medianprops={'color': '#D55E00'})
            ax.scatter(index + np.linspace(-.16, .16, len(values)), values, s=9,
                       color='#0072B2', alpha=.65)
        ax.text(index, 1.01, f'n={len(values)}\nNA={len(group)-len(values)}',
                transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
    ax.set_xticks(range(1, len(categories)+1), categories, rotation=25)
    ax.set_ylabel(ylabel)
    if bounded:
        ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=.2)


def plot_training_panels(history, directory):
    """Separate stages because their objectives and update indices differ."""
    artifacts = {}
    for experiment in sorted({r['experiment'] for r in history}):
        rows = [r for r in history if r['experiment'] == experiment]
        stages = list(dict.fromkeys(r['stage'] for r in rows))
        for name, keys, label in (
            ('training_objectives', ('total', 'segmentation', 'base', 'branch_auxiliary', 'utility'), 'Loss (stage-specific)'),
            ('validation_trajectory', ('validation_dice',), 'Validation patient macro Dice'),
            ('learning_rate', ('learning_rate',), 'Learning rate')):
            fig, axes = plt.subplots(1, len(stages), figsize=(max(5, 4*len(stages)), 3.8), layout='constrained', squeeze=False)
            for ax, stage in zip(axes[0], stages):
                group = [r for r in rows if r['stage'] == stage]
                for key in keys:
                    values = [r for r in group if r.get(key) is not None]
                    if values:
                        ax.plot([r['update'] for r in values], [r[key] for r in values],
                                label=key, lw=.7, marker='o' if key == 'validation_dice' else None, markersize=3)
                ax.set(title=f'{experiment}: Stage {stage}', xlabel='Optimizer update within stage', ylabel=label)
                if name == 'validation_trajectory':
                    ax.set_ylim(0, 1)
                ax.legend(fontsize=7); ax.grid(alpha=.2)
            caption = ('Raw logged values without smoothing. Stages have separate axes/objectives. '
                       'Validation points are measured checkpoints only; lines guide the eye. '
                       f'Incomplete training records: {any(r.get("incomplete", False) for r in rows)}.')
            artifacts[f'{experiment}_{name}'] = save_paper_figure(fig, directory, f'{experiment}_{name}', caption,
                [{k: r[k] for k in ('stage', 'update', *keys) if k in r} for r in rows])
    return artifacts


def plot_accuracy_panels(rows, directory, prefix):
    clean = [r for r in rows if r['condition'] == 'clean_m15' and r['region'] in REGIONS]
    artifacts = {}
    if not clean:
        return artifacts
    for name, metrics in (
        ('overlap', [('dice', 'Dice'), ('iou', 'IoU')]),
        ('boundaries', [('hd95_mm', 'HD95 (mm)'), ('surface_dice', 'Surface Dice')]),
        ('detection', [('precision', 'Voxel precision'), ('recall', 'Voxel sensitivity')])):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout='constrained')
        for ax, (metric, label) in zip(axes, metrics):
            distribution_panel(ax, clean, metric, list(REGIONS), 'region', label, metric != 'hd95_mm')
        fig.suptitle(prefix + ': clean four-modality evaluation')
        caption = ('Dots are patients, boxes show quartiles and median, whiskers use 1.5 IQR. '
                   'NA counts undefined metrics; empty-case handling is in metrics_per_subject.csv. '
                   'Surface Dice uses the saved physical tolerance and unweighted boundary voxel centres. '
                   'These distributions describe patients, not training-seed uncertainty.')
        artifacts[name] = save_paper_figure(fig, directory, prefix+'_'+name, caption, clean)
    et = [r for r in clean if r['region'] == 'ET' and r.get('et_volume_mm3', 0) > 0]
    if et:
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        ax.scatter([r['et_volume_mm3']/1000 for r in et], [r['dice'] for r in et], color='#0072B2', marker='o')
        ax.set(xscale='log', xlabel='Reference ET volume (mL, log scale)', ylabel='ET Dice', ylim=(0, 1), title=prefix)
        artifacts['et_size'] = save_paper_figure(fig, directory, prefix+'_et_size',
            f'One point per ET-positive patient (n={len(et)}); ET-absent patients excluded from the log-volume axis only. No fitted trend.', et)
    return artifacts


def plot_robustness_panels(rows, directory, prefix):
    """Availability heatmap shows counts; corruption curves use matched patients."""
    artifacts = {}
    selected = [r for r in rows if re.fullmatch(r'clean_m\d{2}', r['condition']) and r['region'] in REGIONS]
    if selected:
        matrix = np.full((3, 15), np.nan)
        counts = np.zeros((3, 15), int)
        for i, region in enumerate(REGIONS):
            for bits in range(1, 16):
                values = [r['dice'] for r in selected if r['region'] == region and r['condition'] == f'clean_m{bits:02}']
                if values:
                    matrix[i, bits-1], counts[i, bits-1] = np.mean(values), len(values)
        fig, ax = plt.subplots(figsize=(13, 4.8), layout='constrained')
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap=plt.get_cmap('viridis').with_extremes(bad='#cccccc'), aspect='auto')
        labels = ['+'.join(MODALITIES[m] for m in range(4) if bits & (1 << m)) for bits in range(1, 16)]
        ax.set_xticks(range(15), labels, rotation=65, ha='right'); ax.set_yticks(range(3), list(REGIONS))
        for i in range(3):
            for j in range(15):
                text = f'{matrix[i,j]:.2f}\nn={counts[i,j]}' if counts[i,j] else 'not run'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color='black' if not counts[i,j] or matrix[i,j] > .55 else 'white')
        fig.colorbar(im, ax=ax, label='Patient macro Dice'); ax.set_title(prefix + ': modality availability')
        artifacts['missing_modalities'] = save_paper_figure(fig, directory, prefix+'_missing_modalities',
            'Each cell averages measured patients for that region/subset; n is shown. Gray means not evaluated. Unequal n indicates incomplete evaluation; do not interpret such cells as matched comparisons.', selected)
    damaged = [r for r in rows if not r['condition'].startswith('clean') and r['region'] in REGIONS]
    if damaged:
        conditions = sorted({r['condition'] for r in damaged})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
        sources = []
        for ax, region in zip(axes, REGIONS):
            for i, condition in enumerate(conditions):
                bits = re.search(r'_m(\d+)', condition).group(1)
                clean = {r['subject']: r['dice'] for r in rows if r['region'] == region and r['condition'] == 'clean_m'+bits}
                paired = [r for r in damaged if r['region'] == region and r['condition'] == condition and r['subject'] in clean]
                delta = [r['dice'] - clean[r['subject']] for r in paired]
                if delta:
                    ax.scatter(i + np.linspace(-.12, .12, len(delta)), delta, s=8, color='#0072B2')
                    ax.plot(i, np.mean(delta), 'D', color='#D55E00')
                sources += [dict(r, matched_clean_dice=clean[r['subject']]) for r in paired]
            ax.axhline(0, color='black', lw=.7); ax.set_xticks(range(len(conditions)), conditions, rotation=75, ha='right')
            ax.set(title=region, ylabel='Corrupted minus matched clean Dice')
        artifacts['corruption'] = save_paper_figure(fig, directory, prefix+'_corruption',
            'Dots are paired patient differences with identical modality availability; diamonds are their means. No unmatched patients are included. Simulated corruptions do not establish clinical robustness.', sources)
    return artifacts


def plot_interpretability_panels(records, directory, prefix, experiment, partition):
    """No mixing of patients across splits, seeds, regions, or corruption settings."""
    artifacts = {}
    def select(key):
        return [r for r in records[key] if r['experiment'] == experiment and r['split'] == partition
                and r['condition'] == 'clean_m15' and r['region'] == 'brain']
    gates, corrections, stability = select('gates'), select('corrections'), select('stability')
    if gates:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        distribution_panel(axes[0], gates, 'mean', MODALITIES, 'modality', 'Mean brain gate per patient', True)
        for modality, marker in zip(MODALITIES, ('o', 's', '^', 'D')):
            group = [r for r in gates if r['modality'] == modality and r.get('measured_gated_utility_mean') is not None]
            axes[1].scatter([r['mean'] for r in group], [r['measured_gated_utility_mean'] for r in group], marker=marker, label=modality, s=15)
        axes[1].axhline(0, color='black', lw=.6)
        axes[1].set(xlabel='Mean brain gate per patient', ylabel='CE without correction minus CE with correction', xlim=(0, 1))
        axes[1].legend(fontsize=8)
        artifacts['gate_utility'] = save_paper_figure(fig, directory, prefix+'_gate_utility',
            'Patient summaries on brain voxels. Positive CE difference indicates benefit in the fixed inference context. Gate scores are not calibrated probabilities; this plot is not causal evidence.', gates)
    if corrections:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        distribution_panel(axes[0], corrections, 'absolute_contribution_mean', MODALITIES, 'modality', 'Mean absolute foreground logit contribution')
        for k, label in enumerate(('NCR/NET', 'edema', 'ET')):
            for m, modality in enumerate(MODALITIES):
                vals = [r['signed_contribution_mean'][k] for r in corrections if r['modality'] == modality and r.get('signed_contribution_mean') is not None]
                if vals:
                    axes[1].scatter(np.full(len(vals), m+(k-1)*.2), vals, s=10, marker=('o','s','^')[k], label=label if m == 0 else None)
        axes[1].axhline(0, color='black', lw=.7); axes[1].set_xticks(range(4), MODALITIES)
        axes[1].set_ylabel('Signed mean logit contribution per patient'); axes[1].legend(fontsize=8)
        artifacts['contributions'] = save_paper_figure(fig, directory, prefix+'_contributions',
            'One value per patient/modality (and class on the right), using the independently blended additive contributions. Magnitude alone does not imply benefit.', corrections)
    if stability:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        valid = [r for r in stability if r.get('stable_accuracy') is not None]
        axes[0].scatter([r['stable_fraction'] for r in valid], [r['stable_accuracy'] for r in valid], s=15)
        axes[0].set(xlabel='Conditionally stable brain-voxel fraction', ylabel='Accuracy within stable voxels', xlim=(0, 1), ylim=(0, 1))
        axes[1].scatter([r['repaired_base_errors'] for r in stability], [r['introduced_errors'] for r in stability], s=15)
        axes[1].set(xlabel='Repaired base errors (voxels/patient)', ylabel='Introduced errors (voxels/patient)')
        artifacts['stability'] = save_paper_figure(fig, directory, prefix+'_stability',
            'Each point is one patient. Stability is conditional on fixed base logits and correction maps; it is not correctness or robustness to re-encoding a missing modality. Raw counts depend on brain volume.', stability)
    return artifacts


def plot_detail_removal(rows, directory, prefix):
    clean = {(r['subject'], r['region']): r['dice'] for r in rows if r['condition'] == 'clean_m15' and r['region'] in REGIONS}
    conditions = ['clean_m15_base_only'] + ['clean_m15_remove_detail_'+m for m in MODALITIES]
    selected = [r for r in rows if r['condition'] in conditions and (r['subject'], r['region']) in clean]
    if not selected:
        return {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    source = []
    for ax, region in zip(axes, REGIONS):
        for i, condition in enumerate(conditions):
            group = [r for r in selected if r['region'] == region and r['condition'] == condition]
            changes = [clean[(r['subject'], region)] - r['dice'] for r in group]
            ax.scatter(i + np.linspace(-.12, .12, len(changes)), changes, s=8)
            source += [dict(r, full_model_dice=clean[(r['subject'], region)]) for r in group]
        ax.axhline(0, color='black', lw=.6); ax.set_xticks(range(5), ['all', *MODALITIES], rotation=35)
        ax.set(title=region, xlabel='Detail corrections disabled', ylabel='Full minus disabled Dice')
    return {'detail_removal': save_paper_figure(fig, directory, prefix+'_detail_removal',
            'Paired patient Dice differences after disabling frozen contributions. Positive means the full correction improved Dice. This is an inference intervention, not a retrained ablation.', source)}


def plot_ablation_comparisons(rows, directory):
    """Display multiple measured variants on their shared clean patients only."""
    artifacts = {}
    for partition in sorted({r['split'] for r in rows}):
        clean = [r for r in rows if r['split'] == partition and r['condition'] == 'clean_m15' and r['region'] in REGIONS]
        experiments = sorted({r['experiment'] for r in clean})
        if len(experiments) < 2:
            continue
        patients = set.intersection(*({r['subject'] for r in clean if r['experiment'] == ex} for ex in experiments))
        matched = [r for r in clean if r['subject'] in patients]
        if not matched:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(max(10, len(experiments)*2), 4.5), layout='constrained')
        for ax, region in zip(axes, REGIONS):
            distribution_panel(ax, [r for r in matched if r['region'] == region], 'dice', experiments, 'experiment', region+' Dice', True)
        artifacts[partition+'_variant_comparison'] = save_paper_figure(fig, directory, partition+'_variant_comparison',
            'Measured variants on intersecting clean patients only. One dot per patient/model, not independent training replicates. Check training budgets and experiment status before treating this as a controlled ablation.', matched)
    return artifacts or {'variant_comparison': dict(status='not_produced', reason='At least two measured variants with matched patients required')}


def generate_paper_figures(records, output_dir):
    directory = Path(output_dir)/'figures'
    artifacts = plot_training_panels(records['history'], directory)
    artifacts.update(plot_ablation_comparisons(records['rows'], directory))
    groups = sorted({(r['experiment'], r['split']) for r in records['rows']})
    expected = ('overlap', 'boundaries', 'detection', 'et_size', 'missing_modalities', 'corruption', 'gate_utility', 'contributions', 'stability', 'detail_removal')
    for experiment, partition in groups:
        rows = [r for r in records['rows'] if r['experiment'] == experiment and r['split'] == partition]
        prefix = re.sub(r'[^A-Za-z0-9_-]', '_', experiment+'_'+partition)
        result = plot_accuracy_panels(rows, directory, prefix)
        result.update(plot_robustness_panels(rows, directory, prefix))
        result.update(plot_interpretability_panels(records, directory, prefix, experiment, partition))
        result.update(plot_detail_removal(rows, directory, prefix))
        for name in expected:
            artifacts[prefix+'_'+name] = result.get(name, dict(status='not_produced', reason='Required measured records unavailable'))
    if not groups:
        artifacts['evaluation_figures'] = dict(status='not_produced', reason='No completed patient evaluations; training curves only')
    write_json(directory/'paper_figures_manifest.json', artifacts)
    (directory/'README.md').write_text('Figure candidates for manuscript preparation\n\nPNG: 300 DPI; PDF/SVG: vector graphics. Each JSON sidecar contains the caption and plotted source records.\nValidation/test and experiments are kept separate. Missing measurements are not fabricated.\nReview the run status and patient counts before publication. Journal-specific requirements remain unchecked.\n', encoding='utf-8')
    return artifacts
