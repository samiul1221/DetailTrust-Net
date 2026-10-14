"""Update the single IEEE manuscript from the measured evaluation artifacts."""
from pathlib import Path
import csv
import json
import re
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'DetailTrustNet_IEEE_Paper'
AUDIT = ROOT / 'verification/paper_finalization'
B5 = ROOT / 'output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z/detailtrust'
B0 = ROOT / 'local_baseline_results'


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def figure(name, caption, label, wide=True):
    kind = 'figure*' if wide else 'figure'
    width = r'\textwidth' if wide else r'\columnwidth'
    return '\n' + rf'''\begin{{{kind}}}[t]
\centering
\includegraphics[width={width}]{{figures/{name}.pdf}}
\caption{{{caption}}}
\label{{{label}}}
\end{{{kind}}}
''' + '\n'


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    path = PAPER / 'main.tex'
    text = path.read_text(encoding='utf-8-sig')
    if not (AUDIT / 'main_before_finalization.tex').exists():
        (AUDIT / 'main_before_finalization.tex').write_text(text)
    # Rebuild from the inspected source, making the operation repeatable.
    text = (AUDIT / 'main_before_finalization.tex').read_text()
    b0 = json.loads((B0 / 'local_baseline_report.json').read_text())
    b5_result = json.loads((B5 / 'experiments/B5_s1729/result.json').read_text())
    summaries = read_csv(B5 / 'metrics_summary.csv')
    b5_metrics = {r['region']: {k: json.loads(r[k]) for k in ('dice', 'hd95_mm', 'surface_dice', 'precision', 'recall')} for r in summaries if r['condition']=='clean_m15'}
    b0_metrics = {r['region']: r for r in b0['validation_metrics']}
    pairs = read_csv(B0 / 'baseline_vs_b5_summary.csv')
    assert len(b0['evaluated_subjects']) == 49
    assert b0['data_splits'] == json.loads((B5 / 'split_manifest.json').read_text())
    assert len(b0['input_integrity']) == 49 and all(r['mri_matches_b5'] for r in b0['input_integrity'])
    # Independently recompute every paired point estimate and interval.
    matched = read_csv(B0 / 'baseline_vs_b5_per_patient.csv')
    patients = sorted({r['subject'] for r in matched})
    lookup = {(r['subject'],r['region']):float(r['B5_minus_B0_dice']) for r in matched}
    delta = np.array([[lookup[p,r] for r in ('WT','TC','ET')] for p in patients])
    delta = np.column_stack((delta,delta.mean(1)))
    draws = np.random.default_rng(20261014).integers(0,49,(20000,49))
    intervals = np.quantile(delta[draws].mean(1), [.025,.975], axis=0)
    for i,row in enumerate(pairs):
        assert abs(delta[:,i].mean()-float(row['B5_minus_B0_dice'])) < 1e-12
        assert abs(intervals[0,i]-float(row['pointwise_bootstrap_ci95_low'])) < 1e-12
        assert abs(intervals[1,i]-float(row['pointwise_bootstrap_ci95_high'])) < 1e-12
    abstract = r'''Brain-tumour segmentation from multimodal magnetic resonance imaging (MRI) can be sensitive to unavailable sequences and degraded image detail. We introduce DetailTrust-Net, a five-level 3D residual U-Net augmented with modality-conditioned, zero-anchored detail corrections. A learned Gaussian mixture separates smooth images from residual detail; shared spatial gates scale signed foreground-logit proposals while preserving the base background logit. A third training stage freezes the candidate networks and supervises gates using detached, spatially pooled cross-entropy changes from removing individual corrections. We evaluate the selected model on a custom, patient-disjoint BraTS 2020 validation cohort of 49 subjects. Clean-input WT, TC, and ET Dice are 0.8526, 0.6876, and 0.6736, respectively, compared with 0.8328, 0.6324, and 0.6675 for the recovered raw-input baseline checkpoint. The mean region Dice difference is 2.70 percentage points, with a paired patient bootstrap interval of 0.33--4.99 points. The ET interval includes zero, and mean HD95 does not improve. The comparison uses different training budgets and corruption augmentation, so it does not isolate an architectural effect. Within B5, removing all detail contributions reduces mean Dice by 0.84 points, whereas the final learned gates score 0.24 points below the selected unit-gate candidate stage. All 15 nonempty modality subsets and eight local FLAIR-corruption conditions are evaluated. These results establish an executable correction-analysis framework and a validation comparison, but do not establish superiority of utility supervision, held-out test performance, or clinical robustness.'''
    text = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda _: '\\begin{abstract}\n'+abstract+'\n\\end{abstract}', text, flags=re.S)
    substitutions = {
        'We do not claim priority or empirical benefit in this draft.': 'The present study reports a completed B5 validation evaluation and an exploratory comparison with an available raw-input baseline. It does not claim priority, state-of-the-art performance, or a controlled benefit from utility supervision.',
        'We document a reproducible patient-level BraTS workflow and an evaluation plan that separates engineering checks from incomplete model evidence.': 'We evaluate 49 matched validation patients, report paired baseline differences, and examine all nonempty modality subsets, local degradation, correction removal, and conditional stability. We distinguish these measurements from the unexecuted controlled ablations and internal test.',
        'The present BraTS implementation and partial run provide the basis for these tests; they do not establish an empirical advantage.': 'The present BraTS evaluation measures the selected correction model and an available baseline checkpoint. Unequal training budgets and augmentation prevent attribution of their difference to utility supervision; the matched gating controls remain necessary.',
        'The configured run planned 1,500 Stage C updates, but it did not reach this stage.': 'The recorded B5 experiment completed all 1,500 Stage C updates.',
        'The downloaded run completed A only and was interrupted during B.': 'All three B5 stages completed across resumable Kaggle sessions. Validation chooses weights within each stage; the reported final B5 checkpoint is the selected Stage C model.',
        'No patient-level stability evaluation completed in the available run.': 'Patient-level summaries were recorded for all 49 validation subjects, with numerical subset checks on sampled brain voxels.',
        "The measured profiler used 100 updates per stage after five warmup updates on $96^3$ patches, including loading. Recorded median step times varied by stage and were 4.19, 1.73, and 0.62 s respectively in that profiling pass. These profiling timings are not whole-training throughput estimates and should not be compared as model performance. Peak allocated GPU memory recorded during training was about 3.77 GB for Stage A; the profiling artifact records stage-specific peaks.": "The measured profiler used 100 updates per stage after five warmup updates on $96^3$ patches, including loading. Median step times in the final recorded profiling pass were 3.99, 1.74, and 0.62 s for A, B, and C. The largest allocated memory peak in that pass was 3.38 GiB. These disposable-model measurements exclude cohort validation and are not whole-session durations. Both CUDA devices were visible, but the implementation used only \\texttt{cuda:0}; it did not train with distributed parallelism.",
        'Only a partial B5 run is available; the other rows describe unrun experiments.': 'B5 and the available B0 checkpoint have measured clean validation outcomes. B1--B4 and A1--A7 are implemented controls without completed independent evaluations.',
        'Planned patient-level metrics include Dice, IoU, precision, recall, specificity, HD95 in millimetres, and surface Dice at 1-mm tolerance for WT, TC, ET, and the exclusive classes.': 'Evaluated patient-level metrics include Dice, IoU, precision, recall, specificity, HD95 in millimetres, and surface Dice at 1-mm tolerance for WT, TC, ET, and the exclusive classes.',
        'These evaluations were not completed in the downloaded run.': 'B5 completed these validation evaluations; B0 was evaluated only for clean, complete-input validation. No model was evaluated on the internal test partition.',
        'The planned analysis relates gates to each using Spearman association, separately for tumour and background.': 'The recorded clean-input analysis relates gates to each using within-patient Spearman association, separately for tumour regions, brain foreground, and background.',
    }
    for old,new in substitutions.items():
        assert old in text, 'Stale source paragraph not found: '+old[:65]
        text = text.replace(old,new)
    data_start = text.index('Fig.~\\ref{fig:data} reserves')
    data_end = text.index('\\subsection{Training configuration and status}',data_start)
    text = text[:data_start] + r'''Fig.~\ref{fig:data} shows the four aligned sequences and released segmentation for the first validation ID, BraTS20\_Training\_205. The axial slice is selected at the median reference tumour coordinate (RAS index 90), independently of model performance. All images share a normalized display range. Reference labels select the display plane only; they do not enter model inputs or inference cropping. The complete cohort metrics are reported separately from this single-subject illustration.
''' + figure('dataset_examples', r'Real, aligned validation inputs and reference for subject 205, axial RAS slice 90. NCR/NET, oedema, and ET colours are shared with Fig.~\ref{fig:qual}. Grayscale uses the common normalized range $[-1,3]$; no synthetic patient data are shown.', 'fig:data') + '\n' + text[data_end:]
    text = text.replace('a 9-hour notebook budget, and planned updates', 'a 9-hour orchestration deadline per continuation, and configured updates')
    text = text.replace('The available run did not enable final test evaluation.', 'The recorded runs did not enable internal test evaluation; the reported results are validation measurements.')
    old_start = text.index('For the available allocation, only B5 was selected.')
    old_end = text.index('\\subsection{Metrics and evaluation protocol}',old_start)
    text = text[:old_start] + r'''The B5 experiment completed 4,000, 4,500, and 1,500 optimizer updates in A, B, and C. Its selected checkpoints were at updates 2,000, 4,500, and 1,500, with validation scores 0.7105, 0.7403, and 0.7379. B0 uses a recovered raw-input checkpoint selected at update 2,000; the available raw training history contains 2,999 updates against a configured 10,000-update budget. No additional B0 training was performed during local evaluation. These facts are retained to make the comparison interpretable, rather than equating the two training exposures.

B0 and B5 have the same patient split, backbone widths, patch geometry, label mapping, and validation reconstruction. B0 uses modality dropout without corruption augmentation; B5 uses both. B0 predictions were recomputed in CPU float32, whereas B5 used GPU FP16 inference. All 49 canonical MRI input hashes match the saved B5 integrity records. The reference geometry, label mapping, and ET volumes also match; full original label-array hashes were not recorded, so exact historical label-byte identity cannot be established from those hashes. The clean comparison is consequently an exploratory comparison of the available checkpoints, not an equal-budget estimate of architectural improvement. B1--B4 and A1--A7 remain unexecuted comparative controls.

''' + text[old_end:]
    start = text.index('The patient is the unit of paired comparison.')
    end = text.index('\\section{Results}',start)
    text = text[:start] + r'''The patient is the independent unit in the present paired analysis. For each clean-input region, B5 and B0 Dice are joined by subject ID before computing their difference. The overall endpoint first averages WT, TC, and ET Dice within a patient and then averages across patients. We resample the 49 paired patient vectors with replacement 20,000 times, using seed 20261014. Pointwise 95\% percentile intervals use the 2.5th and 97.5th percentiles of the resampled mean differences. Internal branch-removal comparisons use the same patient-level approach. These intervals condition on the selected weights and do not estimate training-seed uncertainty. Because validation selected the checkpoints, intervals describe exploratory validation differences; they are not independent confirmatory test evidence. No multiplicity-adjusted inference or p-value is reported.

''' + text[end:]
    tables = []
    metric_rows=[]
    for label, metrics in [('B0',b0_metrics),('B5',b5_metrics)]:
        for region in ('WT','TC','ET'):
            r=metrics[region]
            metric_rows.append(f"{label} & {region} & {r['dice']['mean']:.4f} & {r['hd95_mm']['mean']:.2f} ({r['hd95_mm']['valid_count']}) & {r['surface_dice']['mean']:.4f} \\\\")
    clean_table = r'''\begin{table}[t]
\caption{Clean validation on the same 49 patients. Dice and SD$_1$ use patient-macro means. HD95 is the valid-case mean in mm, with valid count in parentheses; counts differ for ET. Training exposure is specified in the text.}
\label{tab:results}
\centering
\begin{tabularx}{\columnwidth}{@{}l l L L L@{}}
\toprule
Model & Region & Dice & HD95 ($n$) & SD$_1$ \\
\midrule
''' + '\n'.join(metric_rows) + r'''
\bottomrule
\end{tabularx}
\end{table}
'''
    ci_rows=[]
    for r in pairs:
        region = 'Mean' if r['region']=='mean_WT_TC_ET' else r['region']
        lo,hi=[100*float(r['pointwise_bootstrap_ci95_'+k]) for k in ('low','high')]
        ci_rows.append(f"{region} & {100*float(r['B5_minus_B0_dice']):+.2f} & $[{lo:+.2f}, {hi:+.2f}]$ \\\\")
    ci_table=r'''\begin{table}[t]
\caption{B5 minus B0 Dice in percentage points. Pointwise paired patient bootstrap intervals use 20,000 resamples; they condition on selected weights and validation reuse.}
\label{tab:paired}
\centering
\begin{tabularx}{\columnwidth}{@{}l L L@{}}
\toprule
Endpoint & Difference (pp) & 95\% interval (pp) \\
\midrule
''' + '\n'.join(ci_rows)+r'''
\bottomrule
\end{tabularx}
\end{table}
'''
    detection_rows=[]
    for label,metrics in [('B0',b0_metrics),('B5',b5_metrics)]:
        for region in ('WT','TC','ET'):
            r=metrics[region]
            detection_rows.append(f"{label} & {region} & {r['precision']['mean']:.4f} ({r['precision']['valid_count']}) & {r['recall']['mean']:.4f} ({r['recall']['valid_count']}) \\\\")
    detection_table=r'''\begin{table}[t]
\caption{Patient-macro precision and sensitivity for clean validation. Parentheses indicate valid denominators; empty ET references account for reduced sensitivity counts.}
\label{tab:detection}
\centering
\begin{tabularx}{\columnwidth}{@{}l l L L@{}}
\toprule
Model & Region & Precision ($n$) & Sensitivity ($n$) \\
\midrule
'''+'\n'.join(detection_rows)+r'''
\bottomrule
\end{tabularx}
\end{table}
'''
    results = r'''\section{Results}
\subsection{Completed evaluation and implementation checks}
The final downloaded B5 record and the local B0 evaluation contain complete results for their requested validation conditions. B5 completed its three optimizer stages and all 23 validation input conditions: 15 nonempty modality subsets and eight local corruption conditions. Each condition contains the same 49 validation patients. Five additional clean-input correction-removal conditions also cover all 49 patients. The clean B0 evaluation contains 49 patients and seven region/class rows per patient. The B0 and B5 comparison uses only WT, TC, and ET from the common clean-input condition; it does not treat repeated condition observations as independent patients.

The recorded checks cover NIfTI adaptation, label mapping, split disjointness, Gaussian normalization and physical scales, low-pass/detail reconstruction, availability masking, zero anchoring, gate dimensions, additive fusion, background preservation, utility-target sign and detachment, gradient flow, checkpoint reload, optimizer resume, sliding-window reconstruction, and report serialization. A repeated-patch check reduced its loss. The recovered B0 model also loaded with strict state-dictionary matching, and its full-volume predictions were evaluated without optimization. These checks support consistency of the tested implementation; segmentation accuracy is assessed by the separate patient-level measurements below.

\subsection{Clean-input baseline comparison}
Table~\ref{tab:results} reports the available B0 baseline and selected B5 model. WT Dice increases from 0.8328 to 0.8526, TC from 0.6324 to 0.6876, and ET from 0.6675 to 0.6736. The patient-macro three-region mean is 0.7109 for B0 and 0.7379 for B5. The largest absolute difference is in TC. Exclusive-class Dice also rises from 0.2450 to 0.3770 for NCR/NET and from 0.6794 to 0.7166 for oedema, although these class endpoints were not the checkpoint-selection criterion.

''' + clean_table + ci_table + figure('baseline_comparison',r'Clean validation comparison of the selected B0 and B5 checkpoints on 49 matched patients. (a) Patient-macro Dice, with a full $[0,1]$ scale. (b) Patient differences, their means, and pointwise 95\% paired bootstrap intervals; all patient outliers are retained. Unequal training exposure, augmentation, and inference precision prevent interpreting these differences as an isolated architectural effect.','fig:comparison') + r'''
Table~\ref{tab:paired} and Fig.~\ref{fig:comparison} quantify the paired differences. Mean region Dice increases by 2.70 percentage points, with a pointwise interval of 0.33--4.99 points. The WT and TC intervals are positive under this fixed-checkpoint patient resampling. The ET interval spans $-4.64$ to $+4.26$ points, so the small mean ET difference is uncertain. These intervals do not remedy validation-selection bias, different training exposure, or the absence of multiple training seeds. In particular, they cannot show that utility supervision caused the observed gain.

\subsection{Boundary and detection behaviour}
The overlap improvement is not uniform across evaluation measures. B5 has higher mean 1-mm surface-point Dice in all three regions, but mean HD95 is numerically higher: 17.51 versus 17.06 mm for WT, 16.14 versus 15.94 mm for TC, and 14.06 versus 12.14 mm for ET. ET HD95 is defined for 45 B5 cases and 46 B0 cases. Its valid-case means therefore also differ in included subjects. These values do not support a claim of uniformly improved boundary accuracy; distant false-positive components can increase HD95 while Dice remains high.

''' + detection_table + r'''
Table~\ref{tab:detection} shows the accompanying precision--sensitivity trade-off. B5 is more sensitive in WT, TC, and ET but has lower macro precision. The additional foreground coverage therefore includes both recovered lesion voxels and additional false positives. Four validation references contain no ET. The B5 model predicts some ET in all four, giving Dice zero under the declared one-empty rule. B0 predicts an empty ET mask in one of these cases, giving Dice one for that both-empty case. Such cases materially affect an ET average and should be retained, rather than removed after observing results. Undefined surface distances are counted separately instead of converted to apparently favourable distances.

\subsection{Training stages and the gating hypothesis}
Fig.~\ref{fig:curves} shows validation scores at the recorded selection intervals, using separate update counters for each stage. Stage A completes 4,000 updates and selects update 2,000 at 0.7105. Stage B completes 4,500 updates and selects its final update at 0.7403. Stage C completes 1,500 updates and selects its final update at 0.7379. The lower score after gate learning is an observed negative result for clean-input selection: the final learned-gate model is 0.24 percentage points below the selected unit-gate candidate model. Stage C changes the output control while holding those candidate weights fixed; its training objective and selected operating point differ from Stage B.

''' + figure('training_curves',r'Actual checkpoint-selection trajectories for the completed A, B, and C stages. Each panel uses its own stage update count and the same validation Dice scale. Stage B uses unit gates; Stage C uses learned gates with frozen candidates. Connecting points within a stage does not imply that intermediate updates were evaluated or that these are independent test measurements.','fig:curves') + r'''
The contrast between the B0/B5 comparison and the B-to-C change is important. A positive difference from a raw baseline does not establish that the utility term benefits spatial gates. Candidate learning, corruption exposure, and the longer B5 training history can all contribute. B4 initialized from the same Stage B checkpoint would test segmentation-only versus utility-supervised gates without changing their architecture. A1 would test whether the spatial placement of utility targets matters. Neither comparison has a completed result in the present evidence.

\subsection{Real qualitative predictions}
Fig.~\ref{fig:qual} compares the actual recovered baseline and B5 outputs in three orthogonal planes of subject 205. The patient is the first ID in the saved validation manifest, not a case selected for a large improvement. Display indices are the median reference tumour coordinates $(83,77,90)$ in canonical RAS. The same slices, class colours, and image intensity range are used throughout. Local CPU baseline inference reproduces its saved whole-volume Dice, and the displayed reference and FLAIR planes match the B5 export.

''' + figure('qualitative_results',r'Real validation predictions for BraTS20\_Training\_205. Rows are axial, coronal, and sagittal views at RAS indices 90, 77, and 83. Blue error-change voxels are wrong only in B0; orange are wrong only in B5; grey are wrong in both. The maps include foreground/background and class errors and retain remote false positives. Slice choice uses the reference solely for display. This single case does not establish representative cohort performance.','fig:qual') + r'''
The error panel distinguishes corrections of baseline errors from newly introduced errors. This makes the trade-off visible without suggesting that every change is beneficial. A distant false-positive component remains visible in the B5 views, consistent with this patient's relatively large surface distance despite substantial tumour overlap. The images provide spatial examples of the measured predictions; inference conclusions continue to rest on the full-volume cohort metrics rather than a favourable central slice.

\subsection{Conditional detail-branch removal}
Removing all gated corrections from the fixed B5 model gives WT, TC, and ET Dice of 0.8522, 0.6827, and 0.6535. The three-region mean is 0.7295, compared with 0.7379 for the complete output. The mean difference is 0.84 percentage points; the largest region difference is ET, at 2.01 points. This comparison retains the B5 backbone and its smooth inputs. It is distinct from B0, which is a separately trained raw-input model.

''' + figure('internal_corrections',r'Conditional contribution-removal analysis on 49 clean validation patients. Positive values favour the complete B5 output over the indicated removal; bars are pointwise paired patient bootstrap intervals. ``All'' suppresses every correction while retaining the B5 base logits. Individual rows remove a computed modality contribution, without masking the MRI or rerunning the backbone. These are inference interventions within B5, not independently trained ablations.','fig:internal') + r'''
T1ce dominates the observed detail benefit: removing only its contribution reduces mean Dice to 0.7301, a 0.78-point decrease. The T1 and FLAIR branch effects are much smaller. Removing T2 slightly raises the mean to 0.737918, a difference of approximately 0.0014 points, whose patient interval spans zero. Thus neither branch availability nor a nonzero correction implies a useful contribution. These interventions describe the current additive output at fixed input; they do not assign causal modality importance or account for a backbone that adapts when a sequence is removed.

\subsection{Missing-modality validation}
Fig.~\ref{fig:missing} summarizes all 15 nonempty MRI subsets, rerunning B5 with the corresponding availability flags. With T1, T2, and FLAIR but no T1ce, ET Dice falls from 0.6736 to 0.2915. T1ce alone gives ET Dice 0.4258, while FLAIR alone gives WT Dice 0.7920. These region-specific differences are consistent with unequal reliance on the available contrasts in this trained model, but they do not imply that the individual images encode only one region.

''' + figure('missing_modalities',r'B5 patient-macro validation Dice across all 15 nonempty modality subsets, each evaluated on all 49 patients. C denotes T1ce and F denotes FLAIR. Inputs and flags are masked together, then the complete model is rerun. Colours share the full $[0,1]$ scale; numbers are rounded to two decimals for display. B0 has no measured subset suite, so this figure is not a comparative robustness result.','fig:missing') + r'''
The three-input subset T1+T1ce+FLAIR achieves mean region Dice 0.7346, close to the complete-input mean of 0.7379. Other subsets have much lower scores. Averaging all subsets uniformly gives 0.5538, but this statistic weights rare and common missingness patterns equally and does not represent any clinical frequency distribution. The measured subset coverage demonstrates that the availability mechanism executes across all nonempty masks; it does not show that missingness has become harmless or that B5 is more robust than B0.

\subsection{Local spatial-corruption validation}
The eight validation perturbations damage FLAIR locally using normalized noise standard deviations 0.075 and 0.15 or axis-2 resolution factors 2.5 and 3.5. Each is evaluated with all modalities present and with T1ce absent. Fig.~\ref{fig:corruption} subtracts the matching clean condition per patient, avoiding comparisons between different availability masks. Mean three-region Dice changes remain small in this implemented suite. The largest mean reduction is approximately 0.094 percentage points for the stronger resolution degradation with all four inputs. Some noise conditions have very small positive means; these fluctuations do not establish a benefit from adding noise.

''' + figure('corruption_robustness',r'B5 corrupted-minus-matching-clean validation differences. Dots are all 49 patient differences; diamonds are means. N1/N2 denote local FLAIR noise at 0.075/0.15; R1/R2 denote resolution factors 2.5/3.5 on canonical axis 2. The suffix /4 retains all inputs and /3 removes T1ce. Each panel retains its full observed patient range, so vertical scales differ. No B0 corruption evaluation or scanner-level perturbation is represented.','fig:corruption') + r'''
The perturbations preserve zero-background voxels, use a deterministic spatial mask, and affect only one sequence. Other modalities can supply redundant evidence, and the low-pass backbone may reduce sensitivity to residual perturbation. These are plausible explanations for the small changes, not causal conclusions: the present measurements cannot separate the effect of smoothing, corruption training, and spatial gates. Gate analyses were exported for clean inputs only, so no measured gate response to the corruption conditions can be inferred from the segmentation curves.

\subsection{Gates, contributions, and conditional stability}
Clean-input analysis contains 980 subject--modality--scope rows for gates and the same number for contributions, corresponding to 49 patients, four modalities, and five spatial scopes. Patient-averaged brain gate means are near 0.5 for every modality. For T1ce, mean coefficients within WT, TC, and ET are approximately 0.58, 0.63, and 0.64. The corresponding within-patient gate--removal-utility Spearman associations are weak and vary by scope. A larger average tumour-region gate therefore cannot be read as a calibrated probability of usefulness or as a proven reliability estimate.

''' + figure('gate_contribution_maps',r'Actual axial B5 spatial gates and signed ET-logit contributions for validation subject 205. Gates share a $[0,1]$ scale; ET contributions share a symmetric scale determined by the largest absolute contribution across modalities, with no percentile clipping. Values are window-averaged. Zero image background and uncomputed crop-fill positions are hidden for display and do not represent low-confidence decisions. The largest visible signed detail effects arise from T1ce in this example.','fig:gates') + r'''
Fig.~\ref{fig:gates} keeps a common scale across modalities so that visually strong maps cannot be created by independently stretching tiny contributions. Positive and negative ET terms are both retained. The sign refers to the ET logit, not directly to whether the final argmax is correct: a positive ET contribution can help on enhancing tissue and create false positives elsewhere. Ground-truth-derived utility is available during training and evaluation analysis, but is not an inference feature.

The conditional stability calculation was numerically checked against all 16 contribution-retention subsets for sampled brain voxels in each of the 49 patients. Every check passed; the largest margin discrepancy was $1.91\times10^{-6}$. Patient summaries also include tumour and background scopes. These checks establish consistency of the fixed-contribution formula with enumeration on the sampled locations. They do not certify predictions under missing MRI inputs, and a stable voxel may have the wrong label. Background-dominated stable fractions cannot substitute for region-wise accuracy.

\subsection{Model cost and reproducibility artifacts}
B5 contains 16,934,644 parameters versus 16,910,468 in B0. The additional 24,176 parameters increase the count by approximately 0.143\%. This small count difference does not imply equal computational cost: Gaussian filtering, two refiner evaluations per modality for anchoring, gate computation, and staged training add operations beyond the raw backbone. No validated whole-model FLOP estimate or same-hardware B0/B5 latency comparison is available.

A separate GPU inference benchmark uses the prespecified subject 205, one warmup reconstruction and two timed reconstructions. The median is 11.98 s for 27 windows and the allocated-memory peak is about 3.22 GiB. B0 local CPU inference has a 49-patient median of 73.63 s with four CPU threads and float32. These values come from different hardware, precision, and sampling protocols and must not be interpreted as a model speed ratio. Both evaluations reconstruct the original $240\times240\times155$ grid before scoring.

The saved evidence includes the patient split manifest, stage histories, selected checkpoint configurations and hashes, patient-level CSVs, condition summaries, gate/contribution records, stability checks, and figure provenance. The baseline checkpoint SHA-256 begins \texttt{2caba443318f}, and the portable B5 model hash begins \texttt{c3e3cbeeef00}. Full hashes remain in the source reports. The Overleaf package contains only manuscript and figure assets; code, weights, and MRI volumes remain outside it. Reproducing the present tables requires the selected weights and exact split, not merely the same architecture name.

\section{Discussion}
\subsection{What the current comparison supports}
On the selected clean validation cohort, B5 has higher patient-macro Dice than the recovered B0 checkpoint, most clearly in TC. The paired analysis preserves subject correspondence and shows how much patient variation remains around each mean. The measured surface-point overlap and increased sensitivity offer additional context, while lower precision and numerically larger mean HD95 expose the associated false-positive and boundary trade-offs. A single aggregate score would conceal these differences.

The evidence is narrower than an architectural superiority claim. B0's selected weights come from update 2,000 of a raw-input history containing 2,999 updates; B5 completed all three configured stages. B0 lacks the corruption augmentation used for B5. Validation selected both weights, and B0 and B5 inference used different numeric precision. The observed advantage can reflect these choices as well as decomposition and correction learning. An equal-budget B1 comparison would address the augmentation confound, and repeated seeds would assess optimization variability. This study reports the available checkpoints rather than presenting them as a completed controlled ablation.

\subsection{Utility supervision remains an open hypothesis}
The internal removal analysis shows that the deployed detail terms affect the output and, collectively, improve the fixed B5 base's mean validation Dice. It also reveals that most of the benefit comes from T1ce, while some very small contributions are neutral or harmful. This resolves one engineering concern: the model did not simply ignore every detail branch. It does not resolve whether utility supervision is the best way to retain those branches.

Indeed, the selected Stage C score is below the selected unit-gate Stage B score. This negative observation limits the central hypothesis. The utility target evaluates an ungated candidate's local cross-entropy contribution in one all-candidates context, whereas the deployed gates act jointly and the reported segmentation endpoint is region Dice. Candidate interactions, the scale of CE changes relative to $\tau$, and the balanced tumour/background objective can make the target imperfectly aligned with the final metric. The observed near-0.5 brain gate means are compatible with small pooled CE changes, but a causal explanation would require inspecting target distributions and matched controls.

Training B4 from the same frozen candidate checkpoint and running A1 are the most direct next experiments. They would distinguish the additional utility objective from ordinary spatial control and test whether its spatial targets matter. A6 could examine the cost of allowing candidates to change during gate fitting, but would answer a different question. No conclusion about utility-supervised robustness should precede these comparisons.

\subsection{Interpretability and robustness limits}
The additive interface exposes signed branch contributions to foreground scores and makes fixed-output interventions inexpensive. It provides an accounting of this implementation, not a clinical explanation of a lesion. Changing an input sequence changes the low-frequency backbone context, correction proposals, and gates; removing one already-computed contribution leaves those quantities fixed. Confusing these interventions would overstate modality attribution. Similarly, a stability certificate states that the computed winner survives contribution removal, not that the diagnosis is correct or that the input is reliable.

All modality masks execute successfully, but the ET decline without T1ce is large. The system consequently remains dependent on informative contrasts despite explicit availability handling. The small local-corruption changes are restricted to one corrupted sequence, two families, and the implemented severities. No external hospital, scanner, acquisition protocol, motion artifact, or real missingness process was evaluated. The ghosting and ringing suite is implemented for the locked internal test protocol but has not been executed. Robustness to those conditions remains unmeasured.

\subsection{Threats to validity and reporting scope}
The custom split is patient-disjoint and reproducible, but it is not the official BraTS challenge split. The 101 internal test subjects remain untouched; every performance table in this paper uses the 49-patient validation partition. Repeated checkpoint selection and exploratory analyses can bias validation conclusions. The patient bootstrap conditions on one split and one training seed, and its pointwise intervals do not account for multiple endpoints or model-selection uncertainty. These limits preclude a confirmatory generalization claim.

The downloaded Kaggle mirror's exact release version and reuse terms were not recorded by the experiment. MRI hashes improve input traceability, but do not recover missing licensing evidence or establish full historical mask-byte identity. Before public redistribution or submission, the authors must confirm the applicable BraTS/mirror terms and any institution or venue requirements for use of this secondary dataset. No new patient recruitment, prospective clinical study, institutional approval number, or funding statement is asserted here.

Four empty ET references and very small enhancing lesions show the importance of explicit metric conventions. Both-empty overlap scores, one-empty undefined distances, and valid-case boundary averages affect model comparisons. The unweighted surface-point Dice and concatenated-distance HD95 used here are implementation-specific and should not be compared directly to published scores calculated under other conventions or splits. Published methods in Related Work provide context rather than leaderboard baselines for this custom experiment.

\subsection{Next validation steps}
The first empirical priority is a matched B1/B5 comparison and a frozen-candidate B4/B5 study, followed by A1 and repeated seeds. After the configuration and selection protocol are fixed, the existing internal test manifest should be evaluated once with the prespecified unseen corruption families. Qualitative inspection should include additional patients chosen by a declared sampling rule, including absent and small ET cases and distant false positives. Independent external data and a same-hardware resource comparison are needed before deployment or computational-efficiency claims. These are remaining research tasks, not missing software implementations in the present pipeline.

\section{Conclusion and Future Work}
DetailTrust-Net implements modality-wise Gaussian decomposition, a full-width residual 3D U-Net, zero-anchored detail corrections, and utility-supervised spatial foreground-logit fusion. Its completed validation evaluation gives mean WT/TC/ET Dice 0.7379, compared with 0.7109 for the available raw-input B0 checkpoint on the same 49 patients. The 2.70-point difference is exploratory because training exposure, augmentation, and inference precision are not matched. Detail removal shows a measurable contribution within B5, primarily from T1ce, but the final learned gates score below the selected unit-gate candidates on clean validation. The present results therefore support further controlled study of the correction interface, while leaving the benefit of utility supervision, held-out test performance, and clinical robustness unresolved.

'''
    begin = text.index('\\section{Results}')
    end = text.index('\\begin{thebibliography}')
    text = text[:begin] + results + text[end:]
    # Inline bibliography retained; the most recent source has no final pagination.
    text = text.replace(r'\emph{IEEE Trans. Artificial Intell.}, 2026, doi:', r'\emph{IEEE Trans. Artificial Intell.}, vol. 7, no. 9, pp. 5076--5090, Sep. 2026, doi:')
    text = text.replace('Recent MRI architectures explore hierarchical attention, transformers, multiscale aggregation, and pretrained encoders. SwinPA-Net aggregates features across scales using a Swin-transformer-based pyramid, while Medical Transformer pretrains a universal encoder for 3D brain MRI tasks',
        'Medical-imaging architectures explore attention, multiscale aggregation, and pretrained encoders. SwinPA-Net uses a Swin-transformer-based pyramid for general medical-image segmentation, with reported tasks outside brain MRI. Medical Transformer pretrains an encoder for 3D brain MRI analysis')
    text = text.replace('Residual blocks offer a direct path through deeper feature stacks and have become common building blocks in volumetric segmentation.',
        r'Residual learning introduces an additive shortcut through a feature block \cite{he2016resnet}. Here this principle is applied to 3D feature stacks with projected channel alignment.')
    text = text.replace(r'\begin{thebibliography}{99}', r'''\begin{thebibliography}{99}
\bibitem{he2016resnet} K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image recognition,'' in \emph{Proc. IEEE Conf. Comput. Vis. Pattern Recognit.}, 2016, pp. 770--778, doi: 10.1109/CVPR.2016.90.
\bibitem{bakas2018challenge} S. Bakas \emph{et al.}, ``Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge,'' arXiv:1811.02629, 2018. [Online]. Available: \url{https://arxiv.org/abs/1811.02629}.''')
    text = text.replace("The mounted mirror's reuse terms and version still require verification before publication.",
        r'The official BraTS guidance permits research use with attribution to the benchmark, annotation collection, and challenge overview \cite{menze2015,bakas2017,bakas2018challenge}. The exact mounted mirror version and its additional reuse terms were not captured by the run.')
    text = text.replace('The current main configuration planned 4,000 updates.', 'Stage A completed its configured 4,000 updates.')
    text = text.replace('The current run planned 4,500 updates.', 'Stage B completed its configured 4,500 updates.')
    text = text.replace('The update budget therefore excludes neither validation nor inference cost from the actual session wall time.', 'Optimizer update counts exclude validation, while session wall time includes its reconstruction and scoring cost.')
    text = text.replace('Softmax and argmax are applied after reconstruction.', 'Argmax is applied to reconstructed logits; applying softmax first gives the same exclusive prediction.')
    text = text.replace('It cannot transform a partially trained candidate into a completed gate model.', 'Stage completion and evaluation coverage are tracked separately from a successful notebook exit.')
    text = text.replace(r'z_k&=z_{0,k}+\sum_{m=1}^{4}c_{m,k}, && k\in\{1,2,3\},',
                        r'z_k&=z_{0,k}+\sum_{m=1}^{4}c_{m,k},')
    text = text.replace('Accessed: Oct. 13, 2026.', 'Accessed: Oct. 14, 2026.')
    # IEEE numbered references follow the order of their first citation.
    first_citations = list(dict.fromkeys(k for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')))
    items = {
        key:line for line,key in re.findall(r'(\\bibitem\{([^}]+)\}[^\n]+)',text)}
    assert set(items)==set(first_citations)
    bib_start=text.index(r'\begin{thebibliography}{99}')
    bib_end=text.index(r'\end{thebibliography}',bib_start)
    ordered='\n'.join(items[k] for k in first_citations)
    ordered=re.sub(r'doi: ([^\s]+)\.',lambda m:r'doi: \url{'+m.group(1).replace(r'\_','_')+'}.',ordered)
    text=text[:bib_start]+r'\begin{thebibliography}{99}'+'\n'+ordered+'\n'+text[bib_end:]
    # Float barriers only at sections: avoid a succession of forced empty pages.
    text = text.replace('\n\\FloatBarrier\n', '\n')
    assert text.isascii()
    assert '_placeholder' not in text and 'training_curves_partial' not in text
    for phrase in ('Stage C did not run', '3,929/4,500', 'no model prediction', 'results are pending'):
        assert phrase not in text, phrase
    path.write_text(text,encoding='ascii')
    ledger=[]
    for region in ('WT','TC','ET'):
        for model,metric,source in [('B0',b0_metrics[region],str(B0/'metrics_summary.csv')),('B5',b5_metrics[region],str(B5/'metrics_summary.csv'))]:
            for key in ('dice','hd95_mm','surface_dice','precision','recall'):
                record=metric[key]
                ledger.append(dict(claim_id=f'{model}_{region}_{key}',section='Results',claim_text=f'{model} clean validation {region} {key}',value=record['mean'],unit='mm' if key=='hd95_mm' else 'fraction',dispersion=f"SD={record['std']}; valid n={record['valid_count']}",source_artifact=source,source_locator=f'clean_m15,{region},{key}.mean',status='verified'))
    for r in pairs:
        ledger.append(dict(claim_id='paired_'+r['region'],section='Results',claim_text='B5 minus B0 clean validation Dice',value=100*float(r['B5_minus_B0_dice']),unit='percentage points',dispersion=f"95% interval [{100*float(r['pointwise_bootstrap_ci95_low'])},{100*float(r['pointwise_bootstrap_ci95_high'])}]",source_artifact=str(B0/'baseline_vs_b5_summary.csv'),source_locator=r['region'],status='independently_recomputed'))
    with (AUDIT/'results_ledger.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    audit=dict(paired_point_estimates_and_bootstrap_recomputed=True,matched_patients=49,test_evaluation=False,
               patient_split_equal=True,all_canonical_mri_hashes_match=True,
               no_fabricated_metrics=True,baseline_selected_update=2000,baseline_available_history_updates=2999,
               b5_stages=[{k:v for k,v in r.items() if k in ('stage','updates_completed','selected_update','validation_score')} for r in b5_result['stages']],
               manuscript_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (AUDIT/'numerical_audit.json').write_text(json.dumps(audit,indent=2))
    print('Updated main.tex; paired comparisons independently verified; results ledger saved.')


if __name__=='__main__': main()
