"""Shorten the measured IEEE paper while preserving its scientific disclosures."""
from pathlib import Path
import csv
import json
import re
import hashlib

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'DetailTrustNet_IEEE_Paper'
OUT=ROOT/'verification/paper_condensed'
B5=ROOT/'output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z/detailtrust'
B0=ROOT/'local_baseline_results'


def fig(name,caption,label,wide=True):
    env='figure*' if wide else 'figure'
    width=r'\textwidth' if wide else r'\columnwidth'
    return '\n'+rf'''\begin{{{env}}}[!t]
\centering
\includegraphics[width={width}]{{figures/{name}.pdf}}
\caption{{{caption}}}
\label{{{label}}}
\end{{{env}}}
'''+'\n'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    before=OUT/'main_before_condense.tex'
    if not before.exists():before.write_bytes((PAPER/'main.tex').read_bytes())
    original=before.read_text(encoding='utf-8-sig')
    preamble=original[:original.index(r'\begin{abstract}')]
    # Section barriers flush double-column floats prematurely; natural column
    # bottoms prevent stretchable heading glue from filling a short column.
    preamble=preamble.replace(r'\usepackage[section]{placeins}'+'\n','')
    preamble=preamble.replace(r'\maketitle',r'\maketitle'+'\n'+r'\raggedbottom')
    bib_start=original.index(r'\begin{thebibliography}')
    bib=original[bib_start:]
    abstract=r'''DetailTrust-Net studies selective use of image detail in multimodal 3D brain-tumour MRI segmentation. A learned Gaussian mixture separates smooth images from detail residuals. A residual U-Net predicts base logits, and shared modality-conditioned branches propose zero-anchored foreground corrections. Spatial gates are fitted with frozen candidates using segmentation loss and a detached correction-removal utility target. On 49 patients from a custom BraTS 2020 validation split, the selected B5 model achieves WT, TC, and ET Dice of 0.8526, 0.6876, and 0.6736, compared with 0.8328, 0.6324, and 0.6675 for the available raw-input B0 baseline. Mean region Dice differs by 2.70 percentage points, with a paired patient bootstrap interval of 0.33--4.99 points. ET uncertainty includes zero, and mean HD95 does not improve. Training exposure and corruption augmentation differ between models. Removing all B5 detail contributions reduces mean Dice by 0.84 points; however, the final learned gates score 0.24 points below the selected unit-gate candidate stage. All 15 nonempty modality subsets and eight local FLAIR-corruption conditions are evaluated. These validation results quantify correction effects and failure modes, while the benefit of utility supervision and independent test performance remain unresolved.'''
    intro=r'''\section{Introduction}
Glioma segmentation from MRI assigns voxel labels to tumour subregions with different appearances and clinical roles. BraTS supplies registered, skull-stripped multimodal images and expert-reviewed annotations for studying this task \cite{menze2015,bakas2017}. Diffuse boundaries, small enhancing lesions, and heterogeneous tumour cores make segmentation difficult even when all sequences are available.

The four inputs considered here are T1, post-contrast T1 (T1ce), T2, and FLAIR. Their information is complementary, and a missing or degraded sequence changes the evidence available to a model. Previous methods address incomplete inputs through feature disentanglement, image synthesis, modality-masked fusion, and expert routing \cite{yang2022d2,yang2023hyper,shi2024m2f,zhang2025mome}. Frequency decomposition is also established, including in D$^2$-Net and HFF-Net \cite{yang2022d2,shao2025hff}.

Our question is whether image detail can be retained according to its measured segmentation utility. DetailTrust-Net first predicts from smooth inputs, then proposes modality-specific foreground-logit corrections from residual detail. Each proposal is anchored to zero at zero detail, and a spatial gate scales its contribution. During gate training, reference labels determine whether removing a candidate increases local cross-entropy; inference uses only images and availability flags.

The work contributes a full-width 3D correction architecture, a frozen-candidate utility-supervision objective, and an additive interface for branch-removal and conditional-stability analysis. We report a completed B5 validation study and a comparison with an available raw-input baseline. Controlled gating comparisons are still needed to attribute any improvement specifically to utility supervision.

\section{Related Work}
\subsection{Volumetric segmentation and multimodal fusion}
3D U-Net extends an encoder--decoder with skip connections to volumetric segmentation \cite{cicek2016}. Residual learning adds shortcuts through feature blocks \cite{he2016resnet}, while nnU-Net emphasizes dataset-dependent configuration, preprocessing, and validation \cite{isensee2021}. DetailTrust-Net uses these established principles in a residual 3D backbone.

Attention and multiscale aggregation offer other representation choices. SwinPA-Net studies a transformer-based pyramid for general medical-image segmentation outside brain MRI; Medical Transformer trains an encoder for 3D brain MRI analysis \cite{du2024swinpa,jun2024medtrans}. MBUNeXt uses modality branches and layer-fusion aggregation for multimodal tumour segmentation \cite{liu2025mbunext}. These methods provide architectural context, rather than reproduced leaderboard comparators for our custom split.

\subsection{Incomplete modalities and image detail}
HeMIS aggregates features over available modalities \cite{havaei2016hemis}. D$^2$-Net disentangles modality and region information; hypernetwork-based synthesis supplies missing evidence; M$^2$FTrans masks modalities in transformer fusion; and MoME+ routes modality experts \cite{yang2022d2,yang2023hyper,shi2024m2f,zhang2025mome}. Mamba-based approaches also study incomplete multimodal learning and feature distillation \cite{zhang2026m2seg,kang2026mamba}.

D$^2$-Net uses spatial-frequency information, and HFF-Net combines frequency decomposition with adaptive filtering and cross-attention \cite{yang2022d2,shao2025hff}. DetailTrust-Net's proposed distinction is the explicit combination of anchored detail-only corrections, additive foreground-logit fusion, and gate targets derived from correction-removal utility. The empirical question is whether that target adds value beyond spatial gates trained through segmentation alone. B4/B5 and shuffled-target controls are defined for this purpose, but have no completed comparison here.

\section{Problem Formulation}
Let $X\in\mathbb R^{4\times D\times H\times W}$ contain registered T1, T1ce, T2, and FLAIR volumes. An availability vector $a\in\{0,1\}^4$, with $\sum_m a_m>0$, gives observed inputs $\tilde X_m=a_mX_m$. The target $Y\in\{0,1,2,3\}^{D\times H\times W}$ contains background, NCR/NET, oedema, and enhancing tumour. Raw BraTS labels $0,1,2,4$ map to $0,1,2,3$. Derived masks use $\mathrm{WT}=\{1,2,3\}$, $\mathrm{TC}=\{1,3\}$, and $\mathrm{ET}=\{3\}$.

The model returns four exclusive class logits at each voxel. Availability changes require rerunning the complete network; suppressing an already-computed correction is a separate, conditional intervention.

'''
    method=r'''\section{Proposed Method: DetailTrust-Net}
\subsection{Physical-scale smooth/detail decomposition}
Fig.~\ref{fig:architecture} gives the implemented layers and shapes. For each modality, separable Gaussian filters use $\sigma_j\in\{0.5,0.8,1.2\}$ mm. Let $G_{\sigma_j}$ denote the normalized filter and $\alpha_{mj}$ three learned scalar logits. The decomposition is
\begin{align}
\pi_{mj}&=\frac{\exp(\alpha_{mj})}{\sum_{l=1}^{3}\exp(\alpha_{ml})},\nonumber\\
L_m&=\sum_{j=1}^{3}\pi_{mj}G_{\sigma_j}(\tilde X_m),\nonumber\\
H_m&=\tilde X_m-L_m.
\label{eq:split}
\end{align}
The mixture has 12 global learned parameters, initialized with $\alpha_m=(-1,1,-1)$. It does not vary spatially or by patient. By construction, $\tilde X_m=L_m+H_m$ up to floating-point error.

Physical sigma is converted to voxel units by dividing by spacing. The integrated model uses the verified 1-mm grid, nine-tap separable kernels, and reflection padding. Different spacing would require passing that spacing or resampling before model input. Missing modalities are zeroed before filtering. Four smooth images and four replicated availability maps form the eight-channel backbone input.
''' + fig('architecture',r'Detailed implemented architecture in a 2$\times$2 layout. (a) Availability masking, three physical-scale Gaussian filters per modality, learned mixture, and halo-to-target cropping; $G_j=G_{\sigma_j}$. (b) All five encoder levels, four decoder levels, dashed skip connections, 32-channel context features, and four base logits. (c) Two evaluations of the shared 37-channel refiner implement zero anchoring; the separate 44-channel spatial gate scales each modality correction. (d) Detached utility supervision during Stage C and additive foreground-logit fusion at inference; the cross-panel dashed arrow denotes training supervision. Background logits remain unchanged.','fig:architecture') + r'''
\subsection{Backbone and anchored corrections}
The residual 3D U-Net has encoder widths $(32,64,128,256,320)$. Each block uses two bias-free $3\times3\times3$ convolutions, eight-group GroupNorm, leaky ReLU with slope 0.01, and a projected $1\times1\times1$ shortcut. Average pooling downsamples features; the decoder uses trilinear interpolation to the matching skip size and concatenation. The final 32-channel feature $F$ and a four-channel logit head occupy the full central patch grid.

One correction network $R_\theta$ is reused for all modalities. It concatenates $F$, the scalar detail volume $H_m$, and a four-channel one-hot identity $e_m$, giving 37 input channels. Two $3\times3\times3$ convolutions with 16 channels and ReLU are followed by a three-channel $1\times1\times1$ output for NCR/NET, oedema, and ET. The anchored proposal is
\begin{equation}
r_m=R_\theta(F,H_m,e_m)-R_\theta(F,0,e_m).
\label{eq:anchor}
\end{equation}
The refiner has no normalization or dropout, so both evaluations use the same deterministic mapping. For fixed $F,e_m$, zero detail gives zero correction. The final layer starts with weight standard deviation 0.001 and zero bias. Anchoring constrains the zero-detail response, not the sign or usefulness of a nonzero proposal.

\subsection{Spatial gates and logit fusion}
The shared gate receives $F,H_m,r_m,e_m$ and all four availability maps: $32+1+3+4+4=44$ channels. Its layers are a $44\to16$ pointwise convolution with ReLU, a depthwise $3\times3\times3$ convolution with ReLU, and a $16\to1$ pointwise convolution with sigmoid. Independent coefficients $q_m(v)\in(0,1)$ permit several modalities to contribute at one voxel. They are not normalized across modalities.

For foreground class $k\in\{1,2,3\}$, define $c_{m,k}=a_mq_mr_{m,k}$. Fusion and prediction are
\begin{equation}
\begin{aligned}
z_k&=z_{0,k}+\sum_{m=1}^{4}c_{m,k},\\
z_{\mathrm{bg}}&=z_{0,\mathrm{bg}},
\end{aligned}
\label{eq:fusion}
\end{equation}
\begin{equation}
\hat Y(v)=\underset{k\in\{0,1,2,3\}}{\operatorname{arg\,max}}z_k(v).
\label{eq:prediction}
\end{equation}
Thus each branch has an explicit signed effect on a foreground score. Gate values are coefficients, not calibrated probabilities of usefulness; measured removal effects determine whether a retained term helps the segmentation.

\subsection{Segmentation loss and staged optimization}
For $p=\operatorname{softmax}(z)$, region probabilities are $p_{\mathrm{WT}}=1-p_0$, $p_{\mathrm{TC}}=p_1+p_3$, and $p_{\mathrm{ET}}=p_3$. With binary reference masks $y_R$ and $\epsilon=10^{-5}$,
\begin{equation}
D_R=\frac{2\sum_v p_R(v)y_R(v)+\epsilon}
{\sum_v p_R(v)+\sum_v y_R(v)+\epsilon}.
\label{eq:softdice}
\end{equation}
Writing $\mathcal R=\{\mathrm{WT},\mathrm{TC},\mathrm{ET}\}$,
\begin{equation}
\mathcal L_{\mathrm{seg}}=\mathcal L_{\mathrm{CE}}(z,Y)
+\frac{1}{3}\sum_{R\in\mathcal R}(1-D_R).
\label{eq:seg}
\end{equation}
CE is averaged over voxels; region Dice is averaged per patient and then over the batch. Loss calculations use float32.

Fig.~\ref{fig:pipeline} summarizes three stages. Stage A trains the U-Net and Gaussian mixture with $\mathcal L_{\mathrm{seg}}(z_0,Y)$, with corrections inactive. Stage B trains the backbone, mixture, and refiner with gates fixed to one. Independent branch retention $\rho_m\sim\operatorname{Bernoulli}(0.75)$ controls final logits $z_\rho=z_0+\sum_m\rho_m\tilde c_m$, where $\tilde c_m=(0,a_mr_m)$. Its loss is
\begin{equation}
\begin{aligned}
\mathcal L_B&=\mathcal L_{\mathrm{seg}}(z_\rho,Y)
+0.3\mathcal L_{\mathrm{seg}}(z_0,Y)\\
&\quad+0.1\mathcal L_{\mathrm{branch}}.
\end{aligned}
\label{eq:stageb}
\end{equation}
Here $\mathcal L_{\mathrm{branch}}$ averages $\mathcal L_{\mathrm{seg}}(z_0+\tilde c_m,Y)$ over available subject--modality pairs. Branch dropout suppresses a proposal without changing backbone input; input-modality dropout changes both images and flags.

Stage C freezes the candidates and trains only the spatial gate. Define ungated all-candidate logits and a modality's removal effect:
\begin{align}
z_{\mathrm{all}}&=z_0+\sum_m\tilde c_m,\label{eq:candidates}\\
\delta_m(v)&=\ell(z_{\mathrm{all}}(v)-\tilde c_m(v),Y(v))\nonumber\\
&\quad-\ell(z_{\mathrm{all}}(v),Y(v)),\label{eq:delta}\\
t_m&=\operatorname{stopgrad}\!\left[
\sigma\!\left(\frac{\operatorname{AvgPool}_{3}(\delta_m)}{\tau}\right)\right].
\label{eq:utility}
\end{align}
Here $\ell$ is voxelwise CE, $\tau=0.1$, and pooling uses a $3^3$ neighbourhood, stride one, padding one, and excludes padded positions. Positive $\delta_m$ means that removing the candidate worsens CE. Targets are computed from ungated proposals and detached; the current gate cannot change its own target by suppressing a candidate.

Let $\mathcal B_m$ be BCE between $q_m$ and $t_m$, averaged equally over tumour and background voxel sets when both exist, or over the nonempty set otherwise. For the actual batch size of one,
\begin{equation}
\mathcal L_C=\mathcal L_{\mathrm{seg}}(z,Y)
+\frac{0.1}{\sum_m a_m}\sum_m a_m\mathcal B_m.
\label{eq:stagec}
\end{equation}
The target describes one candidate context and CE objective. It is neither a Shapley attribution nor a calibrated quality probability. Reference labels are used to construct training targets and evaluate outputs, never as inference features.
''' + fig('training_pipeline',r'Three-stage optimization workflow, separate from the detailed network architecture in Fig.~\ref{fig:architecture}. Fit the smooth-input base, learn anchored candidates with unit gates and branch dropout, then freeze candidates and fit spatial gates with segmentation and detached utility supervision. Each stage uses validation selection and resumable checkpoints; the internal test remains sealed.','fig:pipeline',False) + r'''
\subsection{Conditional stability}
For winner $w=\arg\max_k z_k$ and rival $j\ne w$, set $M_j=z_w-z_j$ and $d_{m,j}=c_{m,w}-c_{m,j}$. The worst rival margin after removing any subset of the fixed contributions is
\begin{equation}
B=\min_{j\ne w}\left[M_j-\sum_m\max(d_{m,j},0)\right].
\label{eq:stability}
\end{equation}
Removing a positive $d_{m,j}$ reduces this margin; removing a negative term increases it. Hence $B>\eta$ certifies the same winner for all 16 retention subsets at tolerance $\eta$. The code checks this calculation against enumeration. Features, proposals, and gates remain fixed: the bound does not certify correctness or robustness after changing MRI inputs.

'''
    setup=r'''\section{Experimental Setup}
\subsection{Dataset, split, and preprocessing}
The mounted Kaggle input is \path{awsaf49/brats20-dataset-training-validation}. Of 494 discovered subjects, 368 have all four modalities and a distributed label. The other 126 lack labels, including the challenge validation images and one training case. BraTS 2020 documents the NIfTI modalities, annotation labels, 1-mm preprocessing, and withheld challenge validation labels \cite{brats2020data}. Its research-use guidance requests the benchmark, annotation, and challenge-overview citations \cite{menze2015,bakas2017,bakas2018challenge}; the exact mirror version and any additional mirror terms were not recorded.

A custom split with seed 20261012 contains 218 training, 49 validation, and 101 internal test patients. Patient IDs are disjoint. This is not the official challenge split: internal test subjects come from the publicly labelled collection, and official comparison ID lists were not recovered. The internal test remains untouched. Checkpoints are selected by patient-macro mean WT/TC/ET validation Dice.

The adapter canonicalizes NIfTI orientation to RAS, checks alignment and geometry, and remaps labels. The inspected grid is $240\times240\times155$ at 1-mm spacing; no further resampling is used. Each modality is z-scored over its nonzero foreground. MRI intensities determine cropping, without segmentation labels. Training images have a four-voxel halo around a $96^3$ target patch. Half of centres are sampled from present tumour classes and half from available-image foreground. Aligned random flips preserve discrete labels without interpolation. Complete availability is sampled with probability 0.25; otherwise one of the 14 incomplete nonempty masks is sampled uniformly.

Inference uses available-image foreground cropping and overlapping $96^3$ windows with 50\% stride and reflected halos. Base logits and each gated contribution are accumulated separately on CPU with uniform overlap counts, reconstructed into the original volume, and summed before argmax. There is no probability blending or nonlinear postprocessing. Fig.~\ref{fig:qual} combines a real MRI example with aligned predictions; the chosen reference slice is used only for display.

\subsection{Training and baseline provenance}
B5 uses batch size one, seed 1729, AdamW, candidate learning rate $2\times10^{-4}$, gate learning rate $10^{-3}$, and weight decay $10^{-5}$. Each stage has 100 warmup updates followed by cosine decay. Stages A/B/C complete 4,000/4,500/1,500 updates; their selected checkpoints are at 2,000/4,500/1,500. Validation runs every 500 updates and at each stage end. Compatible continuations restore optimizer, scheduler, scaler, update, and deterministic patch-stream state; the selected candidate weights initialize the following stage.

B0 is the raw-input residual U-Net with availability flags and modality dropout. Evaluation uses its update-2,000 selected checkpoint from an available 2,999-update training history against a configured 10,000-update budget; no further B0 training was performed. B0 lacks the corruption augmentation used by B5. Both models have the same patient split, backbone widths, label mapping, patch geometry, and reconstruction. All 49 canonical MRI hashes match the saved B5 integrity record; reference geometry, mapping, and ET volumes match, but original complete label-array hashes were not stored.

Kaggle used GPU FP16 and activation checkpointing on one Tesla T4, despite two visible devices. Recorded versions are Python 3.13.15, PyTorch 2.11.0+cu128, and CUDA 12.8. B0 evaluation used CPU float32 and four threads. Thus training exposure, augmentation, and inference precision are not matched; the comparison is exploratory checkpoint evidence.

\subsection{Corruptions and implemented controls}
B5 training applies a corruption with probability 0.5 to one available modality. It samples resolution loss, noise, or bias, a random axis, and local or global application. Resolution factors span 1.5--3; noise and bias parameters span 0.05--0.15. The bias field interpolates a random $4^3$ lattice and clips its multiplicative factor to $[0.8,1.2]$.

Evaluation reruns B5 under all 15 nonempty modality masks. Eight further conditions locally corrupt FLAIR with noise standard deviation 0.075 or 0.15, or axis-2 resolution factor 2.5 or 3.5, with all inputs or T1ce absent. Resolution loss uses Gaussian blur, adaptive-average coarsening, and trilinear reconstruction. Local blending uses a Gaussian mask with centre in 20--80\% of each dimension and scale 22\% of that dimension; original zero background remains zero. The unexecuted internal-test suite uses ghosting and ringing rather than these validation families.

The remaining registry defines B1 as B0 with corruption training; B2 as anchored corrections with unit gates; B3 as global sigmoid gates; and B4 as spatial gates with segmentation-only supervision. A1 shuffles utility targets, A2 fixes the 0.8-mm Gaussian, A3 uses raw-image context/refinement, A4 uses Haar low-pass reconstruction, A5 removes anchoring, A6 unfreezes candidates during gate training, and A7 adds a weight-0.1 resolution-matched occupancy loss for global coarsening. These variants are implemented but not completed comparisons. B4/B5 should share the selected Stage B candidates to isolate gate supervision.

\subsection{Metrics, branch analysis, and uncertainty}
Patient-macro summaries include Dice, IoU, precision, sensitivity, specificity, HD95 in mm, surface-point Dice at 1-mm tolerance, and volume errors. For true/false positive and negative counts,
\begin{equation}
\operatorname{Dice}=\frac{2\mathrm{TP}}{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}.
\end{equation}
Both-empty overlap is one. Precision is undefined if prediction is empty; sensitivity is undefined if reference is empty. Surface metrics use six-connected boundary voxel centres and physical distances. HD95 is the 95th percentile of concatenated directed distances; surface-point Dice is unweighted by surface area. Both-empty surfaces give distance zero and overlap one; one-empty surfaces give undefined distance and overlap zero. Aggregates retain valid counts. ET lesion diagnostics use 26-connected components and maximum-cardinality one-to-one matching by any overlap.

Correction removal suppresses already-computed gated contributions while retaining the input and B5 base logits. Candidate utility uses the ungated context in \eqref{eq:delta}; deployed utility removes the gated term. Clean-input records include signed contributions, correction magnitude, gate distributions, and within-patient Spearman associations by spatial scope.

B0/B5 rows are joined by patient, condition, and region before averaging. The primary mean first averages WT, TC, and ET within each patient. Pointwise 95\% percentile intervals resample the 49 paired patient vectors 20,000 times with seed 20261014. Internal removal comparisons use the same procedure. Intervals condition on selected weights and validation reuse; they do not include seed variability, model-selection correction, or multiplicity adjustment. No p-values are reported.

'''
    b0=json.loads((B0/'local_baseline_report.json').read_text())
    b0lookup={r['region']:r for r in b0['validation_metrics']}
    with (B5/'metrics_summary.csv').open(newline='') as h:
        b5lookup={r['region']:{key:json.loads(r[key]) for key in ('dice','hd95_mm','surface_dice','precision','recall')} for r in csv.DictReader(h) if r['condition']=='clean_m15'}
    lines=[]
    for model,lookup in [('B0',b0lookup),('B5',b5lookup)]:
        for region in ('WT','TC','ET'):
            r=lookup[region]
            lines.append(f"{model} & {region} & {r['dice']['mean']:.4f} & {r['hd95_mm']['mean']:.2f} ({r['hd95_mm']['valid_count']}) & {r['surface_dice']['mean']:.4f} & {r['precision']['mean']:.4f} ({r['precision']['valid_count']}) & {r['recall']['mean']:.4f} ({r['recall']['valid_count']}) \\\\")
    metrics_table=r'''% Source: local_baseline_results/metrics_summary.csv and B5 metrics_summary.csv, clean_m15.
\begin{table*}[t]
\caption{Clean validation on the same 49 patients. Values are patient-macro means. Parentheses give valid counts; HD95 is in mm, and SD$_1$ is surface-point Dice at 1-mm tolerance. Training provenance is specified in Experimental Setup.}
\label{tab:results}
\centering
\begin{tabularx}{\textwidth}{@{}l l L L L L L@{}}
\toprule
Model & Region & Dice & HD95 ($n$) & SD$_1$ & Precision ($n$) & Sensitivity ($n$) \\
\midrule
'''+'\n'.join(lines)+r'''
\bottomrule
\end{tabularx}
\end{table*}
'''
    ci_table=re.search(r'\\begin\{table\}\[t\]\s*\\caption\{B5 minus B0.*?\\end\{table\}',original,re.S).group(0)
    results=r'''\section{Results}
\subsection{Clean-input comparison}
Table~\ref{tab:results} gives the complete clean-input B0/B5 validation comparison. B5's WT, TC, and ET Dice are 0.8526, 0.6876, and 0.6736, versus 0.8328, 0.6324, and 0.6675 for B0. Mean region Dice is 0.7379 versus 0.7109. Table~\ref{tab:paired} reports a paired mean difference of 2.70 percentage points (95\% interval 0.33--4.99). TC has the largest difference; ET's interval spans $-4.64$ to $+4.26$ points. Exclusive-class Dice also rises from 0.2450 to 0.3770 for NCR/NET and from 0.6794 to 0.7166 for oedema.

''' + metrics_table + '\n' + ci_table + r'''

B5 has higher surface-point Dice and sensitivity, but lower precision and numerically higher mean HD95. ET distances include 45 B5 and 46 B0 cases, so valid-case aggregates differ in included subjects. Four references have no ET; B5 predicts ET in all four, whereas B0 has one both-empty case. These cases remain in the declared overlap averages. The results show an overlap/sensitivity gain with false-positive and boundary trade-offs, not a uniform improvement across metrics.

\subsection{Training and conditional correction effects}
Fig.~\ref{fig:comparison} combines the patient comparison and completed stage trajectories. Selected A/B/C validation scores are 0.7105, 0.7403, and 0.7379. Stage C's learned gates score 0.24 percentage points below the selected unit-gate candidates, an observed negative clean-input result. The raw-baseline difference therefore does not establish that utility supervision helped the gates.
''' + fig('comparison_training',r'Validation comparison and training. (a) Clean patient-macro Dice. (b) All 49 paired patient differences; diamonds are means and bars are pointwise 95\% paired bootstrap intervals. (c--e) Completed A/B/C selection trajectories with separate update counters and a shared Dice scale. Stage B uses unit gates and C uses learned gates with frozen candidates. The differing training exposures are stated in Experimental Setup.','fig:comparison') + r'''
Removing every gated detail term from B5 gives WT/TC/ET Dice 0.8522/0.6827/0.6535 and mean 0.7295: a 0.84-point reduction from the full output. T1ce removal accounts for most of this effect, reducing mean Dice by 0.78 points. Removing T2 gives a tiny 0.0014-point mean increase, with an interval spanning zero. Fig.~\ref{fig:internal} distinguishes these fixed-output interventions from independently trained models.
''' + fig('internal_corrections',r'Conditional detail removal on 49 clean validation patients. Positive values favour the full B5 output; bars are pointwise paired bootstrap intervals. The All row suppresses every correction while retaining B5 base logits. Other rows remove individual computed contributions without masking MRI inputs or rerunning the backbone.','fig:internal') + r'''
\subsection{Missing and degraded modalities}
B5 completes all 15 nonempty input subsets and eight local corruption conditions, each with 49 patients. Fig.~\ref{fig:robustness} combines both analyses. Without T1ce, using T1+T2+FLAIR, ET Dice falls to 0.2915 from 0.6736 with complete inputs. FLAIR alone gives WT Dice 0.7920; T1ce alone gives ET Dice 0.4258. The T1+T1ce+FLAIR subset reaches mean 0.7346. Uniformly averaging all subsets gives 0.5538, which does not weight clinical missingness frequencies.
''' + fig('robustness_combined',r'B5 robustness validation, 49 patients per condition. (a) All 15 input subsets; C=T1ce, F=FLAIR. (b--d) Corrupted-minus-matching-clean patient differences and mean diamonds. N1/N2 are local FLAIR noise 0.075/0.15; R1/R2 are resolution factors 2.5/3.5 on axis 2. /4 retains all inputs and /3 removes T1ce. Corruption panels retain all observations and use their own vertical ranges. B0 has no measured robustness suite.','fig:robustness') + r'''
Mean changes under the tested local corruptions are small; the largest three-region reduction is approximately 0.094 percentage points for stronger resolution loss with four inputs. Other available sequences may supply redundant evidence. These measurements cannot separate smoothing, corruption training, and gates, and clean-only gate diagnostics do not reveal gate responses to corruption.

\subsection{Qualitative outputs and gate behaviour}
Fig.~\ref{fig:qual} combines four real MRI inputs, reference segmentation, the independent B0 prediction, and B5 base/final outputs for subject 205, the first validation ID. Axial RAS slice 90 is the median reference tumour coordinate, chosen without consulting model scores. Baseline full-volume Dice was reproduced locally, and the displayed reference/FLAIR planes match the B5 export. A distant false positive remains visible; the error-change map shows both repaired and introduced errors.
''' + fig('mri_predictions',r'Real axial validation example, BraTS20\_Training\_205 at RAS slice 90. Top: aligned inputs and reference. Bottom: independent B0 baseline, B5 smooth-input base, final B5 output, and error change, with FLAIR for orientation. Blue is wrong only in B0, orange only in B5, grey in both. B5 base is not B0. Reference labels select the display slice only; grayscale shares the normalized range $[-1,3]$.','fig:qual') + r'''
The clean-input gate and contribution records each contain 980 patient--modality--scope rows. Brain-average gate coefficients are near 0.5; T1ce tumour-region means are approximately 0.58, 0.63, and 0.64 for WT, TC, and ET. Gate--removal-utility Spearman associations are weak and scope-dependent. Fig.~\ref{fig:gates} uses common scales to expose actual contribution magnitude rather than independently stretching each modality.
''' + fig('gate_contribution_maps',r'Actual axial B5 gates and signed ET-logit contributions for subject 205. Gates share $[0,1]$; contributions share a symmetric range set by the largest absolute value across modalities, without percentile clipping. Maps are window-averaged. Zero image background and uncomputed crop fill are hidden for display. Gate values are coefficients, not calibrated confidence.','fig:gates') + r'''
\subsection{Numerical checks and model cost}
All 49 sampled-brain stability checks agree with enumeration of 16 retention subsets; the largest margin discrepancy is $1.91\times10^{-6}$. Code-level checks also pass for geometry, labels, split separation, Gaussian reconstruction, anchoring, availability, gate/fusion shapes, gradients, utility detachment, checkpoint/optimizer reload, and sliding-window reconstruction. A repeated-patch sanity check reduces loss. These are implementation checks, separate from cohort accuracy.

B5 has 16,934,644 parameters: 16,910,468 backbone, 12 mixture, 22,979 refiner, and 1,185 gate parameters. Relative to B0, the extra 24,176 parameters are 0.143\%, but filtering, anchoring, and staged training add operations. The final profiler's median A/B/C step times are 3.99/1.74/0.62 s, using 100 measured updates after five warmups; its largest allocated peak is 3.38 GiB. A separate subject-205 GPU benchmark measures median 11.98 s over two reconstructions after warmup, 27 windows, and 3.22 GiB allocated peak. B0's four-thread CPU reconstruction median is 73.63 s over 49 patients. Hardware, precision, and sampling differ, so these times do not define a model speed ratio; no validated FLOP comparison is available.

Split manifests, histories, selected weights/configurations, hashes, patient CSVs, and gate/contribution/stability records are retained outside the manuscript package. Baseline and portable B5 checkpoint SHA-256 prefixes are \texttt{2caba443318f} and \texttt{c3e3cbeeef00}. Every reported figure is generated from these measured artifacts or the verified architecture.

'''
    discussion=r'''\section{Discussion}
The selected B5 checkpoint has higher validation overlap than the available B0 baseline, most clearly in tumour core. Greater sensitivity accompanies lower precision and larger mean HD95. The conditional removal analysis confirms that detail terms affect B5's output, primarily through T1ce, while some small branches are neutral or harmful. A nonzero correction or large gate therefore does not establish usefulness.

The lower selected score after Stage C limits the utility-supervision hypothesis. Its target measures an ungated candidate's local CE effect in one context; deployed gates act jointly, and checkpoint selection uses region Dice. Candidate interactions and the scale of pooled CE changes relative to $\tau$ may contribute to this mismatch. The near-0.5 brain gate means motivate target-distribution inspection, but do not establish a causal explanation. B4 initialized from the same Stage B candidates and A1 are the direct controls needed to test supervision and target location.

The additive interface provides transparent score accounting. It does not establish clinical causality: changing MRI inputs can also change the backbone, proposals, and gates. Likewise, conditional stability can certify an unchanged but incorrect label. The missing-modality results show continuing dependence on T1ce for ET, and local one-sequence simulations do not establish robustness to scanner shifts or real acquisition failures.

\subsection{Validity limits and next experiments}
This is a single-seed validation study. Checkpoint selection reuses the evaluated patients, while baseline training and augmentation differ as recorded in Experimental Setup. Patient-bootstrap intervals condition on fixed selected weights and do not address those confounds or training-seed variation. The custom split and implementation-specific HD95/surface-point conventions also prevent direct leaderboard comparison to publications using other cohorts or metrics.

The mirror's exact release and additional terms require confirmation before redistribution or submission. MRI hashes improve traceability but cannot recover historical mask-byte identity. Empty ET references and small lesions are retained under explicit scoring rules. The analysis uses released research scans and includes no prospective recruitment or clinical validation.

The next experiments are matched-budget/corruption baseline training, frozen-candidate B4/B5 and A1 comparisons, and repeated seeds. Once the protocol is fixed, the sealed internal test should be evaluated with its unseen corruption families. External data and a same-hardware resource comparison would be needed for generalization and efficiency claims. Additional qualitative cases should follow a declared sampling rule, including absent ET and remote false positives.

\section{Conclusion}
DetailTrust-Net combines smooth-input segmentation with anchored detail proposals and utility-supervised spatial fusion. Its selected model reaches mean WT/TC/ET validation Dice 0.7379 versus 0.7109 for the available B0 checkpoint, under unmatched training exposure. Detail contributions measurably improve the fixed B5 base, but final learned gates score below the selected unit-gate candidates. These results support controlled study of the correction interface; utility-supervision benefit, independent test performance, and clinical robustness remain open.

'''
    # Retain explanations that make the mathematics and protocol assessable.
    # These add distinct information; repeated disclaimers remain removed.
    intro=intro.replace('The work contributes a full-width',r'''The separation is motivated by a practical tension. Smooth inputs may preserve broad spatial context while reducing sensitivity to local detail changes, but the removed residual can contain useful lesion structure as well as nuisance variation. A branch that always accepts residual detail may therefore add both correct foreground and false positives. The gate is intended to control that trade-off locally. Whether it succeeds is an empirical question: aggregate overlap, boundary errors, and the outcome of removing its contributions can give different answers.

The primary validation endpoint averages the three nested tumour regions, while the analysis retains each region separately. This avoids interpreting an improvement in whole tumour as evidence that small enhancing components also improve. Our study consequently examines clean segmentation together with modality removal, prespecified degradation, and the behaviour of each detail branch.

The work contributes a full-width''')
    method=method.replace('Missing modalities are zeroed before filtering.',r'''Gaussian low-pass filtering is a spatial smoothing operation rather than a hard separation into disjoint frequency bands. The residual is defined by subtraction, so it contains everything not retained by the chosen smooth mixture. The learned weights change the relative smoothing scales but do not add a patient-specific filter predictor. Thus physical scale, normalization, and the input grid are explicit parts of the decomposition. Kernel normalization preserves constant inputs away from boundary effects, and differentiable convolutions allow the mixture logits to train with the segmentation loss.

Missing modalities are zeroed before filtering.''')
    method=method.replace('Anchoring constrains the zero-detail response, not the sign or usefulness of a nonzero proposal.',r'''Anchoring constrains the zero-detail response, not the sign or usefulness of a nonzero proposal.

Both refiner evaluations retain the common context and modality identity. Without the subtraction, the same network could produce a correction from context alone even when its own detail channel is zero. Anchoring removes that constant-in-detail response for the current context. It does not make the branch independent of the other modalities: their smooth images and availability flags influence $F$. The shared network can therefore interpret the same residual differently under different available-image contexts, while its one-hot identity distinguishes sequence roles.

For a central $96^3$ patch, encoder grids are $96^3,48^3,24^3,12^3,6^3$. Decoder blocks receive concatenations of $320+256$, $256+128$, $128+64$, and $64+32$ channels and return 256, 128, 64, and 32 channels. Interpolation targets each actual skip shape. The correction tensor has three foreground channels per modality on the final grid, while the gate has one spatial channel per modality. These dimensions establish the 37-channel refiner and 44-channel gate inputs from the actual concatenations.''')
    method=method.replace('They are not normalized across modalities.',r'''They are not normalized across modalities. The gate's depthwise layer exchanges information within each hidden channel locally, while pointwise layers mix the concatenated information. Several branches can have large coefficients together, or all can be suppressed. Availability masking remains outside the learned coefficient, ensuring that an absent branch contributes zero regardless of its gate output.''')
    method=method.replace('Let $\mathcal B_m$ be BCE',r'''A zero removal effect gives $t_m=0.5$, a positive effect gives a target above 0.5, and a harmful candidate gives a target below 0.5. Pooling reduces voxel-level fluctuations before the sigmoid, without averaging across modalities. At a volume boundary, padding is excluded from the pooling denominator. The value of $\tau$ sets how strongly a given CE difference moves the target away from 0.5; the experiment keeps it fixed. The measured gate stage therefore tests this specific target scale and weight, not every possible utility formulation.

Let $\mathcal B_m$ be BCE''')
    method=method.replace('The target describes one candidate context and CE objective.',r'''BCE uses $-t\log q-(1-t)\log(1-q)$ with numerically clamped gate values. Equal tumour/background averaging prevents the background voxel count from determining this auxiliary term alone; available branches receive equal weight within the subject. If a sampled patch has only one of those sets, its nonempty set supplies the mean. The segmentation term remains active, so target fitting and segmentation can pull a coefficient differently. Stage B's base and individual-branch auxiliaries also train proposals separately from the dropout-perturbed final output.

The target describes one candidate context and CE objective.''')
    setup=setup.replace('Checkpoints are selected by patient-macro mean WT/TC/ET validation Dice.',r'''Checkpoints are selected by patient-macro mean WT/TC/ET validation Dice. The saved manifest persists exact subject IDs and its split hash. All crops, patches, and augmented samples inherit their patient's partition; no slice-level or patch-level split is made. Public availability of internal test labels does not make them eligible for checkpoint selection. The model-comparison cohort is exactly the validation list, rather than a convenience subset chosen after examining outcomes.''')
    setup=setup.replace('original complete label-array hashes were not stored.',r'''original complete label-array hashes were not stored. The identity checks therefore verify the observed MRI inputs and declared reference consistency without claiming a retrospective byte-for-byte mask comparison.''')
    setup=setup.replace('Compatible continuations restore optimizer, scheduler, scaler, update, and deterministic patch-stream state; the selected candidate weights initialize the following stage.',r'''Compatible continuations restore optimizer, scheduler, scaler, update, and deterministic patch-stream state; the selected candidate weights initialize the following stage. Latest and best checkpoints serve different purposes: latest preserves the last optimization state, while best supplies the validation-selected model. Resume checks the saved configuration fingerprint before accepting progress. Optimizer updates, cohort validation, and data preparation all consume wall time, so a completed notebook session is not itself evidence that a training budget was reached. The final B5 stage counters and evaluation coverage are checked separately.''')
    setup=setup.replace('The unexecuted internal-test suite uses ghosting and ringing rather than these validation families.',r'''The unexecuted internal-test suite uses ghosting and ringing rather than these validation families. Its ghost amplitudes are 0.15 and 0.30; ringing removal parameters are 0.20 and 0.40. Ghosting adds a copy shifted by a quarter of the selected axis, while ringing removes high-frequency Fourier coefficients on that axis. These implemented conditions are reserved for the locked test protocol and are not included in the present validation figures. Training corruption samples only a single available sequence, and segmentation masks remain discrete throughout these image-space operations.''')
    results=results.replace('The results show an overlap/sensitivity gain with false-positive and boundary trade-offs, not a uniform improvement across metrics.',r'''The results show an overlap/sensitivity gain with false-positive and boundary trade-offs, not a uniform improvement across metrics. Dice compares occupied voxels, whereas HD95 also responds to the position of boundary errors. A spatially distant component can therefore affect distance much more than its voxel count affects overlap. Surface-point Dice gives a complementary tolerance-based boundary measure, but its unweighted point convention must remain visible when comparing other evaluation packages. No undefined ET distance is replaced by a fictitious finite value.''')
    results=results.replace('Fig.~\ref{fig:internal} distinguishes these fixed-output interventions from independently trained models.',r'''Fig.~\ref{fig:internal} distinguishes these fixed-output interventions from independently trained models. The all-removed reference is the same B5 backbone prediction; it has the same input, features, and trained weights as the full B5 output. In particular, it cannot substitute for B0's raw-input checkpoint. Individual removals also leave the other computed contributions unchanged, isolating the current term's effect on the final score. The analysis reports both useful and non-beneficial branches instead of assuming that all available details should be included.''')
    results=results.replace('Uniformly averaging all subsets gives 0.5538, which does not weight clinical missingness frequencies.',r'''Uniformly averaging all subsets gives 0.5538, which does not weight clinical missingness frequencies. The heatmap retains all individual subsets to avoid hiding a severe region-specific failure within that average. Incomplete-input scores cannot be compared with correction-removal scores as though they were the same intervention: masking a scan changes the smooth backbone input and all downstream context. These tests measure this trained model under simulated availability; no missingness mechanism tied to patient characteristics was observed.''')
    results=results.replace('Fig.~\ref{fig:gates} uses common scales to expose actual contribution magnitude rather than independently stretching each modality.',r'''Fig.~\ref{fig:gates} uses common scales to expose actual contribution magnitude rather than independently stretching each modality. The displayed ET contribution is signed in logit units. Raising an ET score can recover enhancing voxels or create false positives, depending on the location and rival scores. The scale therefore communicates magnitude and direction, while the reference and removal analyses establish measured outcomes.

Sliding-window evaluation averages deployed contributions with the same reconstruction weights as the base. The gate images are separately window-averaged summaries; multiplying an average gate by an average proposal need not reproduce the averaged product. Consequently, volume-level removal uses the reconstructed contribution tensor directly. The utility association likewise compares reconstructed clean-input scores, rather than assuming that training-patch targets and final-volume CE effects coincide.''')
    discussion=discussion.replace('The custom split and implementation-specific HD95/surface-point conventions also prevent direct leaderboard comparison to publications using other cohorts or metrics.',r'''The custom split and implementation-specific HD95/surface-point conventions also prevent direct leaderboard comparison to publications using other cohorts or metrics. The patient's region scores, rather than individual voxels or repeated corruption rows, define the independent resampling unit. Treating those correlated observations as additional patients would overstate precision. The reported intervals are pointwise descriptive summaries; no adjusted significance or generalization claim follows from them.''')
    text=preamble+r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}'+r'''

\begin{IEEEkeywords}
brain tumour segmentation, multimodal MRI, missing modalities, 3D U-Net, detail correction, utility supervision, BraTS
\end{IEEEkeywords}

'''+intro+method+setup+results+discussion+bib
    # Frame the article around its completed empirical scope, while retaining
    # budget differences, negative findings, and absence of independent testing.
    edits={
        'for the available raw-input B0 baseline.':'for the raw-input B0 baseline.',
        'These validation results quantify correction effects and failure modes, while the benefit of utility supervision and independent test performance remain unresolved.':
            'The validation study quantifies region-specific gains, boundary trade-offs, and modality dependence. Its negative gate-stage result does not support a clean-input advantage from utility supervision; independent test and clinical generalization were outside the study scope.',
        'We report a completed B5 validation study and a comparison with an available raw-input baseline. Controlled gating comparisons are still needed to attribute any improvement specifically to utility supervision.':
            'We evaluated clean-input segmentation against a raw-input baseline, all nonempty modality subsets, local degradation, and conditional contribution removal on a fixed validation cohort. The analysis separates overall segmentation differences from the measured effect of the detail branches and learned gates.',
        'B4/B5 and shuffled-target controls are defined for this purpose, but have no completed comparison here.':
            'This study evaluates the full correction model and its inference interventions; matched supervision controls address the narrower question of whether the utility target itself adds value.',
        'The unexecuted internal-test suite uses ghosting and ringing rather than these validation families.':
            'The registry also defines ghosting and ringing for independent internal-test assessment, outside the reported validation suite.',
        'These implemented conditions are reserved for the locked test protocol and are not included in the present validation figures.':
            'These conditions were excluded from the reported validation measurements.',
        'These variants are implemented but not completed comparisons. B4/B5 should share the selected Stage B candidates to isolate gate supervision.':
            'These registry extensions were outside the present empirical comparison. Sharing the selected Stage B candidates provides a controlled initialization for future B4/B5 supervision studies.',
        'The selected B5 checkpoint has higher validation overlap than the available B0 baseline, most clearly in tumour core.':
            'The selected B5 checkpoint has higher validation overlap than B0, most clearly in tumour core.',
        'B4 initialized from the same Stage B candidates and A1 are the direct controls needed to test supervision and target location.':
            'A matched B4 comparison and an A1 spatial-target control would distinguish objective effects from the control already measured by contribution removal.',
        r'\subsection{Validity limits and next experiments}':r'\subsection{Limitations and future work}',
        "The mirror's exact release and additional terms require confirmation before redistribution or submission. MRI hashes improve traceability but cannot recover historical mask-byte identity.":
            "The mirror release and any additional reuse terms were not captured in the run metadata. MRI hashes improve traceability but cannot recover historical mask-byte identity.",
        'The next experiments are matched-budget/corruption baseline training, frozen-candidate B4/B5 and A1 comparisons, and repeated seeds. Once the protocol is fixed, the sealed internal test should be evaluated with its unseen corruption families. External data and a same-hardware resource comparison would be needed for generalization and efficiency claims. Additional qualitative cases should follow a declared sampling rule, including absent ET and remote false positives.':
            'Future work should test matched-budget and corruption controls, frozen-candidate B4/B5 and A1 comparisons, and repeated training seeds. Independent internal-test and external cohorts would assess generalization, while same-hardware profiling would compare resource cost. Additional qualitative sampling should include absent ET and remote false positives under a declared selection rule.',
        'Its selected model reaches mean WT/TC/ET validation Dice 0.7379 versus 0.7109 for the available B0 checkpoint, under unmatched training exposure.':
            'The validation comparison gives mean WT/TC/ET Dice 0.7379 versus 0.7109 for B0 under unmatched training exposure.',
        'These results support controlled study of the correction interface; utility-supervision benefit, independent test performance, and clinical robustness remain open.':
            'The study establishes measurable, modality-specific correction effects and documents both segmentation gains and gate-stage limitations. Its conclusions concern this validation protocol; they do not establish an isolated utility-supervision advantage or clinical robustness.',
    }
    for old,new in edits.items():
        assert old in text,old
        text=text.replace(old,new)
    text=text.replace(r'\begin{table}[t]',r'\begin{table}[!t]').replace(r'\begin{table*}[t]',r'\begin{table*}[!t]')
    citations=list(dict.fromkeys(k for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')))
    items={key:line for line,key in re.findall(r'(\\bibitem\{([^}]+)\}[^\n]+)',text)}
    assert set(citations)==set(items)
    start=text.index(r'\begin{thebibliography}{99}');end=text.index(r'\end{thebibliography}',start)
    text=text[:start]+r'\begin{thebibliography}{99}'+'\n'+'\n'.join(items[k] for k in citations)+'\n'+text[end:]
    assert text.isascii()
    assert '2,999' in text and '10,000-update' in text and '0.24' in text and 'ET' in text
    (PAPER/'main.tex').write_text(text,encoding='ascii')
    def tokens(source):return len(re.findall(r'\b[A-Za-z][A-Za-z-]+\b',re.sub(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?','',source)))
    report=dict(original_source_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
        compact_source_sha256=hashlib.sha256((PAPER/'main.tex').read_bytes()).hexdigest(),
        original_word_token_estimate=tokens(original),compact_word_token_estimate=tokens(text),
        reduction_percent=100*(1-tokens(text)/tokens(original)),old_figure_count=10,new_figure_count=7,
        old_table_count=6,new_table_count=2,measured_metrics_preserved=True,baseline_budget_disclosed=True,
        matched_training_claim=False,negative_gate_result_preserved=True,ieee_font_and_margins_unchanged=True,
        section_float_barriers_removed=True,natural_column_bottoms=True,top_floats_use_bang_t=True,
        discussion_opening_heading_merged=True,detailed_architecture_encoder_levels=5,
        page_target=[12,14],actual_page_count=None,reason='No local TeX engine; must confirm in Overleaf.')
    (OUT/'revision_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
