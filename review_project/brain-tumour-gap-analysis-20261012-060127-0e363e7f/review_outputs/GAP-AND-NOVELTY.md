# A research direction for incomplete brain-tumour MRI segmentation

Status: provisional research recommendation, 12 October 2026. No experiments have been run and architectural novelty has not been established. This is a targeted method comparison and novelty check, not a systematic or exhaustive review.

## Recommendation

Investigate **local, frequency-band-specific evidence routing under jointly missing and degraded MRI sequences**. The proposed mechanism should learn whether retaining a particular modality's frequency component helps segmentation at a given location. Its value must be demonstrated against simpler dropout-trained and quality-gated baselines.

In everyday terms: a scan can be sharp in one part and unreliable in another. Even in the same part, its broad anatomical information may remain useful while its fine details contain artifacts. We want to retain the useful information without allowing damaged components to dominate the prediction.

A possible working title is **Learning Local Spectral Utility for Brain Tumour Segmentation with Missing and Degraded MRI**. This title describes a hypothesis; it is not a claim of novelty or improved performance.

## What was actually available

The local `papers` directory contained two PDF files, not eleven folders or eleven PDFs. Both were converted to searchable text and inspected, including their methods, experiments, and failure discussion:

- HFF-Net: the supplied file identifies itself as arXiv:2506.10142v1, June 2025.
- MoME: the supplied file identifies itself as arXiv:2405.10246v2, July 2024. Its filename resembles the later journal title, but the contents are the earlier preprint; a MoME+ extension is not established from this file.

`notion.md` was read as the user's secondary explanation notes. The other nine methods were checked using author repositories, original abstracts, or an author institution's publication record where accessible. These checks do not substitute for reading the nine missing full texts and supplements. No published performance numbers were pooled or used to rank architectures across different protocols.

## What the eleven papers contribute

| Paper | Relevant contribution, at the inspected access level | Evidence available |
| --- | --- | --- |
| D²-Net | Spatial–frequency modality contrastive learning plus region-level distillation | [Official implementation](https://github.com/CityU-AIM-Group/D2Net), overview; notes |
| Unified Hyper-Network | Adaptive image/feature completion followed by segmentation | [Original abstract](https://pubmed.ncbi.nlm.nih.gov/37540616/); notes |
| MoME | Modality experts, hierarchical gating, curriculum training; unseen-dataset evaluation | Local preprint, Abstract, Methods, Table 2; [record](https://arxiv.org/abs/2405.10246v2) |
| HFF-Net | Separate frequency representations and adaptive edge processing; partial-FLAIR failure analysis | Local full text, IV-C5/Fig. 11 and Conclusion; [full text](https://arxiv.org/html/2506.10142v1) |
| MBUNeXt | Feature attention, encoder aggregation and large-kernel skips | [Official implementation](https://github.com/liuqinghao2018/MBUNeXt), overview; notes |
| Medical Transformer | Multi-plane, slice-based representation with healthy-brain self-supervised pretraining | [Original preprint abstract](https://arxiv.org/abs/2104.13633); notes |
| SwinPA-Net | Multiscale aggregation; original evaluated tasks include polyps, skin lesions and laparoscopy | [Original abstract](https://pubmed.ncbi.nlm.nih.gov/36121961/); notes |
| MMTSeg | Feature distillation, Mamba–Transformer enhancement/fusion, boundary loss | [Current original abstract](https://arxiv.org/abs/2404.14019v3); notes |
| M²FTrans / MFTrans | Missing-modality-aware transformer fusion | [Official implementation](https://github.com/Jun-Jie-Shi/M2FTrans), overview; notes |
| DCTC-Net | Dynamic deformable convolution and complementary attention in a hybrid architecture | [Original abstract](https://pubmed.ncbi.nlm.nih.gov/41308087/); notes; brain-tumour-specific applicability remains unverified |
| M²SegMamba | Masked pretraining, multi-traversal Mamba, skip integration and regularization for few-sample incomplete inputs | [Author institution record](https://pure.etsmtl.ca/en/publications/msup2supsegmamba-mamba-based-incomplete-multimodal-learning-for-b/), abstract; notes |

This matrix maps mechanisms, not shortcomings. An unreported mechanism in an abstract is not proof that the full paper omitted it. In particular, M²SegMamba should not be described as necessarily requiring a complete-modality teacher based on these notes alone.

## Corrections that change the gap analysis

1. **Frequency + missing modality already exists.** D²-Net's own overview describes spatial–frequency learning. [PF-CMNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC13297487/) explicitly combines frequency-aware cross-modal learning with missing-modality distillation; its abstract distinguishes entire-sequence absence stress tests from partial degradation. The notes' broad novelty claim is therefore untenable.
2. **Quality-aware missing-modality fusion already exists.** [AMGFormer](https://arxiv.org/html/2601.19349v1), Methods 2.2 and 2.4, has spatial fusion and global modality-quality scores. [CoReFuse-Med](https://arxiv.org/html/2609.10261v1), Abstract and Methods, addresses resolution-related corruption and rebalanced fusion. A generic quality gate would have substantial overlap.
3. **Missing-T1ce uncertainty is already studied.** [UAF-AIMM](https://pmc.ncbi.nlm.nih.gov/articles/PMC13319089/), Abstract and Discussion, combines feature compensation and uncertainty-aware fusion. [Set-Inclusive Uncertainty Modeling](https://papers.miccai.org/miccai-2026/0949-Paper1877.html), abstract, studies missing-evidence uncertainty across modality subsets. Adding an uncertainty head is insufficient as a novelty claim.
4. **Generalisation is not exclusive to MBUNeXt.** The local MoME preprint has an explicit unseen-dataset evaluation in Table 2. Population transfer, unseen-dataset inference, and retraining on another dataset must be distinguished.
5. **The notes' universal ET Dice ceiling near 0.60 is unsupported.** Neither its universality nor an information-theoretic bound follows from this corpus. Loss of direct enhancement evidence motivates careful evaluation; it does not establish a fixed numerical limit.
6. **Metric and clinical simplifications need correction before paper drafting.** HD95 is a 95th-percentile surface-distance measure, not the maximum error. Inter-rater Dice is not a universal upper bound on agreement with a benchmark reference. ET is contrast enhancement, not an exact map of all active tumour. Dice+cross-entropy is a legitimate combination; the local MoME methods explicitly use it. Masking absent inputs also does not guarantee that a discriminative model cannot make unsupported predictions.

## The narrower gap worth testing

HFF-Net's IV-C5/Fig. 11 explicitly links partial FLAIR corruption to subregion confusion. The inspected AMGFormer mechanism estimates sequence-level quality through global pooling; its spatial fusion is separate. CoReFuse-Med focuses principally on aligned resolution mismatch. These are reasons to investigate the proposed mechanism, not evidence that every existing method lacks it.

**Research question:** When an available MRI sequence is only partly degraded, can supervised estimates of local frequency-component usefulness preserve helpful signal better than binary availability masks, whole-sequence quality scores, or a spatial gate without frequency separation?

**Hypothesis:** A local low-/high-frequency utility gate will reduce boundary errors and severe subregion failures under simultaneous missingness and degradation, without losing useful clean-input performance.

The candidate contribution is the precise combination of **local spectral routing and task-utility supervision**, established by direct comparison. Wavelets, gates, region-specific fusion, corruption training and uncertainty estimates individually are prior techniques. A larger search and the closest papers' supplements/code are still required before any priority claim.

## A minimal architecture to test

Use a modest 3D U-Net-style encoder–decoder. Share encoder weights across modalities with a modality embedding, or use small modality-specific stems; keep this choice fixed across comparisons. Avoid adding Mamba, a foundation-model pretraining stage, hypernetworks and multiple teachers to the first experiment.

1. **Record actual availability.** Supply a binary mask for T1, T1ce, T2 and FLAIR. Missing components must have zero fusion weight. Distinguish an absent sequence from a present sequence with a corrupted patch.
2. **Separate two frequency groups.** At one or two shallow encoder scales, apply a fixed 3D wavelet decomposition. Begin with one low-frequency group and a grouped high-frequency representation; defer separate directional experts until an ablation justifies them. Project the groups to compatible coarse grids.
3. **Predict local utility.** A small 3D head predicts a score per modality, frequency group and coarse spatial location, conditioned on the observed features and availability. It sees neither segmentation labels nor the synthetic corruption mask at inference.
4. **Fuse groups separately.** Mask absent modalities, normalize weights over available modalities within each group, then combine the low-/high-frequency fused features in the decoder. Keep an explicit low-confidence indicator if all scores are low; softmax normalization alone cannot express that all evidence is poor.
5. **Predict voxel labels.** Use the selected dataset's actual class scheme and derive ET/TC/WT for evaluation. Do not hard-code label IDs from a different BraTS release.

For modality m, group b, location v, a candidate rule is:

`alpha[m,b,v] = masked_softmax_m(g[m,b,v])`

`fused[b,v] = sum_m alpha[m,b,v] * feature[m,b,v]`

The novel part to investigate is how `g` is trained, not the weighted sum.

### Train the gate on segmentation usefulness

After warming up a reference segmenter with modality dropout and corruption augmentation, freeze it temporarily. Sample a modality/group and compare its predictions with that component retained versus suppressed. On training data only, compute:

`utility[m,b,v] = local_loss(component_suppressed, label) - local_loss(component_retained, label)`

A positive value means keeping the component helped in that local neighbourhood. Regress this stopped-gradient target, or supervise useful/harmful ordering, with the small utility head. Use training labels to form targets, never test labels. Sample one component per batch initially and record the additional forward-pass cost.

This target estimates usefulness for a particular trained model; it does **not** identify biological causality or prove a component contains objectively correct anatomy. Suppression can itself create distribution shift, so the reference must be trained with the same component-masking scheme. Refresh targets cautiously if the segmenter changes; initially test fixed targets to avoid a moving-target feedback loop.

Distinguish usefulness from corruption detection: a clean scan can have little information for one subregion, and a noisy scan can still contain valuable information. Compare task-utility training against synthetic corruption-mask supervision. If either simpler target performs equally well, retain the simpler approach and narrow the claim.

Do not enforce identical outputs between complete and incomplete inputs: genuinely missing evidence can justify different predictions. Do not enforce monotonic improvement when adding a corrupted component.

## Other directions considered

| Candidate | Assessment |
| --- | --- |
| Frequency fusion plus modality dropout | Broad overlap with D²-Net/PF-CMNet; reject as the sole novelty |
| New Mamba traversal order | Learnable sorting is already studied in CVPR 2025; requires a specific additional mechanism |
| MoE for missing sequences | Existing expert and missing-modality methods; large scope |
| Generic confidence-aware missing-T1ce segmentation | Direct overlap with UAF-AIMM and SIUM |
| Local spectral utility routing | Preferred: concrete failure, narrow mechanism and useful falsification tests |
| Adaptive reacquisition advice | Interesting backup: predict benefit of each feasible next sequence; needs paired data, cost assumptions and a separate novelty search |
| Cross-center few-label spectral adaptation | Possible backup, but combines too many constraints for a first experiment |
| Teacher distillation restricted to recoverable evidence | Worth studying later; close distillation/uncertainty overlap and costly training |
| Standardized missing-plus-degraded stress benchmark | Useful companion contribution; publishability depends on diagnostic insight and rigor, not a new metric table alone |
| Misregistration-aware local fusion | Realistic boundary case; needs registration-specific prior-art inspection |
| Subregion-aware healthy false-positive control | Useful evaluation direction; requires no-tumour data and scope beyond tumour-only BraTS |
| Low-resource shared-encoder fusion | Practical baseline; efficiency alone needs matched quality and cost evidence |

Top three: local spectral utility routing; a rigorous stress benchmark as part of that work; reacquisition-benefit modeling if suitable paired data become available. These are judgments, not scored scientific findings.

## Experiments that decide whether to proceed

Keep the same patient split, preprocessing, augmentation, training budget and backbone throughout. Split by patient before extracting patches. Check patient overlap when adding datasets or BraTS years. Do not treat another BraTS year as automatically independent external validation.

**Minimum ablations:** (A) backbone with modality dropout; (B) A plus identical corruption augmentation; (C) B plus global modality-quality gating; (D) B plus a spatial gate without frequency decomposition; (E) D plus low-/high-frequency separation with ordinary segmentation supervision; (F) E plus utility supervision. Match parameter capacity where practical and report remaining differences. The decisive comparisons are F versus D/E, not only F versus A.

**Test regimes:** complete clean inputs; all 15 nonempty modality subsets; present-but-degraded inputs; joint missing-plus-degraded inputs. Include whole-sequence and local-region perturbations. Select a declared stress matrix rather than pretending to cover every possible corruption combination. Synthetic missingness is a controlled proxy, not evidence of performance on naturally missing clinical data.

Use noise, bias-field/intensity variation, through-plane resolution loss and MRI-appropriate simulated motion. Keep at least one corruption family or severity range out of training. If misregistration is included, describe it separately from signal-quality degradation. Never supply ground-truth degradation masks at test time.

**Primary outcomes:** ET/TC/WT Dice and HD95 in physical units, reported by subset and corruption regime. Predeclare empty-reference-region handling. Report per-patient distributions, lesion-size strata, severe failures and false positives, not just a mean. Any confidence/deferral analysis is secondary: assess error detection and risk–coverage on held-out patients, with thresholds selected only on validation data.

Bootstrap paired patient-level differences; do not count voxels or multiple corruptions of the same patient as independent patients. Confirm a promising result with multiple seeds. Record parameters, measured peak VRAM, training hours and full-volume inference time, including all extra routing/calibration passes.

**Failure criterion:** If frequency separation or utility supervision adds no reproducible benefit over the spatial gate with identical augmentations, drop the architectural novelty claim. If gains vanish under unseen artifacts, describe robustness as limited to the tested simulation. If the training overhead is disproportionate, simplify before a full study.

## A two-week pilot, subject to available compute

- Days 1–3: choose one accessible BraTS release, verify labels/splits, reproduce a modest baseline and define evaluation/empty-region rules.
- Days 4–6: quantify failures with missing T1ce, partial FLAIR degradation and through-plane resolution loss. If the baseline does not show the predicted failure, revise the problem before adding modules.
- Days 7–10: compare corruption augmentation, global gating and spatial gating; add the two-group utility gate only after these work.
- Days 11–14: evaluate a held-out corruption, paired patient differences and measured cost. Repeat the strongest contrast if resources permit.

This is a proposed schedule, not a promise that training will fit an unknown GPU. GPU/VRAM, available dataset, label budget and target venue remain unresolved. The next decision is whether the pilot exposes the intended failure and whether the narrow mechanism survives the strongest simpler baseline.

## Source and novelty boundaries

HFF-Net and MoME were examined as the supplied preprint versions, not verified final journal PDFs. Exact performance claims from `notion.md` were not adopted. For the other papers, limitations absent from accessible abstracts/repositories remain unknown. Additional 2026 work was included to challenge the candidate novelty; preprints are labeled as such and author claims are not independently reproduced.

Before committing to a paper, inspect full texts and code for PF-CMNet, AMGFormer, CoReFuse-Med, SIUM, UAF-AIMM, RFNet and D²-Net. Specifically look for local spectral reliability, target generation by component suppression, and combined missing/degraded evaluation. This targeted search did not find an exact verified match to the proposed utility-supervised mechanism, but cannot establish its absence from the literature.
