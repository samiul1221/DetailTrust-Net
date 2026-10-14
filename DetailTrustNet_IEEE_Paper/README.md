# DetailTrust-Net IEEE paper

Upload **DetailTrustNet_IEEE_Paper_Overleaf.zip** as a new Overleaf project, or upload `main.tex` and `figures/` from this folder. Select **main.tex**, **pdfLaTeX**, and **Normal** compilation; recompile from scratch. The ZIP has `main.tex` at its root. All prose, equations, tables, captions, and 18 bibliography entries are in that one file. No BibTeX run is needed.

For a local TeX installation:

```sh
cd DetailTrustNet_IEEE_Paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Evidence in this version

The paper uses the completed B5 output downloaded on 2026-10-14 at 06:24:37 UTC and the finished local B0 evaluation. Both models were scored on the same 49 clean validation patients. B5 also has all 15 modality subsets, eight local corruption conditions, five contribution-removal conditions, and clean-input gate/stability analyses. B0 has clean-input evaluation only.

Mean WT/TC/ET validation Dice is **0.7109 for B0** and **0.7379 for B5**, a paired difference of **2.70 percentage points** (pointwise patient-bootstrap interval 0.33–4.99). ET's interval includes zero. Mean HD95 does not improve. B5 completes 4,000/4,500/1,500 stage updates; B0 uses the update-2,000 selected checkpoint from an available 2,999-update history. Training exposure and corruption augmentation differ; the manuscript records these details.

The final learned gates score 0.24 points below the selected unit-gate Stage B model. The paper reports this negative observation. Controlled gate-supervision comparisons, repeated training seeds, external validation, and the untouched 101-patient internal test have no results. They are discussed as limitations and future experiments rather than populated with fabricated tables or images.

## Figures

There are **no result placeholders** in this package. Seven figures cover the 2×2 architecture, training workflow, combined baseline/training plots, real MRI/prediction panels, internal correction removal, combined robustness panels, and actual gates/contributions. Each has PDF and PNG versions; the manuscript uses PDF. The registry is condensed to a protocol paragraph, and six tables are consolidated into two without changing measured values.

`figures/architecture.drawio` and `figures/training_pipeline.drawio` are editable diagrams.net sources. Their PDF/PNG versions were rendered from the same XML with a local renderer because a Draw.io desktop exporter was unavailable. Open the `.drawio` files to edit them and re-export to the same PDF filenames after changes. Figure 1 is the detailed 2×2 architecture: all five encoder levels, four decoder levels and skips, feature/logit heads, both anchored refiner evaluations, the separate spatial gate, detached utility target, and fusion. Figure 2 is the optimization workflow. The exact architecture PDF/PNG was visually inspected; its text was also measured against each box at the final physical width.

The compact MRI panel uses BraTS20_Training_205, the first validation ID, at axial RAS slice 90 (median reference-tumour coordinate). Baseline inference was recomputed and its full-volume Dice checked against the saved result. The earlier three-plane views and separate figures are archived outside this upload folder. Captions distinguish independent B0 predictions from B5's own smooth-input base. `figures/figure_manifest.txt` records each figure's provenance.

## Verification and submission decisions

The source uses IEEEtran conference layout with its default margins and body fonts. Section-level float barriers have been removed; figures and tables use ordinary top placement with `[!t]`. Standard `\raggedbottom` prevents a short column from stretching paragraph and heading spaces. Discussion opens directly with its text, eliminating the redundant isolated subsection heading. No negative spacing, forced page breaks, or forced-position figures are used. These source changes address the blank areas in the previous rendering; final placement must still be checked after compilation.

Prose and captions are substantially shortened, with 12–14 pages as a target rather than a verified count. Static checks pass for LaTeX structure, every label/citation, asset paths, ASCII-safe source, Draw.io XML, and package contents. Figure PDFs open and their text fits within page bounds. **No local TeX engine is installed, so manuscript compilation, final page count, and TeX overfull-box checks remain unverified.** See `build_check.txt`; compiling in Overleaf is the final layout check.

Reference metadata was checked against publisher/proceedings pages, author repositories, university records, or indexed original abstracts. Kang et al.'s September 2026 issue pagination was refreshed. No reference is intentionally left with invented metadata. Confirm the dataset mirror's exact release/reuse terms and the venue's length, disclosure, and secondary-data requirements before submission; training did not record those terms.

Generators and numerical evidence remain outside this upload folder, under `tools/paper/`, `verification/paper_finalization/`, and `local_baseline_results/`. No Python code, credentials, MRI data, caches, or checkpoints are included in the Overleaf ZIP.
