"""Audit and package the single-source IEEE project without data or code."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
import numpy as np
import fitz
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'DetailTrustNet_IEEE_Paper'
AUDIT=ROOT/'verification/paper_finalization'
B5=ROOT/'output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z/detailtrust'

README='''# DetailTrust-Net IEEE paper

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

The source uses IEEEtran conference layout with its default margins and body fonts. Section-level float barriers have been removed; figures and tables use ordinary top placement with `[!t]`. Standard `\\raggedbottom` prevents a short column from stretching paragraph and heading spaces. Discussion opens directly with its text, eliminating the redundant isolated subsection heading. No negative spacing, forced page breaks, or forced-position figures are used. These source changes address the blank areas in the previous rendering; final placement must still be checked after compilation.

Prose and captions are substantially shortened, with 12–14 pages as a target rather than a verified count. Static checks pass for LaTeX structure, every label/citation, asset paths, ASCII-safe source, Draw.io XML, and package contents. Figure PDFs open and their text fits within page bounds. **No local TeX engine is installed, so manuscript compilation, final page count, and TeX overfull-box checks remain unverified.** See `build_check.txt`; compiling in Overleaf is the final layout check.

Reference metadata was checked against publisher/proceedings pages, author repositories, university records, or indexed original abstracts. Kang et al.'s September 2026 issue pagination was refreshed. No reference is intentionally left with invented metadata. Confirm the dataset mirror's exact release/reuse terms and the venue's length, disclosure, and secondary-data requirements before submission; training did not record those terms.

Generators and numerical evidence remain outside this upload folder, under `tools/paper/`, `verification/paper_finalization/`, and `local_baseline_results/`. No Python code, credentials, MRI data, caches, or checkpoints are included in the Overleaf ZIP.
'''


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit_figures():
    generated=json.loads((AUDIT/'generated_figures.json').read_text())
    for name,meaning in [('architecture','Detailed 2x2 Draw.io architecture: masked physical-scale Gaussian split; five encoder/four decoder levels and skips; both zero-anchored refiner evaluations and separate spatial gate; detached utility supervision and foreground fusion. Shapes and layer widths checked against source.'),
                         ('training_pipeline','Three completed B5 training stages; validation selection and sealed internal test.')]:
        generated[name]={'source_artifacts':[f'figures/{name}.drawio','detailtrust/model.py','detailtrust/losses.py'],
                         'meaning':meaning,'fabricated_values':False}
    assets=[]
    for name,info in sorted(generated.items()):
        path=PAPER/'figures'/f'{name}.pdf'
        doc=fitz.open(path);assert len(doc)==1
        page=doc[0];bad=[];sizes=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox'];sizes.append(span['size'])
                    if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:bad.append(span['text'])
        assert not bad,(name,bad)
        fonts=page.get_fonts()
        assert all(f[2]!='Type3' for f in fonts),(name,fonts)
        with Image.open(path.with_suffix('.png')) as preview:
            preview.verify()
        assets.append(dict(name=name,source=info,sha256=hash_file(path),
            preview_sha256=hash_file(path.with_suffix('.png')),pdf_size_points=list(page.rect)[2:],
            pdf_text_within_page=True,embedded_font_types=sorted({f[2] for f in fonts}),
            minimum_text_size_points=min(sizes),pdf_pages=1))
        doc.close()
    (AUDIT/'figure_audit.json').write_text(json.dumps(assets,indent=2))
    lines=['Figure manifest -- all assets in this package reflect measured data or verified implementation.','',
           'B5 source: output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z/detailtrust',
           'B0 source: local_baseline_results; selected checkpoint output_dump/B0_s1729/RAW/best.pt.',
           'Each figure has PDF and 300-DPI-or-higher PNG; the manuscript includes PDF.',
           'No experimental result placeholders remain.','']
    for asset in assets:
        info=asset['source'];lines.extend([asset['name']+'.pdf / .png',info['meaning'],
            'Sources: '+', '.join(info['source_artifacts']),'PDF SHA-256: '+asset['sha256'],''])
    (PAPER/'figures/figure_manifest.txt').write_text('\n'.join(lines),encoding='utf-8')
    return assets


def complete_numerical_ledger():
    with (AUDIT/'results_ledger.csv').open(newline='') as h:ledger=list(csv.DictReader(h))
    def record(key,value,unit,source,locator,claim=None):
        ledger.append(dict(claim_id=key,section='Methods/Results',claim_text=claim or key,
            value=value,unit=unit,dispersion='conditional on the recorded single-seed evaluation',
            source_artifact=source,source_locator=locator,status='recomputed_or_verified'))
    b0=json.loads((ROOT/'local_baseline_results/local_baseline_report.json').read_text())
    result=json.loads((B5/'experiments/B5_s1729/result.json').read_text())
    for r in result['stages']:
        record('B5_stage_'+r['stage']+'_updates',r['updates_completed'],'updates','B5 result.json',r['stage']+'.updates_completed')
        record('B5_stage_'+r['stage']+'_selected',r['selected_update'],'updates','B5 result.json',r['stage']+'.selected_update')
        record('B5_stage_'+r['stage']+'_score',r['validation_score'],'Dice','B5 result.json',r['stage']+'.validation_score')
    record('gate_stage_change_pp',100*(result['stages'][2]['validation_score']-result['stages'][1]['validation_score']),
           'percentage points','B5 result.json','C.validation_score - B.validation_score')
    record('B0_CPU_median_seconds',np.median([t['seconds'] for t in b0['inference']['timings']]),
           'seconds','local_baseline_report.json','inference.timings.seconds; 49 patients')
    record('B5_GPU_median_seconds',result['inference_benchmark']['median_seconds'],'seconds','B5 result.json','inference_benchmark')
    record('B5_GPU_inference_allocated_GiB',result['inference_benchmark']['peak_allocated_bytes']/2**30,'GiB','B5 result.json','inference_benchmark.peak_allocated_bytes')
    record('B0_parameters',b0['model_parameters']['total'],'parameters','local_baseline_report.json','model_parameters.total')
    record('B5_parameters',result['parameters']['total'],'parameters','B5 result.json','parameters.total')
    record('parameter_difference_percent',100*(result['parameters']['total']/b0['model_parameters']['total']-1),
           'percent','B0/B5 reports','parameter count relative difference')
    split=json.loads((B5/'split_manifest.json').read_text())
    for s in ('train','validation','test'):
        record('split_'+s,len(split[s]),'patients','split_manifest.json',s)
    with (B5/'metrics_summary.csv').open(newline='') as h:summary=list(csv.DictReader(h))
    lookup={(r['condition'],r['region']):json.loads(r['dice'])['mean'] for r in summary}
    def mean(c):return np.mean([lookup[c,r] for r in ('WT','TC','ET')])
    for condition in sorted({r['condition'] for r in summary}):
        record(condition+'_mean_region_dice',mean(condition),'Dice','B5 metrics_summary.csv',condition+',mean(WT,TC,ET)')
        if condition.startswith(('noise','resolution')):
            clean='clean_m13' if '_m13_' in condition else 'clean_m15'
            record(condition+'_change_pp',100*(mean(condition)-mean(clean)),'percentage points','B5 metrics_summary.csv',condition+' minus '+clean)
    record('all_detail_removal_change_pp',100*(mean('clean_m15')-mean('clean_m15_base_only')),'percentage points','B5 metrics_summary.csv','clean_m15 minus clean_m15_base_only')
    record('mean_all_15_subsets',np.mean([mean(f'clean_m{i:02d}') for i in range(1,16)]),'Dice','B5 metrics_summary.csv','uniform average across 15 subsets, equal region weighting')
    stability=json.loads((B5/'stability_analysis.json').read_text())
    checks=stability['enumeration_checks'];assert len(checks)==49 and all(r['passed'] for r in checks)
    record('stability_max_enumeration_error',max(r['max_margin_error'] for r in checks),'logit margin','stability_analysis.json','enumeration_checks.max_margin_error')
    ledger=list({r['claim_id']:r for r in ledger}.values())
    with (AUDIT/'results_ledger.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    assert abs(np.median([t['seconds'] for t in b0['inference']['timings']])-73.6337346)<.001
    assert result['parameters']['total']==16934644
    assert hash_file(B5/'best_detailtrust_model.pt').startswith('c3e3cbeeef00')
    return len(ledger)


def build_check():
    compiler=shutil.which('pdflatex') or shutil.which('latexmk') or shutil.which('tectonic')
    status='not_run_no_local_tex_engine';pages=None;warnings=[]
    command=None
    if compiler:
        build_dir=AUDIT/'tex_build';build_dir.mkdir(exist_ok=True)
        if Path(compiler).stem=='pdflatex':command=[compiler,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build_dir),'main.tex']
        elif Path(compiler).stem=='latexmk':command=[compiler,'-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(build_dir),'main.tex']
        else:command=[compiler,'--outdir',str(build_dir),'main.tex']
        outputs=[]
        for _ in range(2 if Path(compiler).stem=='pdflatex' else 1):
            outcome=subprocess.run(command,cwd=PAPER,capture_output=True,text=True)
            outputs.append(outcome.stdout+outcome.stderr)
            if outcome.returncode:break
        (build_dir/'build_stdout.txt').write_text('\n'.join(outputs),encoding='utf-8')
        status='failed' if outcome.returncode else 'compiled'
        if not outcome.returncode:
            doc=fitz.open(build_dir/'main.pdf');pages=len(doc);doc.close()
            shutil.copy2(build_dir/'main.pdf',ROOT/'DetailTrustNet_IEEE_Paper.pdf')
        warnings=[line for line in '\n'.join(outputs).splitlines() if 'Warning' in line or 'Overfull' in line or 'Underfull' in line]
    static=json.loads((ROOT/'verification/paper_layout_revision_checks.json').read_text())
    report=dict(date='2026-10-14',status=status,compiler=compiler,compile_command=command,page_count=pages,
                compile_warnings=warnings,static_checks=static)
    (AUDIT/'build_report.json').write_text(json.dumps(report,indent=2))
    lines=['DetailTrust-Net IEEE paper build check -- 2026-10-14','',
       'Manuscript: main.tex; class: IEEEtran conference; default 10-point conference body.',
       'Actual compilation status: '+status, 'Compiler: '+str(compiler),'Actual command executed: '+str(command),
       'Compiled page count: '+str(pages),
       'Suggested Overleaf settings: main.tex / pdfLaTeX / Normal / Recompile from scratch.',
       'Local command if pdfLaTeX is installed: pdflatex -interaction=nonstopmode -halt-on-error main.tex (twice).','',
       'Passed: balanced environments/braces/math; unique/resolved labels; all citations have bibliography entries;',
       'IEEE references ordered by first citation; all referenced figure paths exist; ASCII-safe LaTeX;',
       'no body-font/margin/page-count overrides; no distributed manuscript includes; no Python files in upload folder;',
       'Draw.io XML parses with valid unique IDs/connectors; PDF/PNG assets open;',
       'Detailed 2x2 architecture has five encoder/four decoder levels; box text widths checked;',
       'Section float barriers removed; ordinary [!t] floats; natural column bottoms via raggedbottom;',
       'Redundant Discussion opening subsection merged into its parent section;',
       'figure PDF text lies inside page bounds; embedded non-Type-3 fonts;',
       '49-patient pairing, point estimates, and 20,000-draw bootstrap independently recomputed;',
       'one baseline qualitative volume reproduced and checked against saved full-volume Dice.','',
       'Unverified without TeX: final page layout, TeX overfull/underfull boxes, rendered equation/table widths,',
       'compiled reference numbering, and final page count. Static checks do not replace compiling.',
       'No local TeX download succeeded; the release endpoint was inaccessible in this environment.',
       'Compact revision: seven figures, two tables; target 12-14 pages, actual count unverified.',
       'No fabricated result image or pending-result table remains. Test/controlled-ablation results are unmeasured.']
    if compiler:lines+=['','Compiler warnings:']+warnings
    (PAPER/'build_check.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    subprocess.run(['python',str(ROOT/'tools/paper/check_manuscript.py')],check=True,capture_output=True)
    text=(PAPER/'main.tex').read_text()
    cites=list(dict.fromkeys(k for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')))
    assert cites==re.findall(r'\\bibitem\{([^}]+)\}',text),'References not ordered by first citation'
    assert not re.search('placeholder|training_curves_partial',text,re.I)
    figures=audit_figures();ledger_count=complete_numerical_ledger();build=build_check()
    (PAPER/'README.md').write_text(README,encoding='utf-8')
    allowed={'.tex','.md','.txt','.pdf','.png','.drawio'}
    files=sorted((p for p in PAPER.rglob('*') if p.is_file()),
                 key=lambda p:(p!=PAPER/'main.tex',p.as_posix()))
    assert all(p.suffix in allowed for p in files),[str(p) for p in files if p.suffix not in allowed]
    assert len(list(PAPER.rglob('*.tex')))==1
    assert not any('placeholder' in p.name for p in files)
    output=ROOT/'DetailTrustNet_IEEE_Paper_Overleaf.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert 'main.tex' in archive.namelist() and len(archive.namelist())==len(files)
    (AUDIT/'package_audit.json').write_text(json.dumps(dict(zip_path=str(output),zip_sha256=hash_file(output),
        files=[str(p.relative_to(PAPER)) for p in files],file_count=len(files),figure_count=len(figures),
        numerical_ledger_rows=ledger_count,zip_crc_verified=True,main_tex_at_zip_root=True,
        python_or_data_or_checkpoints_in_package=False,compilation_status=build['status']),indent=2))
    print('Overleaf ZIP saved:',output)
    print('Files:',len(files),'Figures:',len(figures),'Numerical ledger:',ledger_count,'Build:',build['status'])


if __name__=='__main__':main()
