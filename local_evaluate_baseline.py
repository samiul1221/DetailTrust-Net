"""Evaluate the recovered B0 checkpoint on CPU; download validation inputs only.

Default: all 49 clean validation patients, no training and no test-set access.
Results resume per patient. --robustness includes the saved 23-condition suite.
The B0 figure label is concise; training/checkpoint details remain in the report.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
B5_DUMP=ROOT/'output_dump/samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T062437Z/detailtrust'


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def compare_patient_rows(baseline_rows,b5_rows):
    """Join identical patient/condition/region observations before any averaging."""
    import numpy as np
    eligible=[r for r in b5_rows if r['split']=='validation' and r['region'] in ('WT','TC','ET')
              and r['experiment']=='B5_s1729']
    lookup={(r['subject'],r['condition'],r['region']):r for r in eligible}
    assert len(lookup)==len(eligible),'Duplicate B5 comparison rows'
    paired=[]
    for row in baseline_rows:
        key=(row['subject'],row['condition'],row['region'])
        if row['region'] in ('WT','TC','ET') and key in lookup:
            reference=lookup[key]
            paired.append(dict(subject=key[0],condition=key[1],region=key[2],
                               B0_dice=row['dice'],B5_dice=reference['dice'],
                               B5_minus_B0_dice=reference['dice']-row['dice']))
    summaries=[]
    for condition in sorted({r['condition'] for r in paired}):
        group=[r for r in paired if r['condition']==condition]
        patients=sorted({r['subject'] for r in group})
        assert len(group)==3*len(patients)
        matrix={(r['subject'],r['region']):r for r in group}
        a=np.asarray([[matrix[p,r]['B0_dice'] for r in ('WT','TC','ET')] for p in patients])
        b=np.asarray([[matrix[p,r]['B5_dice'] for r in ('WT','TC','ET')] for p in patients])
        a=np.column_stack((a,a.mean(1)));b=np.column_stack((b,b.mean(1)))
        delta=b-a
        indices=np.random.default_rng(20261014).integers(0,len(patients),size=(20000,len(patients)))
        boot=delta[indices].mean(1)
        for i,region in enumerate(('WT','TC','ET','mean_WT_TC_ET')):
            low,high=np.quantile(boot[:,i],[.025,.975]) if len(patients)>1 else (None,None)
            summaries.append(dict(condition=condition,region=region,patients=len(patients),
                B0_dice=float(a[:,i].mean()),B5_dice=float(b[:,i].mean()),
                B5_minus_B0_dice=float(delta[:,i].mean()),
                pointwise_bootstrap_ci95_low=None if low is None else float(low),
                pointwise_bootstrap_ci95_high=None if high is None else float(high),
                bootstrap_replicates=20000,bootstrap_seed=20261014,
                unit='patient; same subjects and conditions; fixed selected weights'))
    return paired,summaries


def comparison_figure(paired,summary,directory):
    """Measured Dice only; model legends use B0 baseline and DetailTrust-Net."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in summary if r['condition']=='clean_m15']
    if not selected:return
    fig,axes=plt.subplots(1,2,figsize=(7,3.8),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.13,1,.87))
    x=np.arange(4)
    axes[0].plot(x,[r['B0_dice'] for r in selected],'s',label='B0 baseline',color='#0072B2')
    axes[0].plot(x,[r['B5_dice'] for r in selected],'o',label='DetailTrust-Net',color='#D55E00')
    axes[0].set(xticks=x,xticklabels=['WT','TC','ET','Mean'],ylim=(0,1),ylabel='Validation patient macro Dice')
    axes[0].legend(fontsize=8,loc='lower left')
    clean=[r for r in paired if r['condition']=='clean_m15']
    for i,region in enumerate(('WT','TC','ET')):
        values=[r['B5_minus_B0_dice']*100 for r in clean if r['region']==region]
        axes[1].scatter(i+np.linspace(-.12,.12,len(values)),values,s=8,color='#0072B2',alpha=.65)
    for i,s in enumerate(selected):
        mean=s['B5_minus_B0_dice']*100
        low,high=s['pointwise_bootstrap_ci95_low'],s['pointwise_bootstrap_ci95_high']
        if low is not None:
            axes[1].errorbar(i,mean,yerr=[[mean-low*100],[high*100-mean]],fmt='D',color='#D55E00',capsize=3)
        else:axes[1].plot(i,mean,'D',color='#D55E00')
    axes[1].axhline(0,color='black',lw=.7)
    axes[1].set(xticks=x,xticklabels=['WT','TC','ET','Mean'],ylabel='DetailTrust-Net minus B0\nDice change (percentage points)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'Clean validation comparison: {selected[0]["patients"]} matched patients',fontsize=10)
    fig.text(.5,.025,'Dots: patient differences; diamonds: means; bars: pointwise 95% paired bootstrap intervals.\nTraining budgets and inference precision are documented in local_baseline_report.json.',ha='center',fontsize=7)
    for suffix in ('png','pdf','svg'):fig.savefig(directory/f'baseline_vs_b5.{suffix}',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'validation_data',help='Local BraTS20 NIfTI folder; missing files are downloaded unless --no-download.')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'output_dump/B0_s1729/RAW/best.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'local_baseline_results')
    parser.add_argument('--subjects',type=int,help='Limit for first-patient timing; full default is 49. Subset results are explicitly recorded.')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--robustness',action='store_true',help='Evaluate all 15 modality subsets plus eight validation corruption conditions.')
    parser.add_argument('--no-download',action='store_true',help='Use existing local NIfTI files without Kaggle requests.')
    args=parser.parse_args()
    if args.subjects is not None and not 1<=args.subjects<=49:parser.error('--subjects must be 1..49')
    if not 1<=args.threads<=max(1,os.cpu_count() or 1):parser.error('Invalid CPU thread count')
    if not args.checkpoint.is_file():parser.error('Baseline checkpoint not found')
    # Dependency imports happen after --help and argument validation.
    import numpy as np
    import torch
    from detailtrust.config import RunConfig,write_json,environment_record
    from detailtrust.data import discover_subjects,load_subject,prepare_subject,assert_split
    from detailtrust.model import DetailTrustNet,parameter_counts
    from detailtrust.evaluation import sliding_window,apply_condition,evaluation_conditions
    from detailtrust.metrics import subject_metrics,aggregate_metrics
    from detailtrust.reporting import write_csv
    from tools.download_validation_data import download_validation_data
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=True)
    started=time.perf_counter();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    print('CPU evaluation only: no optimizer, backward pass, training updates, or test subjects.',flush=True)
    digest=file_hash(args.checkpoint)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state.get('stage')!='RAW' or state['architecture']['gate']!='raw':raise ValueError('Expected a genuine RAW baseline checkpoint')
    if state['variant']['corruption'] is not False:raise ValueError('B1 and B0 differ in corruption augmentation; this runner labels only B0.')
    split=state['split_manifest'];assert_split(split)
    b5=json.loads((B5_DUMP/'detailtrust_report.json').read_text())
    if split!=b5['data_splits']:raise ValueError('B0/B5 patient split mismatch')
    cfg=replace(RunConfig(**state['config']),precision='float32',variants=('B0',),
                output_dir=str(output),run_final_test=False,protocol_locked=False)
    subjects=split['validation'][:args.subjects] if args.subjects else split['validation']
    assert not(set(subjects)&set(split['train']) or set(subjects)&set(split['test']))
    checkpoint_info=dict(path=str(args.checkpoint.resolve()),sha256=digest,selected_update=state['update'],
        selected_validation_score=state['validation_score'],selection_criterion=state['selection_criterion'],
        seed=state['seed'],architecture=state['architecture'],variant=state['variant'],
        original_checkpoint_configuration=state['config'],planned_training_updates=cfg.updates_a+cfg.updates_b+cfg.updates_c,
        source='user-supplied selected checkpoint; no further training performed')
    model=DetailTrustNet(**state['architecture']);model.load_state_dict(state['model_state'],strict=True);model.eval()
    checkpoint_info['strict_state_dict_load']='passed'
    del state
    model.requires_grad_(False)
    if not args.no_download:download_validation_data(args.data_root,split['validation'],ROOT)
    records,discovery=discover_subjects(args.data_root)
    mapping={r['id']:r for r in records}
    if not set(subjects)<=set(mapping):raise ValueError('Required validation MRIs or masks are missing')
    expected={r['id']:r for r in b5['dataset_integrity'] if r.get('status')=='passed'}
    verified=set();integrity=[]
    class ValidationStore:
        def load(self,subject):
            assert subject in subjects,'Only the selected validation patients are permitted'
            images,labels,metadata=load_subject(mapping[subject])
            if subject not in verified:
                raw_hash=hashlib.sha256(images.tobytes()).hexdigest()
                reference=expected[subject]
                if raw_hash!=reference['image_sha256']:raise ValueError(f'{subject}: MRI hash differs from the B5 input')
                et_volume=float((labels==3).sum()*np.prod(metadata['spacing']))
                if et_volume!=reference['et_volume_mm3']:raise ValueError(f'{subject}: reference ET volume differs from B5')
                integrity.append(dict(subject=subject,canonical_mri_sha256=raw_hash,mri_matches_b5=True,
                    et_volume_matches_b5=True,label_mapping='0,1,2,4 -> 0,1,2,3'))
                verified.add(subject)
            return prepare_subject(images,labels,metadata)
    store=ValidationStore();device=torch.device('cpu')
    suite=evaluation_conditions(cfg) if args.robustness else [dict(id='clean_m15',available=[1,1,1,1],corruption=None)]
    module_hashes={n:file_hash(ROOT/'detailtrust'/f'{n}.py') for n in ('model','data','evaluation','metrics')}
    b5_report_hash=file_hash(B5_DUMP/'detailtrust_report.json')
    rows=[];timings=[];completed=[]
    for condition in suite:
        condition_dir=output/'patients'/condition['id'];condition_dir.mkdir(parents=True,exist_ok=True)
        for i,subject in enumerate(subjects,1):
            path=condition_dir/f'{subject}.json'
            identity=dict(checkpoint_sha256=digest,b5_report_sha256=b5_report_hash,
                subject=subject,condition=condition,split_sha256=split['sha256'],patch=list(cfg.patch),
                precision='float32',surface_tolerance_mm=cfg.surface_tolerance_mm,module_sha256=module_hashes)
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['identity']!=identity:raise ValueError('Saved evaluation identity differs; choose a new output folder.')
                patient_rows=saved['rows'];timing=saved['timing']
                if subject not in verified:
                    integrity.append(saved['input_integrity']);verified.add(subject)
                print(f'Reusing {condition["id"]} patient {i}/{len(subjects)}: {subject}',flush=True)
            else:
                print(f'Evaluating {condition["id"]} patient {i}/{len(subjects)} on CPU: {subject}',flush=True)
                before=time.perf_counter()
                with torch.inference_mode():
                    images,truth,metadata=store.load(subject)
                    damaged,flags=apply_condition(images,subject,condition,cfg.split_seed)
                    result=sliding_window(model,damaged,flags,cfg,device,stage='A',analysis=False)
                    prediction=result['logits'].argmax(0).numpy()
                    patient_rows=subject_metrics(prediction,truth,metadata['spacing'],subject,'B0_s1729',condition['id'],'validation',cfg.surface_tolerance_mm)
                    timing=dict(subject=subject,condition=condition['id'],seconds_including_loading_and_metrics=time.perf_counter()-before,**result['runtime'])
                    del result,prediction,images,damaged,truth
                proof=next(r for r in integrity if r['subject']==subject)
                write_json(path,dict(identity=identity,rows=patient_rows,timing=timing,input_integrity=proof))
                print(f'Patient saved; {timing["seconds_including_loading_and_metrics"]:.1f}s including loading/metrics.',flush=True)
            rows+=patient_rows;timings.append(timing);completed.append([condition['id'],subject])
            write_json(output/'progress.json',dict(completed_patient_conditions=completed,requested_patients=subjects,
                requested_conditions=[c['id'] for c in suite],checkpoint=checkpoint_info,status='running',training_updates_executed=0))
            write_csv(output/'metrics_per_subject.csv',rows)
    summary=aggregate_metrics(rows)
    paired,comparisons=compare_patient_rows(rows,b5['per_subject_metrics'])
    write_csv(output/'metrics_summary.csv',summary)
    write_csv(output/'baseline_vs_b5_per_patient.csv',paired)
    write_csv(output/'baseline_vs_b5_summary.csv',comparisons)
    comparison_figure(paired,comparisons,output)
    report=dict(schema_version='local_baseline_evaluation_1',status='completed_selected_conditions',
        full_validation_cohort_evaluated=len(subjects)==49,model_label='B0 baseline',checkpoint=checkpoint_info,
        training_updates_executed=0,dataset_discovery=discovery,data_splits=split,evaluated_subjects=subjects,
        conditions=suite,input_integrity=integrity,model_parameters=parameter_counts(model),
        validation_metrics=summary,per_subject_metrics=rows,paired_b5_comparisons=comparisons,
        inference=dict(device='cpu',precision='float32',threads=args.threads,patch=cfg.patch,
            sliding_window='same halo, foreground crop and uniform logit overlap as Kaggle',timings=timings),
        environment=environment_record(output),wall_seconds_this_invocation=time.perf_counter()-started,
        test_metrics=dict(status='not_run',reason='test remains sealed'),
        limitations=['B0 uses a selected update-2000 checkpoint; original planned RAW budget was 10000 updates.',
            'B5 finished its configured training; this accuracy comparison does not match training budgets.',
            'B0 CPU float32 versus saved B5 GPU FP16; no hardware latency comparison is valid.',
            'Validation used for model selection; this is exploratory, not final test evidence.',
            'Single training seed; bootstrap intervals condition on selected weights, not training-seed uncertainty.',
            'Raw MRI hashes match B5; original label-array hashes were not recorded, so only label geometry/mapping/ET-volume checks are available.'])
    write_json(output/'local_baseline_report.json',report)
    json.loads((output/'local_baseline_report.json').read_text())
    write_json(output/'progress.json',dict(status='completed_selected_conditions',completed_patient_conditions=completed,training_updates_executed=0))
    print('Evaluation finished. Outputs:',output,flush=True)
    for value in comparisons:
        if value['condition']=='clean_m15':
            print(value['region'],f'B0={value["B0_dice"]:.4f}',f'B5={value["B5_dice"]:.4f}',flush=True)


if __name__=='__main__':
    main()
