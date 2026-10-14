"""Combine measured panels at readable IEEE widths, without rescaling data."""
from pathlib import Path
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.paper.finalize_figures import B5,B0,AUDIT,FIG,REGIONS,CLASS_COLORS,segmentation_overlay

OUT=ROOT/'verification/paper_condensed'
OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((AUDIT/'generated_figures.json').read_text())
plt.rcParams.update({'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
    'pdf.fonttype':42,'ps.fonttype':42})


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def save(fig,name,source,meaning):
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.png',dpi=300)
    manifest[name]=dict(source_artifacts=source,meaning=meaning,
        figure_size_inches=list(fig.get_size_inches()),fabricated_values=False)
    plt.close(fig)
    print('Combined measured panels:',name,flush=True)


def comparison_and_training():
    summary=rows(B0/'baseline_vs_b5_summary.csv');paired=rows(B0/'baseline_vs_b5_per_patient.csv')
    fig=plt.figure(figsize=(7,4.6),layout='constrained')
    grid=fig.add_gridspec(2,1,height_ratios=(1.55,1))
    top=grid[0].subgridspec(1,2);bottom=grid[1].subgridspec(1,3)
    left,right=[fig.add_subplot(top[0,i]) for i in range(2)]
    for field,mark,color,label in [('B0_dice','s','#0072B2','B0 baseline'),('B5_dice','o','#D55E00','DetailTrust-Net')]:
        left.plot(range(4),[float(r[field]) for r in summary],mark,color=color,label=label,ms=5)
    left.set(xticks=range(4),xticklabels=['WT','TC','ET','Mean'],ylim=(0,1),
        ylabel='Patient macro Dice',title='(a) Clean validation')
    left.legend(loc='lower left',frameon=False)
    patients=sorted({r['subject'] for r in paired})
    by_patient={(r['subject'],r['region']):100*float(r['B5_minus_B0_dice']) for r in paired}
    for i,region in enumerate(REGIONS+('mean',)):
        values=[np.mean([by_patient[p,r] for r in REGIONS]) if i==3 else by_patient[p,region] for p in patients]
        right.scatter(i+np.linspace(-.12,.12,49),values,s=7,alpha=.5,color='#0072B2',rasterized=True)
        r=summary[i];mean=100*float(r['B5_minus_B0_dice'])
        lo,hi=[100*float(r['pointwise_bootstrap_ci95_'+s]) for s in ('low','high')]
        right.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',color='#D55E00',capsize=3,ms=4)
    right.axhline(0,color='black',lw=.6)
    right.set(xticks=range(4),xticklabels=['WT','TC','ET','Mean'],ylabel='B5 minus B0 Dice (pp)',title='(b) Paired differences')
    history=json.loads((B5/'training_history.json').read_text())
    for i,stage in enumerate('ABC'):
        ax=fig.add_subplot(bottom[0,i]);entries=[r for r in history if r['stage']==stage and 'validation_dice' in r]
        assert entries
        ax.plot([r['update'] for r in entries],[r['validation_dice'] for r in entries],'o-',color='#0072B2',lw=1,ms=3)
        ax.set(ylim=(0,1),xlabel='Stage update',title=f'({chr(99+i)}) Stage {stage}')
        if i==0:ax.set_ylabel('Selection Dice')
        ax.grid(alpha=.2)
    for ax in (left,right):ax.grid(axis='y',alpha=.2)
    save(fig,'comparison_training',['baseline_vs_b5_summary.csv','baseline_vs_b5_per_patient.csv','B5 training_history.json'],
        '49 matched validation patients; all paired outliers retained; means and pointwise 20,000-draw percentile intervals; separate completed-stage trajectories; unchanged [0,1] Dice axes.')


def mri_and_predictions():
    source=dict(np.load(AUDIT/'qualitative_source.npz'))
    fig,axes=plt.subplots(2,5,figsize=(7,3.5),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.12,1,.85))
    for i,name in enumerate(('T1','T1ce','T2','FLAIR')):
        axes[0,i].imshow(np.ma.masked_equal(source[f'Axial_{name}'],0),
            cmap=plt.get_cmap('gray').with_extremes(bad='black'),origin='lower',vmin=-1,vmax=3)
        axes[0,i].set_axis_off();axes[0,i].set_title(name)
    image=source['Axial_FLAIR'];truth=source['Axial_Reference']
    segmentation_overlay(axes[0,4],image,truth);axes[0,4].set_title('Reference')
    axes[1,0].imshow(np.ma.masked_equal(image,0),cmap=plt.get_cmap('gray').with_extremes(bad='black'),origin='lower',vmin=-1,vmax=3)
    axes[1,0].set_axis_off();axes[1,0].set_title('FLAIR')
    for i,(key,label) in enumerate((('Axial_B0','B0 baseline'),('Axial_Base','B5 base'),('Axial_Final','B5 final')),1):
        segmentation_overlay(axes[1,i],image,source[key]);axes[1,i].set_title(label)
    old,new=source['Axial_B0']!=truth,source['Axial_Final']!=truth
    error=np.zeros(truth.shape,np.uint8)
    error[old&~new]=1;error[~old&new]=2;error[old&new]=3
    axes[1,4].imshow(np.ma.masked_equal(image,0),cmap=plt.get_cmap('gray').with_extremes(bad='black'),origin='lower',vmin=-1,vmax=3)
    axes[1,4].imshow(np.ma.masked_equal(error,0),cmap=ListedColormap(['black','#0072B2','#D55E00','#999999']),
        origin='lower',vmin=0,vmax=3,alpha=.85,interpolation='nearest')
    axes[1,4].set_axis_off();axes[1,4].set_title('Error change')
    legend=[Patch(color=CLASS_COLORS[i],label=s) for i,s in enumerate(('Background','NCR/NET','Oedema','ET')) if i]
    legend += [Patch(color=c,label=s) for c,s in (('#0072B2','B0 error only'),('#D55E00','B5 error only'),('#999999','Both wrong'))]
    fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False)
    save(fig,'mri_predictions',['qualitative_source.npz','qualitative_provenance.json','B0 best.pt','B5 display_source.npz'],
        'Actual prespecified validation subject 205; axial RAS 90 selected from median reference tumour coordinates, independent of scores. Top: MRI and reference. Bottom: independent B0, B5 base, B5 final and error differences. Original complete-volume metrics are unchanged; orthogonal previews retained outside the compact upload package.')


def missing_and_corruption():
    records=rows(B5/'metrics_per_subject.csv')
    lookup={(r['subject'],r['condition'],r['region']):float(r['dice']) for r in records if r['region'] in REGIONS}
    patients=sorted({r['subject'] for r in records})
    values=np.array([[np.mean([lookup[p,f'clean_m{i:02d}',region] for p in patients]) for i in range(1,16)] for region in REGIONS])
    labels=['+'.join(n for i,n in enumerate(('T1','C','T2','F')) if mask&(1<<i)) for mask in range(1,16)]
    fig=plt.figure(figsize=(7,4.9),layout='constrained')
    grid=fig.add_gridspec(2,1,height_ratios=(1,1.25))
    ax=fig.add_subplot(grid[0]);mesh=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set(xticks=range(15),xticklabels=labels,yticks=range(3),yticklabels=REGIONS,title='(a) B5 modality subsets')
    plt.setp(ax.get_xticklabels(),rotation=55,ha='right',rotation_mode='anchor')
    for i in range(3):
        for j in range(15):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if values[i,j]<.58 else 'black')
    fig.colorbar(mesh,ax=ax,label='Macro Dice',shrink=.9,pad=.01)
    bottom=grid[1].subgridspec(1,3)
    corruptions=sorted({r['condition'] for r in records if r['condition'].startswith(('noise','resolution'))})
    assert len(corruptions)==8
    ticks=[('N' if c.startswith('noise') else 'R')+c.split('_')[0][-1]+('/3' if '_m13_' in c else '/4') for c in corruptions]
    for j,region in enumerate(REGIONS):
        ax=fig.add_subplot(bottom[0,j])
        for i,c in enumerate(corruptions):
            clean='clean_m13' if '_m13_' in c else 'clean_m15'
            delta=np.array([100*(lookup[p,c,region]-lookup[p,clean,region]) for p in patients])
            ax.scatter(i+np.linspace(-.12,.12,49),delta,s=6,alpha=.5,color='#0072B2',rasterized=True)
            ax.plot(i,delta.mean(),'D',color='#D55E00',ms=4)
        ax.axhline(0,color='black',lw=.6);ax.grid(axis='y',alpha=.2)
        ax.set(xticks=range(8),xticklabels=ticks,title=f'({chr(98+j)}) {region} corruption')
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right',rotation_mode='anchor')
        if j==0:ax.set_ylabel('Damaged minus clean\nDice (pp)')
    save(fig,'robustness_combined',['B5 metrics_per_subject.csv'],
        'All 15 modality subsets and 8 deterministic local FLAIR damage conditions; all 49 patients per condition. Corruption dots retain all paired differences and full observed ranges; diamond means; panels have different axes. No baseline corruption/subset result is inferred.')


def main():
    comparison_and_training();mri_and_predictions();missing_and_corruption()
    keep={'comparison_training','mri_predictions','robustness_combined','internal_corrections','gate_contribution_maps'}
    reduced={k:v for k,v in manifest.items() if k in keep}
    assert len(reduced)==5
    (AUDIT/'generated_figures.json').write_text(json.dumps(reduced,indent=2))
    (OUT/'combined_figures.json').write_text(json.dumps(reduced,indent=2))


if __name__=='__main__':main()
