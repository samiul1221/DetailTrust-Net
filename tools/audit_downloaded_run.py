"""Read-only audit of every JSON, CSV metric row and figure in a downloaded run."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dump', type=Path, default=ROOT / 'output_dump' / 'samseikhnsut--detailtrust-net-reproducible-brats-experiments--20261014T004555Z')
parser.add_argument('--output', type=Path, default=ROOT / 'verification' / 'run_20261014_audit')
arguments = parser.parse_args()
DUMP = arguments.dump.resolve()
DATA = DUMP / 'detailtrust'
OUT = arguments.output.resolve()
OUT.mkdir(exist_ok=True, parents=True)


def reject_constant(value):
    raise ValueError('Non-JSON numeric constant: ' + value)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject_constant)


def json_inventory():
    result = []
    for path in sorted(DUMP.rglob('*.json')):
        record = {'path': str(path.relative_to(DUMP)), 'bytes': path.stat().st_size}
        try:
            obj = read_json(path)
            record.update(status='parsed', type=type(obj).__name__,
                          top_keys=list(obj) if isinstance(obj, dict) else None,
                          length=len(obj) if isinstance(obj, (dict, list)) else None)
        except Exception as error:
            record.update(status='failed', error=str(error))
        result.append(record)
    return result


def csv_inventory():
    result, tables = [], {}
    for path in sorted(DUMP.rglob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            names = reader.fieldnames
        tables[path.name] = rows
        record = {'path': str(path.relative_to(DUMP)), 'rows': len(rows), 'fields': names}
        for row in rows:
            if None in row:
                raise ValueError('Malformed CSV row in ' + str(path))
        result.append(record)
    return result, tables


def figure_inventory():
    result = []
    for path in sorted((DATA/'figures').iterdir()):
        if path.suffix not in ('.png', '.pdf', '.svg', '.npz'):
            continue
        record = {'path': str(path.relative_to(DUMP)), 'bytes': path.stat().st_size}
        try:
            if path.suffix == '.png':
                with Image.open(path) as image:
                    image.load()
                    record.update(size=list(image.size), mode=image.mode, dpi=image.info.get('dpi'),
                                  extrema=image.getextrema())
            elif path.suffix == '.svg':
                element = ET.parse(path).getroot()
                record.update(width=element.get('width'), height=element.get('height'), viewBox=element.get('viewBox'))
            elif path.suffix == '.pdf':
                assert path.read_bytes().startswith(b'%PDF-')
                completed = subprocess.run(['C:/popler/poppler-24.08.0/Library/bin/pdfinfo.exe', str(path)],
                                           capture_output=True, text=True, check=True)
                record.update(pdfinfo=completed.stdout)
            else:
                with np.load(path, allow_pickle=False) as arrays:
                    record.update(arrays={key: {'shape': list(arrays[key].shape),
                                              'finite': bool(np.isfinite(arrays[key]).all())} for key in arrays.files})
            record['status'] = 'read'
        except Exception as error:
            record.update(status='failed', error=str(error))
        result.append(record)
    return result


def number(value):
    return None if value in ('', None) else float(value)


def metric_audit(report, tables):
    rows = tables['metrics_per_subject.csv']
    summaries = report['validation_metrics']
    bounded = ('dice','iou','precision','recall','specificity','surface_dice','lesion_precision','lesion_recall')
    issues = []
    keys = [(r['experiment'],r['split'],r['condition'],r['subject'],r['region']) for r in rows]
    if len(keys) != len(set(keys)):
        issues.append('Duplicate patient-condition-region records')
    manifest = read_json(DATA/'split_manifest.json')
    groups = defaultdict(list)
    for row in rows:
        if row['subject'] not in manifest[row['split']]:
            issues.append('Incorrect split: ' + str(row))
        for metric in bounded:
            v=number(row.get(metric))
            if v is not None and (not math.isfinite(v) or not 0 <= v <= 1):
                issues.append('Invalid bounded metric: '+metric)
        v=number(row.get('hd95_mm'))
        if v is not None and (not math.isfinite(v) or v < 0):
            issues.append('Invalid HD95')
        groups[(row['experiment'],row['split'],row['condition'],row['region'])].append(row)
    max_error=0.
    compared=0
    statistics_compared=0
    max_statistics_error=0.
    metrics=bounded+('hd95_mm','fp_mm3','absolute_volume_error_mm3')
    for summary in summaries:
        key=tuple(summary[k] for k in ('experiment','split','condition','region'))
        group=groups[key]
        if summary['subject_count'] != len(group): issues.append('Subject count mismatch: '+str(key))
        for metric in metrics:
            values=np.asarray([number(r.get(metric)) for r in group if number(r.get(metric)) is not None])
            saved=summary.get(metric)
            if not isinstance(saved,dict): continue
            if saved['valid_count'] != len(values): issues.append('Valid count mismatch: '+str(key)+metric)
            if len(values) and saved['mean'] is not None:
                error=abs(float(values.mean())-saved['mean'])
                max_error=max(max_error,error); compared+=1
                if error > 1e-8: issues.append('Mean mismatch: '+str(key)+metric)
            statistics = dict(median=float(np.median(values)) if len(values) else None,
                              std=float(values.std(ddof=1)) if len(values)>1 else None,
                              q05=float(np.quantile(values,.05)) if len(values) else None,
                              q95=float(np.quantile(values,.95)) if len(values) else None)
            for statistic, expected in statistics.items():
                actual = saved.get(statistic)
                statistics_compared += 1
                if expected is None:
                    if actual is not None: issues.append('Undefined statistic mismatch: '+str(key)+metric+statistic)
                elif actual is None:
                    issues.append('Missing statistic: '+str(key)+metric+statistic)
                else:
                    error=abs(expected-actual)
                    max_statistics_error=max(max_statistics_error,error)
                    if error>1e-8: issues.append('Statistic mismatch: '+str(key)+metric+statistic)
            if saved['undefined_count'] != len(group)-len(values):
                issues.append('Undefined count mismatch: '+str(key)+metric)
    clean=[r for r in rows if r['condition']=='clean_m15' and r['region'] in ('WT','TC','ET')]
    metrics_csv=tables['metrics_summary.csv']
    return {'per_subject_rows':len(rows),'report_groups':len(summaries),'summary_csv_rows':len(metrics_csv),
            'recomputed_means':compared,'max_mean_discrepancy':max_error,
            'recomputed_other_statistics':statistics_compared,
            'max_other_statistics_discrepancy':max_statistics_error,'issues':issues,
            'partitions':dict(Counter(r['split'] for r in rows)),
            'conditions':dict(Counter(r['condition'] for r in rows)),
            'empty_ET_clean':[{'subject':r['subject'],'truth_voxels':int(r['truth_voxels']),
                               'prediction_voxels':int(r['prediction_voxels']),'dice':number(r['dice'])}
                              for r in clean if int(r['truth_voxels'])==0 or int(r['prediction_voxels'])==0],
            'worst_ET_clean':sorted([{'subject':r['subject'],'dice':number(r['dice']),
                                     'truth_voxels':int(r['truth_voxels'])} for r in clean if r['region']=='ET'],
                                    key=lambda r:r['dice'])[:8]}


def cross_check_exports(report, tables):
    """Compare every CSV field with its corresponding JSON source record."""
    sources={'metrics_per_subject.csv':report['per_subject_metrics'],
             'metrics_summary.csv':report['validation_metrics'],
             'gate_analysis.csv':report['utility_gate_analysis'],
             'correction_contributions.csv':report['detail_branch_contributions']}
    results={}
    for name, expected in sources.items():
        actual=tables[name]
        assert len(actual)==len(expected),(name,len(actual),len(expected))
        checked=0
        for a,b in zip(actual,expected):
            for key,value in b.items():
                cell=a.get(key)
                if value is None: good=cell in ('',None)
                elif isinstance(value,bool): good=cell==str(value)
                elif isinstance(value,(float,int)): good=math.isclose(float(cell),value,rel_tol=1e-12,abs_tol=1e-12)
                elif isinstance(value,(dict,list)): good=json.loads(cell)==value
                else: good=cell==value
                assert good,(name,key,cell,value)
                checked+=1
        results[name]={'rows_matched':len(actual),'fields_matched':checked}
    return results


def training_audit(record):
    output=[]
    for stage in record['stages']:
        history=stage['history']
        updates=[r['update'] for r in history]
        output.append({k:stage.get(k) for k in ('stage','status','updates_completed','updates_planned',
                      'selected_update','validation_score','training_seconds')})
        output[-1].update(history_rows=len(history), contiguous_updates=updates==list(range(1,stage['updates_completed']+1)),
                          nonfinite_loss_rows=sum(not math.isfinite(r['total']) for r in history),
                          first_100_mean_loss=float(np.mean([r['total'] for r in history[:100]])),
                          last_100_mean_loss=float(np.mean([r['total'] for r in history[-100:]])))
    return output


def gate_summary(report):
    groups=defaultdict(list)
    for row in report['utility_gate_analysis']:
        if row['condition']=='clean_m15': groups[(row['modality'],row['region'])].append(row)
    results=[]
    for (modality,region),group in groups.items():
        summary={'modality':modality,'region':region,'patients':len(group)}
        for metric in ('mean','median','measured_gated_utility_mean','gate_utility_spearman','candidate_utility_spearman'):
            values=[r[metric] for r in group if r.get(metric) is not None]
            summary[metric+'_patient_mean']=float(np.mean(values)) if values else None
            summary[metric+'_valid_patients']=len(values)
        results.append(summary)
    return results


def contact_sheets():
    figures=sorted((DATA/'figures').glob('*.png'))
    files=[]
    for start in range(0,len(figures),4):
        subset=figures[start:start+4]
        sheet=Image.new('RGB',(2600,1700),'white')
        draw=ImageDraw.Draw(sheet)
        for index,path in enumerate(subset):
            x=(index%2)*1300; y=(index//2)*850
            draw.text((x+18,y+12),path.name,fill='black')
            with Image.open(path) as image:
                thumbnail=ImageOps.contain(image.convert('RGB'),(1270,810))
                sheet.paste(thumbnail,(x+(1300-thumbnail.width)//2,y+38+(810-thumbnail.height)//2))
        target=OUT/f'figures_{start//4+1}.png'
        sheet.save(target)
        files.append(str(target))
    return files


if __name__=='__main__':
    report=read_json(DATA/'detailtrust_report.json')
    record=read_json(DATA/'experiments/B5_s1729/result.json')
    inventory,tables=csv_inventory()
    checkpoint=DATA/'best_detailtrust_model.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit={'json':json_inventory(),'csv':inventory,'figures':figure_inventory(),
           'metrics':metric_audit(report,tables),'csv_json_cross_checks':cross_check_exports(report,tables),
           'training':training_audit(record),
           'gates':gate_summary(report),'checkpoint_hash':digest,
           'checkpoint_hash_matches':digest==report['best_checkpoint']['sha256'],
           'experiment_status':record['status'],'test':record['test'],
           'conditions':record['validation']['conditions'],'contact_sheets':contact_sheets()}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    compact={'json_files':len(audit['json']),'json_failures':[v for v in audit['json'] if v['status']=='failed'],
             'csv':[(v['path'],v['rows']) for v in inventory],
             'figure_failures':[v for v in audit['figures'] if v['status']=='failed'],
             'metrics':audit['metrics'],'training':audit['training'],'gates':audit['gates'],
             'checkpoint_hash_matches':audit['checkpoint_hash_matches'],'contact_sheets':audit['contact_sheets']}
    print(json.dumps(compact,indent=2))
