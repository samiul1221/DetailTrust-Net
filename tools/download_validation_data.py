"""Download only the saved validation patients, with redacted Kaggle diagnostics."""
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

DATASET = 'awsaf49/brats20-dataset-training-validation'
MODALITIES = ('t1', 't1ce', 't2', 'flair', 'seg')


def parse_file_listing(output):
    lines = output.splitlines()
    token = re.search(r'Next Page Token\s*=\s*(\S+)', output)
    for i,line in enumerate(lines):
        header = next(csv.reader([line.lstrip('\ufeff')]), [])
        if 'name' in header and 'size' in header:
            rows = list(csv.DictReader(io.StringIO('\n'.join([line.lstrip('\ufeff'),*lines[i+1:]]))))
            return [r for r in rows if r.get('name') and not r['name'].startswith('Next Page Token')], token.group(1) if token else None
    raise ValueError('No Kaggle file CSV header found; refusing an unfiltered dataset download.')


def select_validation_files(rows, subjects):
    selected = {}
    for row in rows:
        name = Path(row['name'].replace('\\','/')).name
        match = re.fullmatch(r'(BraTS20_Training_\d+)_(t1ce|t1|t2|flair|seg)\.nii(?:\.gz)?', name, re.I)
        if match and match[1] in subjects:
            key = (match[1],match[2].lower())
            if key in selected:
                raise ValueError(f'Duplicate MRI file for {key}; choose an explicit dataset version.')
            selected[key] = dict(row,subject=key[0],modality=key[1],basename=name)
    expected = {(s,m) for s in subjects for m in MODALITIES}
    if set(selected)!=expected:
        raise ValueError(f'Dataset mirror lacks {len(expected-set(selected))} required validation files. No whole-dataset fallback.')
    return [selected[key] for key in sorted(selected)]


def download_validation_data(root, subjects, project_root):
    """No credentials are written to manifests, commands, or notebook files."""
    from dotenv import dotenv_values
    from sync_kaggle import run_kaggle, fail_cli
    root=Path(root).resolve();project_root=Path(project_root)
    root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'validation_download_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        if manifest['dataset']!=DATASET or manifest['subjects']!=sorted(subjects):
            raise ValueError('Existing download manifest belongs to another dataset/patient list.')
        selected=select_validation_files(manifest['files'],set(subjects))
    else:
        token=os.environ.get('KAGGLE_API_TOKEN') or dotenv_values(project_root/'.env').get('KAGGLE_API_TOKEN')
        if not token:
            raise RuntimeError('KAGGLE_API_TOKEN is absent from the existing .env/environment. No credentials were requested or embedded.')
        environment=os.environ.copy();environment['KAGGLE_API_TOKEN']=token
        rows=[];page=None;seen=set()
        print('Listing dataset files to select validation subjects only.',flush=True)
        while True:
            args=['datasets','files',DATASET,'--csv','--page-size','200']
            if page:args += ['--page-token',page]
            response=run_kaggle(args,environment)
            if response.returncode:fail_cli('validation file inventory',response,environment)
            found,page=parse_file_listing(response.stdout);rows += found
            if not page:break
            if page in seen:raise RuntimeError('Repeated Kaggle pagination token; refusing incomplete inventory.')
            seen.add(page)
        selected=select_validation_files(rows,set(subjects))
        manifest=dict(dataset=DATASET,subjects=sorted(subjects),files=selected,
                      dataset_version='not returned by CLI; MRI hashes checked against B5 before inference',
                      scope='four modalities and segmentation labels for validation patients only',completed_files=[])
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    token=None;environment=None
    for index,file in enumerate(selected,1):
        target=root/file['subject']/file['basename']
        assert target.resolve().is_relative_to(root), 'Download target escapes the configured data directory'
        size=int(file['size']) if str(file.get('size','')).isdigit() else None
        if target.is_file() and file['name'] in manifest['completed_files'] and (size is None or target.stat().st_size==size):
            continue
        if environment is None:
            token=os.environ.get('KAGGLE_API_TOKEN') or dotenv_values(project_root/'.env').get('KAGGLE_API_TOKEN')
            if not token:raise RuntimeError('A Kaggle API token is required to download missing MRI files.')
            environment=os.environ.copy();environment['KAGGLE_API_TOKEN']=token
        if size and shutil.disk_usage(root).free<size+2*1024**3:
            raise RuntimeError('Insufficient disk space for this file plus 2 GiB working reserve.')
        print(f'Downloading validation file {index}/{len(selected)}: {file["basename"]}',flush=True)
        with tempfile.TemporaryDirectory(prefix='.transfer-',dir=root) as temporary:
            response=run_kaggle(['datasets','download',DATASET,'-f',file['name'],'-p',temporary,'--quiet','--force'],environment,timeout=1800)
            if response.returncode:fail_cli('validation MRI download',response,environment)
            downloads=[p for p in Path(temporary).rglob('*') if p.is_file()]
            if len(downloads)!=1:raise RuntimeError('Expected exactly one downloaded MRI file or archive.')
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.part')
            if zipfile.is_zipfile(downloads[0]):
                with zipfile.ZipFile(downloads[0]) as archive:
                    matches=[x for x in archive.infolist() if not x.is_dir() and Path(x.filename).name==file['basename']]
                    if len(matches)!=1:raise RuntimeError('Archive does not contain the unique requested MRI; no bulk extraction attempted.')
                    with archive.open(matches[0]) as source,partial.open('wb') as destination:
                        shutil.copyfileobj(source,destination)
            else:
                shutil.copyfile(downloads[0],partial)
            if size and partial.stat().st_size!=size:
                raise RuntimeError('Downloaded NIfTI size differs from Kaggle inventory; incomplete file retained as .part.')
            partial.replace(target)
        if file['name'] not in manifest['completed_files']:manifest['completed_files'].append(file['name'])
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Validation inputs ready: {len(selected)} files for {len(subjects)} patients. No training/test MRI or cache downloaded.',flush=True)
    return manifest
