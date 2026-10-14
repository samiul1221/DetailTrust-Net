"""Static manuscript/asset audit; explicitly does not replace a TeX build."""
from collections import Counter
from pathlib import Path
import json
import re
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'DetailTrustNet_IEEE_Paper'
source = (PAPER / 'main.tex').read_text(encoding='utf-8-sig')
tex = re.sub(r'(?<!\\)%[^\n]*', '', source)
checks = {}

environments = []
for kind, name in re.findall(r'\\(begin|end)\{([^}]+)\}', tex):
    if kind == 'begin':
        environments.append(name)
    else:
        assert environments and environments.pop() == name, f'Mismatched environment: {name}'
assert not environments, environments
checks['balanced_environments'] = True

depth = 0
for brace in re.findall(r'(?<!\\)[{}]', tex):
    depth += 1 if brace == '{' else -1
    assert depth >= 0, 'Unexpected closing brace'
assert depth == 0, f'Brace depth={depth}'
assert len(re.findall(r'(?<!\\)\$', tex)) % 2 == 0
checks['balanced_braces_and_inline_math'] = True

labels = re.findall(r'\\label\{([^}]+)\}', tex)
assert len(labels) == len(set(labels)), 'Duplicate labels'
references = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex)
assert set(references) <= set(labels), set(references) - set(labels)
bib = re.findall(r'\\bibitem\{([^}]+)\}', tex)
cited = {key for group in re.findall(r'\\cite\{([^}]+)\}', tex) for key in group.split(',')}
assert len(bib) == len(set(bib)) and cited <= set(bib)
checks['labels'] = len(labels)
checks['references_resolve'] = True
checks['bibliography_entries'] = len(bib)
checks['unused_bibliography_entries'] = sorted(set(bib) - cited)

assets = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', tex)
assert all((PAPER / asset).is_file() for asset in assets), assets
checks['referenced_assets'] = assets
assert r'\documentclass[conference]{IEEEtran}' in tex
assert not re.search(r'\\(?:geometry|fontsize|linespread|vspace|input|include)\b', tex)
assert not re.search(r'\\(?:tiny|scriptsize|footnotesize|small)\b', tex)
assert not list(PAPER.rglob('*.py'))
checks['standard_ieee_class_no_body_font_or_margin_overrides'] = True
checks['python_files_inside_upload_folder'] = 0
checks['ascii_tex_source'] = source.isascii()
assert 'placeins' not in tex and 'FloatBarrier' not in tex
assert r'\raggedbottom' in tex
assert 'Evidence for detail correction' not in tex
assert not re.search(r'\\(?:clearpage|newpage|pagebreak|enlargethispage)\b',tex)
assert all(options=='!t' for options in re.findall(r'\\begin\{(?:figure\*?|table\*?)\}\[([^]]+)\]',tex))
checks['section_float_barriers_removed'] = True
checks['natural_column_bottoms_without_vertical_stretch'] = True
checks['ordinary_top_float_placement'] = '!t'
checks['discussion_opening_heading_merged'] = True

checks['drawio'] = {}
for name in ('architecture', 'training_pipeline'):
    path = PAPER / 'figures' / f'{name}.drawio'
    document = ET.parse(path)
    assert document.getroot().get('compressed') == 'false'
    model = document.find('.//mxGraphModel')
    cells = model.findall('.//mxCell')
    ids = [cell.get('id') for cell in cells]
    assert len(ids) == len(set(ids)), f'Duplicate cells in {name}'
    for cell in cells:
        if cell.get('edge') == '1':
            assert cell.get('source') in ids and cell.get('target') in ids
    for suffix in ('pdf', 'png'):
        asset = path.with_suffix('.' + suffix)
        assert asset.is_file() and asset.stat().st_size > 1000
    assert path.with_suffix('.pdf').read_bytes().startswith(b'%PDF-')
    checks['drawio'][name] = {'cells': len(cells), 'valid_xml_and_connectors': True,
                             'rendered_assets_present': True}

checks['word_token_estimate_including_references'] = len(re.findall(r'\b[A-Za-z][A-Za-z-]+\b',
    re.sub(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?', '', tex)))
checks['compiler'] = next((shutil.which(name) for name in ('pdflatex', 'latexmk', 'tectonic')
                          if shutil.which(name)), None)
checks['latex_compilation_status'] = 'not_run'
checks['page_count'] = None
checks['limitations'] = ['No local TeX engine; final overfull-box/page-layout checks require Overleaf.',
    'PDF/PNG diagrams were rendered from authored Draw.io XML by the local renderer, not the Draw.io desktop export CLI.',
    'Static checks and prose editing cannot guarantee a Turnitin score.']
out = ROOT / 'verification' / 'paper_layout_revision_checks.json'
out.write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks, indent=2))
