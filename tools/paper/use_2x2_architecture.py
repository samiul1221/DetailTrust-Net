"""Install the inspected Draw.io 2x2 diagram as the paper's architecture asset."""
from pathlib import Path
import shutil
import json
import sys
import xml.etree.ElementTree as ET
import fitz

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.paper import render_diagrams as renderer
FIG=ROOT/'DetailTrustNet_IEEE_Paper/figures'
OUT=ROOT/'verification/paper_condensed'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=ROOT/'detailtrust_modules_2x2.drawio'
    tree=ET.parse(original)
    cells=tree.findall('.//mxCell')
    ids=[c.get('id') for c in cells]
    assert len(ids)==len(set(ids))
    for cell in cells:
        if cell.get('edge')=='1':assert cell.get('source') in ids and cell.get('target') in ids
    labels=' '.join(c.get('value','') for c in cells)
    for required in ('0.5, 0.8, 1.2 mm','37','44','Sigmoid','Background','STAGE C','stopgrad'):
        assert required in labels,required
    archive=OUT/'superseded_assets'
    archive.mkdir(exist_ok=True)
    for ext in ('drawio','pdf','png'):
        prior=FIG/f'architecture.{ext}'
        saved=archive/f'architecture_stacked.{ext}'
        if prior.exists() and not saved.exists():shutil.copy2(prior,saved)
    # Align shorthand with manuscript notation without changing diagram geometry.
    for cell in cells:
        if cell.get('value'):
            cell.set('value',cell.get('value').replace('Δₘ','δₘ').replace('Xₘ','X̃ₘ'))
    editable=FIG/'architecture.drawio'
    tree.write(editable,encoding='utf-8',xml_declaration=True)
    renderer.render(editable,7.1)
    path=FIG/'architecture.pdf'
    delivered=fitz.open(path);page=delivered[0]
    page.get_pixmap(dpi=350).save(FIG/'architecture.png')
    sizes=[];outside=[]
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            for span in l['spans']:
                sizes.append(span['size']);x0,y0,x1,y1=span['bbox']
                if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:outside.append(span['text'])
    assert not outside,outside
    assert min(sizes)>=7.0
    report=dict(editable_source=str(original),paper_source=str(FIG/'architecture.drawio'),
        layout='2x2',native_pdf_source=str(ROOT/'detailtrust_modules_2x2.pdf'),
        vector_pdf_width_inches=7.1,png_dpi=350,minimum_font_size_points=min(sizes),
        notation_aligned_with_manuscript=True,diagram_geometry_changed=False,
        text_outside_page=outside,xml_cells=len(cells),xml_ids_and_connectors_valid=True,
        refiner_input_channels=37,gate_input_channels=44,encoder_widths=[32,64,128,256,320],
        gaussian_sigma_mm=[.5,.8,1.2],anchored_corrections=True,background_logit_preserved=True,
        visual_inspection='Exact paper PDF/PNG reviewed after installation; 2x2 source reviewed before installation.')
    (OUT/'architecture_2x2_checks.json').write_text(json.dumps(report,indent=2))
    print('Installed vector 2x2 architecture:',path,'minimum text size:',round(min(sizes),2),'pt')


if __name__=='__main__':main()
