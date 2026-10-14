"""Draw the full implemented architecture in four editable, detailed panels."""
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
import fitz
import html
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.paper import render_diagrams as rd


def main():
    d=rd.Diagram(1600,1200)
    def box(key,text,x,y,w,h,color=rd.BLUE,size=23):
        d.node(key,text,x,y,w,h,color,size)
    def note(key,text,x,y,w,h,size=22.5):
        d.node(key,text,x,y,w,h,size=size,kind='text')
    for key,title,x,y in [('a','(a) Availability and physical-scale Gaussian split',8,8),
                          ('b','(b) Full five-level residual encoder–decoder',817,8),
                          ('c','(c) Shared anchored correction and spatial gate',8,615),
                          ('d','(d) Utility supervision and additive fusion',817,615)]:
        d.node(key,title,x,y,775,570,size=27,kind='panel')
        cell=d.root.find(f"mxCell[@id='{key}']")
        cell.set('style',cell.get('style')+'fillColor=none;strokeColor=none;')
    for key,x,y,w,h in [('v',800,8,1,1180),('h',8,602,1584,1)]:
        box(key,'',x,y,w,h,'#d9dee5',1)
        cell=d.root.find(f"mxCell[@id='{key}']")
        cell.set('style',cell.get('style')+'strokeColor=none;')

    # (a) The haloed image is filtered before extracting the central target grid.
    box('mri','T1 · T1ce\nT2 · FLAIR\n4 × 104³',20,84,145,91,size=23)
    box('mask','Availability a\nX̃ₘ = aₘ Xₘ',186,84,177,91,rd.GREEN,23)
    d.edge('mri_mask','mri','mask')
    for i,(sigma,y) in enumerate(((.5,73),(.8,210),(1.2,327))):
        box(f'g{i}',f'Gaussian\nσ = {sigma} mm',391,y,157,89,rd.GOLD,23)
        d.edge(f'to_g{i}','mask',f'g{i}',(1,.5),(0,.5),via=((377,129.5),(377,y+44.5)))
    box('mix','Learned mixture\nπₘ=softmax(αₘ)\nLₘ = Σⱼ πₘⱼ Gⱼ(X̃ₘ)',573,210,201,115,rd.GOLD,22.5)
    for i,y in enumerate((73,210,327)):
        entry=.2+.3*i
        d.edge(f'g_mix{i}',f'g{i}','mix',(1,.5),(0,entry),via=((561,y+44.5),(561,210+115*entry)))
    note('spacing','1-mm voxel grid\nNormalized nine-tap kernels\nαₘ initially (−1, 1, −1)',20,230,340,100,22.5)
    box('low','L: 4 × 104³\nCrop → 4 × 96³',573,414,201,75,rd.BLUE,23)
    box('detail','Hₘ = X̃ₘ − Lₘ\n104³ → central 96³',273,421,263,92,rd.ROSE,23)
    d.edge('mix_low','mix','low',(.5,1),(.5,0))
    d.edge('raw_detail','mask','detail',(.9,1),(.35,0),via=((345.3,193),(365.05,193)))
    d.edge('low_detail','low','detail',(0,.5),(1,.3),via=((555,451.5),(555,448.6)))
    note('a_outputs','Central backbone input: L(4) + flags(4) → 8 channels\nL → panel (b); Hₘ → shared branch in panel (c)',20,541,754,57,22.5)

    # (b) Actual channel/grid sizes and skip connections, rather than one U-Net box.
    box('input','Input: 8 × 96³ (smooth images + availability maps)',829,67,730,58,size=23)
    xs=[829,975,1121,1267,1413]
    widths=[32,64,128,256,320];grids=[96,48,24,12,6]
    for i,(x,width,grid) in enumerate(zip(xs,widths,grids)):
        box(f'e{i}',f'E{i+1}\n{width} ch\n{grid}³',x,148,129,87,size=23.5)
        if i:d.edge(f'encode{i}',f'e{i-1}',f'e{i}')
    d.edge('input_e','input','e0',(.08,1),(.5,0))
    cats=[(64,32,32,96),(128,64,64,48),(256,128,128,24),(320,256,256,12)]
    for i,(x,(deep,skip,out,grid)) in enumerate(zip(xs,cats)):
        box(f'd{i}',f'{deep}+{skip}\n→ {out} ch\n{grid}³',x,300,129,96,rd.GREEN,23)
        d.edge(f'skip{i}',f'e{i}',f'd{i}',(.5,1),(.5,0),dashed=True)
        if i:d.edge(f'decode{i}',f'd{i}',f'd{i-1}',(0,.5),(1,.5))
    d.edge('deepest','e4','d3',(.5,1),(1,.5),via=((1477.5,348),))
    box('feature','Context F\n32 × 96³',829,439,260,71,rd.BLUE,23)
    box('head','1³ head: 32 → 4\nz₀: 4 × 96³',1145,439,299,71,rd.GREEN,23)
    d.edge('d_feature','d0','feature',(.5,1),(.5,0),via=((893.5,418),(959,418)))
    d.edge('feature_head','feature','head')
    note('b_block','Residual: two 3³ convs, GN8, LeakyReLU; 1³ skip\nDown: avg pool 2³ | Up: trilinear + skip concatenation',829,531,730,61,22.5)

    # (c) Both evaluations share every refiner parameter. Gates have a separate net.
    note('c_input','Refiner input: F(32), Hₘ(1), eₘ(4) → 37 channels',20,664,752,37,22.5)
    for suffix,detail,y in [('live','Hₘ',710),('zero','0',791)]:
        box(f'r_{suffix}',f'[F, {detail}, eₘ]',20,y,173,67,rd.ROSE,23)
        box(f'conv1_{suffix}','3³ conv\n37 → 16',217,y,130,67,rd.ROSE,22.5)
        box(f'conv2_{suffix}','3³ conv\n16 → 16',377,y,130,67,rd.ROSE,22.5)
        box(f'conv3_{suffix}','1³ conv\n16 → 3',537,y,117,67,rd.ROSE,22.5)
        for a,b in [(f'r_{suffix}',f'conv1_{suffix}'),(f'conv1_{suffix}',f'conv2_{suffix}'),(f'conv2_{suffix}',f'conv3_{suffix}')]:
            d.edge(a+'_to_'+b,a,b)
    box('subtract','−\nrₘ: 3 ch',677,750,95,72,rd.GREEN,22.5)
    d.edge('live_minus','conv3_live','subtract',(1,.5),(0,.22))
    d.edge('zero_minus','conv3_zero','subtract',(1,.5),(0,.78))
    note('c_anchor','Shared θ; ReLU after both 3³ convs; no norm/dropout\nHₘ = 0 ⇒ rₘ = 0; output: 3 × 96³ per modality',20,867,752,60,22.5)
    box('gate_inputs','Gate input: [F32, Hₘ1, rₘ3, eₘ4, a4] → 44 channels',20,936,752,57,rd.BLUE,22.5)
    d.edge('correction_gate','subtract','gate_inputs',(1,.5),(1,.5),via=((788,786),(788,964.5)))
    for key,label,x,w in [('gate1','1³ pointwise\n44 → 16\nReLU',20,180),
                          ('gate2','3³ depthwise\n16 → 16, g=16\nReLU',223,192),
                          ('gate3','1³ pointwise\n16 → 1\nSigmoid',441,163),
                          ('q','qₘ\n1 × 96³\n(0,1)',644,128)]:
        box(key,label,x,1020,w,90,rd.GOLD if key!='q' else rd.GREEN,22.5)
    d.edge('gate_in','gate_inputs','gate1',(.12,1),(.5,0))
    for a,b in [('gate1','gate2'),('gate2','gate3'),('gate3','q')]:d.edge(a+'_to_'+b,a,b)
    note('c_out','Shared refiner and gate weights across all four modalities\nOne independent sigmoid coefficient per modality and voxel',20,1126,752,64,22.5)

    # (d) Reference labels feed utility supervision only, never inference features.
    box('frozen','Stage C: freeze U-Net, Gaussian mixture and Rθ\nTrain gate: Lseg + 0.1 balanced utility BCE',829,667,741,66,rd.ROSE,23)
    box('targets','zall = z₀ + Σₘ (0, aₘ rₘ)\nδₘ = CE(zall − (0, aₘ rₘ), Y) − CE(zall, Y)\ntₘ = stopgrad σ(AvgPool₃(δₘ) / 0.1)',829,761,741,113,rd.ROSE,22.5)
    box('reference','Reference Y\nTraining supervision only',829,900,330,65,rd.ROSE,22.5)
    d.edge('y_target','reference','targets',(.55,0),(.25,1))
    d.edge('t_gate','targets','gate3',(0,.5),(.5,0),via=((804,817.5),(804,1004),(522.5,1004)),dashed=True)
    box('fusion','cₘ,ₖ = aₘ qₘ rₘ,ₖ  (foreground k = 1,2,3)\nzₖ = z₀,ₖ + Σₘ cₘ,ₖ ; zbg = z₀,bg\nArgmax → background / NCR/NET / oedema / ET',829,1005,741,99,rd.BLUE,22.5)
    d.edge('q_fusion','q','fusion',(1,.4),(0,.5))
    box('regions','WT={1,2,3}; TC={1,3}; ET={3}\nBackground logit unchanged; aₘ=0 ⇒ cₘ,ₖ=0',829,1131,741,62,rd.GREEN,22.5)
    d.edge('fuse_regions','fusion','regions',(.5,1),(.5,0))
    path=d.save('architecture')
    rd.render(path,7.1)
    tree=ET.parse(path);ids=[c.get('id') for c in tree.findall('.//mxCell')]
    assert len(ids)==len(set(ids))
    pdf=fitz.open(path.with_suffix('.pdf'));page=pdf[0]
    outside=[];sizes=[]
    for b in page.get_text('dict')['blocks']:
        for line in b.get('lines',[]):
            for span in line['spans']:
                sizes.append(span['size']);x0,y0,x1,y1=span['bbox']
                if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:outside.append(span['text'])
    assert not outside,outside
    assert min(sizes)>7
    # Measure every line against its own box, not just the exported page.
    overflows=[]
    for cell in tree.findall('.//mxCell'):
        if cell.get('vertex')!='1':continue
        style=dict(s.split('=',1) for s in cell.get('style','').split(';') if '=' in s)
        if style.get('shape')=='text' or style.get('verticalAlign')=='top':continue
        geometry=cell.find('mxGeometry');limit=float(geometry.get('width'))*7.1*72/1600-2
        fontsize=float(style['fontSize'])*7.1*72/1600
        for line in html.unescape(cell.get('value','')).split('<br>'):
            if not line:continue
            measured=TextPath((0,0),line,size=fontsize,prop=FontProperties(family='DejaVu Sans')).get_extents().width
            if measured>limit:overflows.append(dict(node=cell.get('id'),text=line,width=measured,available=limit))
    assert not overflows,overflows
    audit=dict(layout='Detailed 2x2',encoder_levels=5,decoder_levels=4,
        encoder_widths=widths,encoder_grids=grids,skip_connections=4,feature_channels=32,logit_channels=4,
        refiner_input_channels=37,refiner_outputs=3,gate_input_channels=44,gate_hidden_channels=16,
        zero_anchor_explicit=True,availability_fusion_explicit=True,background_preserved=True,
        shared_weights=True,utility_target_detached=True,reference_training_only=True,
        width_inches=7.1,height_inches=5.325,minimum_font_points=min(sizes),xml_cells=len(ids),
        text_outside_page=outside,text_exceeds_node_width=overflows)
    (ROOT/'verification/paper_condensed/detailed_architecture_checks.json').write_text(json.dumps(audit,indent=2))
    print('Detailed editable architecture exported:',path)


if __name__=='__main__':main()
