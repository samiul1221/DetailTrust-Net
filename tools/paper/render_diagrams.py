"""Author editable Draw.io figures, then render those exact sources locally.

No Draw.io desktop exporter is available here. This deliberately limited renderer
supports only the rectangles, text, and routed connectors authored below; it is
not presented as the Draw.io application's export command.
"""
from pathlib import Path
import html
import re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / 'DetailTrustNet_IEEE_Paper' / 'figures'
BLUE, GREEN, GOLD, ROSE = '#e8f0fa', '#e7f2ee', '#fff3d6', '#f6e9ef'
INK = '#24354b'


class Diagram:
    def __init__(self, width, height):
        self.file = ET.Element('mxfile', host='app.diagrams.net', compressed='false')
        page = ET.SubElement(self.file, 'diagram', id='page1', name='Page-1')
        graph = ET.SubElement(page, 'mxGraphModel', page='1', pageScale='1',
                              pageWidth=str(width), pageHeight=str(height), grid='0')
        self.root = ET.SubElement(graph, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.boxes = {}
        self.width, self.height = width, height

    def node(self, key, text, x, y, w, h, color=BLUE, size=17, kind='box'):
        style = (f'html=1;whiteSpace=wrap;fontFamily=Helvetica;fontSize={size};'
                 f'fontColor={INK};fillColor={color};strokeColor=#6b7c90;'
                 'rounded=1;arcSize=8;align=center;verticalAlign=middle;spacing=6;')
        if kind == 'text':
            style += 'shape=text;fillColor=none;strokeColor=none;align=left;'
        if kind == 'panel':
            style += 'rounded=0;fillColor=#ffffff;strokeColor=#c6cfd9;align=left;verticalAlign=top;'
        cell = ET.SubElement(self.root, 'mxCell', id=key, value=html.escape(text).replace('\n', '<br>'),
                             style=style, vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as': 'geometry'})
        self.boxes[key] = (x, y, w, h)

    def edge(self, key, source, target, start=(1, .5), end=(0, .5), via=(), dashed=False):
        style = (f'html=1;edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor={INK};'
                 f'strokeWidth=1.3;endArrow=block;endFill=1;exitX={start[0]};exitY={start[1]};'
                 f'entryX={end[0]};entryY={end[1]};dashed={int(dashed)};')
        cell = ET.SubElement(self.root, 'mxCell', id=key, source=source, target=target,
                             parent='1', edge='1', style=style)
        geometry = ET.SubElement(cell, 'mxGeometry', relative='1', **{'as': 'geometry'})
        if via:
            points = ET.SubElement(geometry, 'Array', **{'as': 'points'})
            for x, y in via:
                ET.SubElement(points, 'mxPoint', x=str(x), y=str(y))

    def save(self, name):
        ET.indent(self.file)
        path = FIGURES / (name + '.drawio')
        ET.ElementTree(self.file).write(path, encoding='utf-8', xml_declaration=True)
        return path


def architecture():
    d = Diagram(1040, 800)
    for key, text, y, h in [('pa', '(a) Availability handling and physical-scale decomposition', 2, 110),
                            ('pb', '(b) Low-frequency residual U-Net: central 96 × 96 × 96 patch', 122, 291),
                            ('pc', '(c) Zero-anchored detail branch: repeat per modality, shared weights', 423, 174),
                            ('pd', '(d) Spatial gate and explicit foreground-logit fusion', 607, 188)]:
        d.node(key, text, 2, y, 1036, h, size=20, kind='panel')
    d.node('mri', 'T1 · T1ce · T2 · FLAIR\n4 × 104³ haloed input patch', 18, 43, 337, 53, BLUE, 18)
    d.node('mask', 'Mask absent MRI\nbinary flags a (4)', 385, 43, 145, 53, GOLD, 16)
    d.node('split', 'Per-modality Gaussian mixture\nσ = 0.5, 0.8, 1.2 mm; 12 logits', 559, 43, 260, 53, GOLD, 17)
    d.node('lh', 'Lₘ = Σⱼ πₘⱼ Gσⱼ(Xₘ)\nHₘ = Xₘ − Lₘ', 850, 43, 170, 53, GREEN, 17)
    d.edge('mri_mask', 'mri', 'mask')
    d.edge('mask_split', 'mask', 'split')
    d.edge('split_lh', 'split', 'lh')
    d.node('input', '4 smooth images\n+ 4 flag maps\n8 × 96³', 18, 169, 145, 64, BLUE, 17)
    widths = (32, 64, 128, 256, 320)
    for i, (width, n) in enumerate(zip(widths, (96, 48, 24, 12, 6))):
        d.node(f'e{i}', f'Encoder {i+1}\n{width} ch · {n}³', 190+166*i, 169, 145, 62, BLUE, 17)
        if i:
            d.edge(f'enc{i}', f'e{i-1}', f'e{i}')
    d.edge('into_unet', 'input', 'e0')
    for i in range(4):
        width = widths[i]
        deeper = widths[i+1]
        n = (96, 48, 24, 12)[i]
        d.node(f'd{i}', f'Decoder {i+1}\n{deeper}+{width} → {width}\n{n}³', 190+166*i, 280, 145, 66, GREEN, 17)
        d.edge(f'skip{i}', f'e{i}', f'd{i}', (.5, 1), (.5, 0), dashed=True)
        if i:
            d.edge(f'dec{i}', f'd{i}', f'd{i-1}', (0, .5), (1, .5))
    d.edge('bottleneck', 'e4', 'd3', (.5, 1), (1, .5), via=((926.5, 313),))
    d.node('fout', 'Context feature F\n32 × 96³', 190, 365, 210, 40, GREEN, 17)
    d.node('zout', '1×1×1 head → base logits z₀\n4 × 96³', 510, 365, 250, 40, BLUE, 17)
    d.edge('features', 'd0', 'fout', (.5, 1), (.5, 0), via=((262.5, 355), (295, 355)))
    d.edge('head', 'd0', 'zout', (.5, 1), (.5, 0), via=((262.5, 355), (635, 355)))
    d.node('blocknote', 'Residual block:\n3³ conv + GN8\n+ LeakyReLU;\n3³ conv + GN8;\nprojected skip\n+ LeakyReLU', 18, 270, 145, 130, size=16, kind='text')
    d.node('skiplegend', 'Encoder: 2× avg pool.\nDecoder: trilinear upsample.\nDashed: skip connections.', 780, 350, 244, 58, size=15, kind='text')
    d.node('rinput', '[F₃₂, Hₘ, eₘ₍₄₎]\n37 input channels', 18, 459, 208, 50, ROSE, 17)
    d.node('rzero', '[F₃₂, 0, eₘ₍₄₎]\n37 input channels', 18, 534, 208, 50, ROSE, 17)
    chain = '3³ conv: 37→16 + ReLU\n3³ conv: 16→16 + ReLU\n1³ conv: 16→3'
    d.node('r1', chain, 255, 455, 272, 59, ROSE, 16)
    d.node('r0', chain, 255, 530, 272, 59, ROSE, 16)
    d.node('subtract', 'Subtract\nR(F,Hₘ,eₘ)\n− R(F,0,eₘ)', 575, 491, 176, 65, ROSE, 17)
    d.node('rout', 'rₘ: 3 × 96³\nNCR/NET · edema · ET\nHₘ = 0 ⇒ rₘ = 0', 804, 491, 216, 65, GREEN, 17)
    d.edge('ri1', 'rinput', 'r1'); d.edge('ri0', 'rzero', 'r0')
    d.edge('rsubtract1', 'r1', 'subtract', (1, .5), (0, .25), via=((550, 484.5), (550, 507.25)))
    d.edge('rsubtract0', 'r0', 'subtract', (1, .5), (0, .75), via=((550, 559.5), (550, 539.75)))
    d.edge('routedge', 'subtract', 'rout')
    d.node('sharednote', 'Same weights θ; no dropout\nor normalization in R', 575, 560, 445, 32, size=15, kind='text')
    d.node('gin', 'Concatenate:\nF₃₂ + Hₘ + rₘ₍₃₎\n+ eₘ₍₄₎ + a₍₄₎\n44 channels', 18, 651, 210, 103, GOLD, 17)
    d.node('gnet', '1³ conv: 44→16 + ReLU\n3³ depthwise conv + ReLU\n1³ conv: 16→1 + sigmoid', 255, 651, 240, 103, GOLD, 17)
    d.node('q', 'Gate qₘ\n∈ (0,1)\n1 × 96³', 522, 665, 91, 75, GOLD, 17)
    d.node('fuse', 'cₘ,ₖ = aₘ qₘ rₘ,ₖ\nzₖ = z₀,ₖ + Σₘ cₘ,ₖ\nBackground:\nzbg = z₀,bg', 643, 651, 226, 103, BLUE, 17)
    d.node('labels', 'Softmax\n4-class argmax\nWT, TC, ET', 894, 651, 126, 103, GREEN, 16)
    for key, a, b in [('g1','gin','gnet'),('g2','gnet','q'),('g3','q','fuse'),('g4','fuse','labels')]:
        d.edge(key,a,b)
    d.node('gatefoot', 'Gate supervision uses labels only during Stage C. Gate values are coefficients, not calibrated quality probabilities.', 18, 767, 1000, 21, size=16, kind='text')
    return d.save('architecture')


def pipeline():
    d = Diagram(500, 520)
    rows = [
        ('A', 'Stage A: base model · 4,000 updates\nTrain U-Net + Gaussian-mixture logits\nObjective: four-class CE + WT/TC/ET soft Dice', BLUE),
        ('B', 'Stage B: corrections · 4,500 updates\nTrain U-Net, mixture, refiner; gates = 1\nBranch dropout = 0.25\nFinal loss + 0.3 base loss + 0.1 branch loss', ROSE),
        ('C', 'Stage C: spatial gates · 1,500 updates\nFreeze candidates; train only the spatial gate\nSegmentation + 0.1 balanced utility BCE', GOLD),
    ]
    for i,(key,text,color) in enumerate(rows):
        d.node(key,text,10,10+122*i,480,97,color,17)
        if i:
            d.edge('to'+key,rows[i-1][0],key,(.5,1),(.5,0))
    d.node('validation','Validation intervals in every stage:\npatient reconstruction; mean WT/TC/ET Dice\nsave best/latest checkpoints and resume state',10,376,480,79,GREEN,17)
    d.edge('toV','C','validation',(.5,1),(.5,0))
    d.node('final','Selected C checkpoint: validation analysis\nRegions, modality subsets and branch removal',10,474,480,44,BLUE,16)
    d.edge('toFinal','validation','final',(.5,1),(.5,0))
    return d.save('training_pipeline')


def render(path, inches):
    tree = ET.parse(path)
    model = tree.find('.//mxGraphModel')
    width,height = float(model.get('pageWidth')),float(model.get('pageHeight'))
    factor = inches*72/width
    fig, ax = plt.subplots(figsize=(inches,inches*height/width))
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,width),ylim=(height,0)); ax.set_axis_off()
    nodes, edges = {}, []
    for cell in model.findall('.//mxCell'):
        style=dict(s.split('=',1) for s in cell.get('style','').split(';') if '=' in s)
        if cell.get('vertex') == '1':
            g=cell.find('mxGeometry')
            rect=tuple(float(g.get(k)) for k in ('x','y','width','height'))
            nodes[cell.get('id')]=(rect,cell,style)
        elif cell.get('edge') == '1': edges.append((cell,style))
    for (x,y,w,h),cell,style in nodes.values():
        is_text=style.get('shape') == 'text'
        is_panel=style.get('verticalAlign') == 'top'
        if not is_text:
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={0 if is_panel else 6}',
                         facecolor=style['fillColor'],edgecolor=style['strokeColor'],lw=.65))
        text=html.unescape(re.sub('<[^>]+>','\n',html.unescape(cell.get('value','')))).strip()
        ax.text(x+8 if (is_panel or is_text) else x+w/2, y+10 if is_panel else y+h/2,
                text, ha='left' if (is_panel or is_text) else 'center',
                va='top' if is_panel else 'center',fontsize=float(style.get('fontSize',17))*factor,
                fontweight='bold' if is_panel else 'normal',color=INK,fontfamily='DejaVu Sans',linespacing=1.3)
    for cell,style in edges:
        a,b=nodes[cell.get('source')][0],nodes[cell.get('target')][0]
        start=(a[0]+a[2]*float(style['exitX']),a[1]+a[3]*float(style['exitY']))
        end=(b[0]+b[2]*float(style['entryX']),b[1]+b[3]*float(style['entryY']))
        points=[start]+[(float(p.get('x')),float(p.get('y'))) for p in cell.findall('.//Array/mxPoint')]+[end]
        dashed=style.get('dashed') == '1'
        if len(points)>2:
            ax.plot(*zip(*points[:-1]), color=INK,lw=.65,ls='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=6,
                                    lw=.65,color=INK,linestyle='--' if dashed else '-'))
    with plt.rc_context({'pdf.fonttype':42,'ps.fonttype':42}):
        fig.savefig(path.with_suffix('.pdf'),facecolor='white')
        fig.savefig(path.with_suffix('.png'),dpi=350,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True,parents=True)
    for path,inches in ((architecture(),7.1),(pipeline(),3.45)):
        render(path,inches)
        print(f'Authored Draw.io and rendered PDF/PNG: {path.stem}')
