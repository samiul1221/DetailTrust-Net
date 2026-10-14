"""Create a beginner-readable Excalidraw scene and matching local PNG preview."""
from pathlib import Path
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 1200, 2100
INK = '#24354b'
BLUE, GREEN, GOLD, ROSE = '#e8f0fa', '#e7f2ee', '#fff3d6', '#f6e9ef'
elements = []


def base(kind, x, y, width, height, color='transparent'):
    index = len(elements)
    return dict(id=f'detailtrust-{index}', type=kind, x=x, y=y, width=width, height=height,
                angle=0, strokeColor=INK, backgroundColor=color, fillStyle='solid',
                strokeWidth=1.5, strokeStyle='solid', roughness=0, opacity=100,
                groupIds=[], frameId=None, roundness={'type': 3} if kind == 'rectangle' else None,
                seed=1000+index, version=1, versionNonce=10000+index, isDeleted=False,
                boundElements=[], updated=int(time.time()*1000), link=None, locked=False)


def box(x, y, width, height, color='white'):
    element = base('rectangle', x, y, width, height, color)
    elements.append(element)
    return element['id']


def text(value, x, y, width, height, size=21, color=INK, align='left'):
    element = base('text', x, y, width, height)
    element.update(text=value, originalText=value, fontSize=size, fontFamily=2,
                   textAlign=align, verticalAlign='top', containerId=None,
                   autoResize=True, lineHeight=1.25, strokeColor=color)
    elements.append(element)


def arrow(start, end, via=(), color=INK):
    points=[start, *via, end]
    element=base('arrow', start[0], start[1], max(p[0] for p in points)-min(p[0] for p in points),
                 max(p[1] for p in points)-min(p[1] for p in points))
    element.update(points=[[x-start[0],y-start[1]] for x,y in points],
                   lastCommittedPoint=None, startBinding=None, endBinding=None,
                   startArrowhead=None, endArrowhead='arrow', elbowed=False, strokeColor=color)
    elements.append(element)


def card(title, body, x, y, width, height, color, size=21):
    box(x,y,width,height,color)
    text(title,x+20,y+16,width-40,36,24)
    text(body,x+20,y+63,width-40,height-73,size)


def create_scene():
    text('DetailTrust-Net — understand the whole model',25,20,1150,48,36)
    text('Main idea: make a first tumour map, propose detail-based edits,\nthen learn where those edits should contribute.',25,77,1150,63,26)

    box(20,160,1160,144,BLUE)
    text('1  Start with four aligned 3D MRI scans of the same brain',40,175,1120,32,24)
    text('T1: anatomy\nT1ce: contrast enhancement',40,222,340,60,22)
    text('T2: fluid-sensitive contrast\nFLAIR: fluid suppression',385,222,370,60,22)
    text('Missing scan? Set it to zero.\nAdd four present/absent flags.\nAt least one MRI is required.',795,215,350,80,21)
    arrow((600,307),(600,337))

    box(20,340,1160,174,GOLD)
    text('2  Separate each scan into smooth structure and leftover detail',40,355,1120,33,24)
    text('Mix 0.5, 0.8 and 1.2 mm Gaussian blurs; each modality learns one mixture shared across patients.',40,396,1120,28,21)
    box(40,438,520,57,'white')
    text('Smooth part L: broad shapes and context',56,451,486,30,22)
    box(640,438,520,57,'white')
    text('Detail H = original scan − smooth part',656,451,486,30,22)
    arrow((300,517),(300,551))
    arrow((900,517),(900,551))

    card('3  Make the first segmentation guess',
         'A 3D residual U-Net reads the four smooth\nimages plus four availability maps.\n\nIt outputs two things:\n• Base scores z₀ for four tissue classes.\n• Context F: 32 learned feature channels.\n\nFive levels: 32 → 64 → 128 → 256 → 320.',
         20,555,550,294,BLUE,21)
    card('4  Propose an edit from each scan',
         'One shared refiner uses:\ncontext F + that scan’s detail H + its identity.\n\nRun it twice: with detail, and with detail = 0.\nSubtract the second output from the first:\nr = R(F,H,identity) − R(F,0,identity).\nNo detail ⇒ zero edit.\n+ raises a tumour score; − lowers it.',
         630,555,550,294,ROSE,21)
    arrow((574,665),(626,665))
    text('F',588,631,35,30,20,align='center')
    arrow((900,853),(900,889))
    arrow((300,853),(300,1138),via=((300,874),(4,874),(4,1127),(300,1127)))

    card('Base scores take the direct path',
         'The first guess is kept as the starting point.\nDetail branches propose changes to the three\nforeground scores, not a replacement map.\n\nA missing modality contributes no edit.\nShared weights are reused for all four scans.',
         20,893,550,229,GREEN,21)
    card('5  Decide how much of each edit to use',
         'A shared spatial gate reads F, detail, edit,\nmodality identity and availability flags.\n\nAt every voxel, it outputs a weight q (0 to 1).\nSmall q: suppress the edit. Large q: retain it.\nThis weight is not a confidence probability.',
         630,893,550,229,GOLD,21)
    arrow((900,1126),(900,1138))

    box(20,1142,1160,137,GREEN)
    text('6  Add the weighted edits to the first guess',40,1158,1120,35,25)
    text('Final foreground score = base score + sum of (available × gate weight × signed edit)',40,1202,1120,29,23)
    text('Only NCR/NET, edema and ET scores receive corrections. The base background score stays unchanged.',40,1240,1120,28,20)
    arrow((600,1283),(600,1319))

    box(20,1323,1160,192,BLUE)
    text('7  Turn scores into a complete 3D tumour map',40,1339,1120,34,25)
    text('Combine overlapping patch predictions, then choose one class for each voxel (a 3D pixel).',40,1384,1120,29,23)
    text('Background: other tissue     NCR/NET: necrotic or non-enhancing tumour core\nEdema: surrounding tumour-associated swelling     ET: contrast-enhancing tumour',40,1426,1120,60,22)
    text('Report regions: WT = all three foreground labels; TC = NCR/NET + ET; ET = enhancing tumour.',40,1485,1120,25,20)

    text('How the model learns (training only)',25,1551,1150,38,29)
    text('Reference labels teach the model during training. They are never an input during prediction.',25,1594,1150,30,22)
    card('A  Learn the first guess',
         'Train the U-Net and\nGaussian mixture.\n\nUse cross-entropy plus\nWT/TC/ET soft Dice.\nCorrections are inactive.',
         20,1641,365,245,BLUE,21)
    card('B  Learn the edit proposals',
         'Train U-Net, mixture\nand the shared refiner.\nGates are fixed at 1.\nDrop edits with p = 0.25.\n\nFinal loss + 0.3 base loss\n+ 0.1 one-branch loss.',
         417,1641,365,245,ROSE,21)
    card('C  Teach the spatial gates',
         'Freeze U-Net and refiner\n(and smoothing mixture).\nRemove one candidate edit:\ndoes voxelwise CE worsen?\nIf yes, favour that edit.\n\nSegmentation + 0.1 utility BCE.',
         814,1641,365,245,GOLD,20)
    text('Utility target: compare all ungated edits with one removed; average the CE change over 3³ neighbours,\napply sigmoid(change / 0.1), detach it, and supervise available gates with balanced tumour/background BCE.',25,1900,1150,55,19)
    text('CE = cross-entropy: penalizes wrong class scores. Dice = overlap of prediction and reference.\nBCE = binary cross-entropy: fits the gate to its soft utility target. Lower loss is better.',25,1970,1150,58,20)
    text('Layer widths: shared refiner 37→16→16→3; shared gate 44→16→16→1 (middle layer is depthwise).\nThis explains the proposed design; it does not show a measured improvement or calibrated certainty.',25,2040,1150,55,19)
    return {'type':'excalidraw','version':2,'source':'https://excalidraw.com',
            'elements':elements,'appState':{'viewBackgroundColor':'#ffffff','gridSize':None},'files':{}}


def render(scene, path):
    fig,ax=plt.subplots(figsize=(12,HEIGHT/100))
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,WIDTH),ylim=(HEIGHT,0)); ax.set_axis_off()
    texts=[]
    for el in scene['elements']:
        x,y,w,h=el['x'],el['y'],el['width'],el['height']
        if el['type']=='rectangle':
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=9',
                                      facecolor=el['backgroundColor'],edgecolor='#748398',lw=1.0))
        elif el['type']=='text':
            tx=x+w/2 if el['textAlign']=='center' else x
            t=ax.text(tx,y,el['text'],va='top',ha=el['textAlign'],fontsize=el['fontSize']*.72,
                      color=el['strokeColor'],fontfamily='DejaVu Sans',linespacing=1.25)
            texts.append((t,el))
        elif el['type']=='arrow':
            points=[(x+a,y+b) for a,b in el['points']]
            if len(points)>2:
                ax.plot(*zip(*points[:-1]),color=INK,lw=1.3)
            ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                                        mutation_scale=12,lw=1.3,color=INK))
    fig.canvas.draw()
    failures=[]
    for t,el in texts:
        bb=t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        if abs(bb.width)>el['width']+2 or abs(bb.height)>el['height']+5:
            failures.append({'text':el['text'].splitlines()[0],'actual':[abs(bb.width),abs(bb.height)],
                             'allocated':[el['width'],el['height']]})
    if failures:
        raise ValueError('Text overflow: '+json.dumps(failures))
    fig.savefig(path,dpi=250,facecolor='white')
    plt.close(fig)
    return dict(text_elements=len(texts),text_bounds_passed=True,png_dimensions=[3000,int(HEIGHT*2.5)])


if __name__=='__main__':
    scene=create_scene()
    native=ROOT/'detailtrust_arch_explained.excalidraw'
    native.write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
    image=ROOT/'detailtrust_arch_explained.png'
    check=render(scene,image)
    check.update(source=str(native),preview=str(image),remote_mcp_used=False,
                 renderer='Local rendering of the exact editable Excalidraw scene')
    (ROOT/'verification'/'beginner_architecture_checks.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))
