"""A separate Draw.io module figure with a balanced two-by-two layout."""
import render_diagrams as rd

rd.FIGURES = rd.ROOT
d = rd.Diagram(1600, 1090)


def panel(key, title, x, y):
    d.node(key,title,x,y,760,515,size=29,kind='panel')
    cell=d.root.find(f"mxCell[@id='{key}']")
    cell.set('style',cell.get('style')+'strokeColor=none;')


panel('a','(a) Gaussian smooth/detail decomposition',10,10)
panel('b','(b) Backbone and zero-anchored detail edits',830,10)
panel('c','(c) Shared spatial utility gate',10,560)
panel('d','(d) Utility supervision and additive fusion',830,560)
d.node('vdivider','',798,15,1,1050,'#d9dee5',1)
d.node('hdivider','',10,543,1580,1,'#d9dee5',1)
for key in ('vdivider','hdivider'):
    cell=d.root.find(f"mxCell[@id='{key}']")
    cell.set('style',cell.get('style')+'rounded=0;verticalAlign=top;strokeColor=none;')

d.node('scales','Three scales: 0.5, 0.8, 1.2 mm; πₘ = softmax(αₘ)',25,58,720,39,size=24,kind='text')
d.node('mri','T1 · T1ce\nT2 · FLAIR\nmasked Xₘ',25,208,165,95,rd.BLUE,26)
for j, (sigma,y) in enumerate((('0.5',108),('0.8',222),('1.2',336))):
    d.node(f'g{j}',f'Gaussian\nσ = {sigma} mm',248,y,175,75,rd.GOLD,26)
    entry=(0,.18+j*.32)
    d.edge(f'filter{j}','mri',f'g{j}',(1,.5),(0,.5),via=((220,255.5),(220,y+37.5)))
    d.edge(f'mix{j}',f'g{j}','mix',(1,.5),entry,via=((466,y+37.5),(466,222+86*entry[1])))
d.node('mix','Learned mixture\nLₘ = Σⱼ πₘⱼ Gσⱼ(Xₘ)',496,222,245,86,rd.GOLD,24)
d.node('low','Smooth Lₘ\n→ backbone',511,341,216,65,rd.BLUE,26)
d.node('high','Detail Hₘ = Xₘ − Lₘ\n→ correction branch',453,445,289,65,rd.ROSE,24)
d.edge('mixture_low','mix','low',(.5,1),(.5,0))
d.edge('low_detail','low','high',(.5,1),(.5,0))
d.edge('raw_detail','mri','high',(.5,1),(0,.5),via=((107.5,477.5),))

d.node('backbone','Input: four L images + four availability maps\nResidual 3D U-Net: 32 → 64 → 128 → 256 → 320\nOutputs: base logits z₀ (4 channels), context F (32)',849,85,710,120,rd.BLUE,24)
d.node('r1','Rθ(F, Hₘ, eₘ)\n37 → 16 → 16 → 3\nKernels: 3³, 3³, 1³',848,271,321,93,rd.ROSE,24)
d.node('r0','Rθ(F, 0, eₘ)\n37 → 16 → 16 → 3\nKernels: 3³, 3³, 1³',1230,271,321,93,rd.ROSE,24)
d.edge('context_r1','backbone','r1',(.23,1),(.5,0))
d.edge('context_r0','backbone','r0',(.77,1),(.5,0))
d.node('subtract','Anchored subtraction\nrₘ = Rθ(F,Hₘ,eₘ) − Rθ(F,0,eₘ)',950,415,530,72,rd.GREEN,24)
d.edge('r1_minus','r1','subtract',(.5,1),(.2,0),via=((1008.5,391),(1056,391)))
d.edge('r0_minus','r0','subtract',(.5,1),(.8,0),via=((1390.5,391),(1374,391)))
d.node('anchor_note','Shared θ; three foreground channels; Hₘ = 0 ⇒ rₘ = 0.',849,497,710,26,size=23,kind='text')

d.node('concat','Concatenate F(32), Hₘ(1), rₘ(3), eₘ(4), a(4)\n44 input channels at each voxel',27,636,714,71,rd.BLUE,24)
d.node('conv1','1³ pointwise\n44 → 16\nReLU',27,779,187,103,rd.GOLD,26)
d.node('dw','3³ depthwise\n16 → 16\nReLU',290,779,187,103,rd.GOLD,26)
d.node('conv2','1³ pointwise\n16 → 1\nSigmoid',552,779,187,103,rd.GOLD,26)
d.edge('gate_input','concat','conv1',(.13,1),(.5,0))
d.edge('point_dw','conv1','dw')
d.edge('dw_point','dw','conv2')
d.node('q','qₘ(v) ∈ (0,1): one weight per modality and voxel\nA learned coefficient, not calibrated confidence',27,927,714,87,rd.GREEN,24)
d.edge('to_q','conv2','q',(.5,1),(.88,0))
d.node('gate_note','Shared parameters are reused across the four modalities.',27,1034,714,30,size=23,kind='text')

d.node('utility','STAGE C: freeze U-Net, mixture and refiner\nΔₘ = CE(without candidate m) − CE(all candidates)\ntₘ = stopgrad sigmoid(AvgPool₃(Δₘ) / 0.1)\nLoss: segmentation + 0.1 balanced gate BCE',849,634,710,173,rd.ROSE,24)
d.edge('utility_training','utility','conv2',(0,.5),(.5,0),via=((785,720.5),(785,750),(645.5,750)),dashed=True)
d.node('fusion','Foreground: zₖ = z₀,ₖ + Σₘ aₘ qₘ rₘ,ₖ\nBackground: zbg = z₀,bg\nSoftmax / argmax → 4 classes → derive WT, TC, ET',849,897,710,126,rd.BLUE,24)
d.edge('q_fusion','q','fusion',(1,.38),(0,.5))
d.node('fusion_note','aₘ = 0 removes an unavailable modality’s contribution.\nDashed arrow: gate-training supervision only.',849,1040,710,42,size=22,kind='text')

path=d.save('detailtrust_modules_2x2')
rd.render(path,14.2)
print('Created independent Draw.io source and matching PDF/PNG:',path)
