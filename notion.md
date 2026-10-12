# 11 Papers — Simple Explanation Notes

*Written for: someone who knows basic deep learning (CNNs, attention, transformers) but has **never touched medical imaging**.Every medical term is explained with an everyday analogy.*

---

# PART 0 — PREREQUISITES

*Read this once. It’s everything you need. About 10 minutes.*

---

## 0.1 What is a brain tumor?

Your body is made of cells that grow, divide and die in a controlled way. Sometimes a group of cells **stops obeying the rules** and keeps multiplying. That extra lump of cells is a **tumor**.

In the brain this is especially bad because **your skull is a sealed box**. There’s no room for the lump to expand, so it presses on the brain and damages it.

That’s all the biology you need.

---

## 0.2 What is an MRI?

**MRI = Magnetic Resonance Imaging.** A machine that photographs the inside of your body using magnets and radio waves.

**Key thing to understand: it does not produce one photo.**

Imagine a loaf of bread. You slice it, lay the slices in a row, and photograph each slice. Now you have ~150 photos, and together they describe the whole loaf in 3D.

An MRI does exactly this to a brain. **~150 cross-section slices, stacked = a 3D volume.**

```
        Real brain (3D)              What the MRI gives you

           ▄▄▄▄▄                     [slice 1]  ○
         ▄███████▄                   [slice 2]  ◯
        █████████████       ──►      [slice 3]  ◎    ...150 of these
         ▀███████▀                   [slice 4]  ◯
           ▀▀▀▀▀                     [slice 5]  ○
```

### New word: voxel

- A **pixel** = “picture element” = one square in a 2D image
- A **voxel** = “volume element” = one cube in a 3D volume

You’ll see “voxel” everywhere. It just means “3D pixel.”

**Scale check:** one brain MRI has roughly **10 million voxels**. That’s why 3D models are expensive to train.

---

## 0.3 Why is this a problem worth solving?

A **radiologist** (a doctor who specialises in reading medical images) has to:

1. Look through all ~150 slices
2. Find the tumor
3. **Trace its outline by hand** on every slice it appears in

**This takes hours per patient.** And two different doctors will draw slightly different outlines.

**Why does anyone need the outline at all?**

- **Radiation therapy** points a beam at the tumor. If the outline is wrong, you burn healthy brain or miss part of the tumor.
- **Checking if treatment is working** means measuring the tumor’s *volume* today vs three months ago. No outline, no volume.

**So the goal of all 11 papers:** make a neural network draw that outline automatically.

In DL terms: **3D semantic segmentation.** Classify every voxel as tumor or not-tumor.

---

## 0.4 ⚠ The four MRI images (most important prerequisite)

Here’s the thing that confuses everyone at first.

**An MRI machine can be tuned in four different ways.** Each tuning makes different tissue light up. So you don’t get one image of the brain — **you get four completely different-looking images of the same brain.**

> **Analogy:** photograph one room four times — with a normal camera, an infrared camera, a UV camera, and an X-ray. Same room, four totally different pictures. Each one reveals something the others can’t see.
> 

These four are called **modalities**:

| Name | What lights up in it | Think of it as |
| --- | --- | --- |
| **T1** | Normal anatomy | The “plain photo.” Tumor barely visible. Good for seeing brain structure |
| **T1ce** | **The aggressive, actively growing tumor** | They **inject a dye** into the patient’s blood. Healthy brain blocks it. Aggressive tumors have leaky blood vessels, so dye leaks in and **that region glows bright** |
| **T2** | Water and fluid | Anything wet is bright |
| **FLAIR** | The swelling around the tumor | Same as T2, but the *normal* brain fluid is darkened, so the *abnormal* swelling stands out. Like a Photoshop layer that hides one thing so another pops |

### How the network eats this

You know how a colour image is 3 channels (R, G, B)? Same idea here — except **4 channels, one per modality**:

```
Normal image:   3 × H × W          (RGB)
Brain MRI:      4 × H × W × D      (4 modalities, and it's 3D)
```

### ⚠ But they are NOT like RGB channels

This is the single most important thing in these notes.

In an RGB photo, if you delete the blue channel, the image gets weird but you can still recognise everything. The channels overlap a lot.

**Here, deleting T1ce is catastrophic.** The aggressive tumor region is *literally defined* by where the injected dye glows. Delete that image and the information is simply **gone** — the other three don’t contain it.

**Five of your eleven papers exist because of this one fact.** Hold on to it.

---

## 0.5 The three tumor regions

A tumor isn’t one uniform blob. It has layers.

> **Analogy — think of a bruise:**
- A dark damaged centre
- An angry active ring around it
- Mild swelling spreading outward
> 

Brain tumors are labelled the same way, as **three nested regions**:

```
┌──────────────────────────────────────────┐
│  WT = Whole Tumor                        │  ← everything, incl. swelling
│   ┌────────────────────────────────┐     │     BIGGEST, EASIEST
│   │  TC = Tumor Core               │     │  ← the solid mass
│   │   ┌──────────────────────┐     │     │     medium difficulty
│   │   │  ET = Enhancing      │     │     │  ← the aggressive
│   │   │       Tumor          │     │     │     growing rim
│   │   └──────────────────────┘     │     │     SMALLEST, HARDEST
│   └────────────────────────────────┘     │
└──────────────────────────────────────────┘
```

- **ET** — the aggressive growing edge. Glows in **T1ce** because of the dye
- **TC** — ET plus the dead tissue in the middle. The solid mass
- **WT** — TC plus the swelling around it

**Every paper reports three numbers, usually in ET / TC / WT order.** A typical good result looks like `0.80 / 0.85 / 0.91`.

**ET is always the lowest number.** It’s the smallest and has the blurriest edge. **That’s expected, not a bug.** If you see a paper where ET ≈ WT, be suspicious.

---

## 0.6 How performance is measured

### Plain accuracy is useless here — understand why

Out of ~10 million voxels in a scan, maybe 100,000 are tumor. So **~99% of voxels are background.**

A model that outputs “there is no tumor anywhere, ever” scores **99% accuracy** while being completely worthless.

This is called **class imbalance**, and it’s why this field never uses plain accuracy or plain cross-entropy loss. They use Dice loss and Focal loss instead.

### Dice score — the real metric

**Dice measures how much your predicted shape overlaps the doctor’s shape.**

```
Dice = 2 × (overlap) / (your area + doctor's area)
```

- **1.0** = perfect match
- **0.0** = no overlap
- **Good result = 0.80 to 0.90**

If you know F1-score, **Dice is exactly F1 computed over voxels.** It ignores background completely, which is the point — the “predict nothing” cheat scores 0.

### ⚠ The number that anchors everything

**Two expert doctors, given the same scan, only agree with each other at Dice 0.74–0.85.**

Read that again. There is **no objective correct answer** here — only expert opinion, and the experts disagree with each other.

**What this means for you:** if a paper claims Dice **0.95 or higher**, it is *not* “better than a doctor.” Something in their measurement is broken. This is your main bullshit detector.

### HD95 — the other metric

Dice can look great while your boundary is horribly wrong in one spot. Imagine you outline the tumor perfectly, but also draw a small stray blob on the opposite side of the brain. Dice barely changes. But you just told a surgeon to irradiate healthy brain.

**HD95 = the worst-case boundary error, in millimetres.** Lower is better.

> **Analogy:** Dice is your overall exam percentage. HD95 is your worst single question. You can score 90% overall and still have failed the one that mattered.
> 

Good papers report both.

---

## 0.7 BraTS — the dataset

**BraTS = Brain Tumor Segmentation Challenge.** A public dataset plus an annual competition, running since 2012.

> **Analogy:** BraTS is to medical segmentation what **ImageNet** is to image classification. It’s the shared exam everyone sits, so results can be compared.
> 

Two things to know:

1. **The test labels are hidden.** You submit predictions to a server. So you literally cannot cheat, which makes challenge rankings the most trustworthy numbers in the field.
2. **Every year is a different dataset.** BraTS 2018 ≠ BraTS 2021. **Numbers from different years are not comparable.** If one paper says 0.85 on BraTS 2018 and another says 0.88 on BraTS 2021, that tells you nothing.

---

## 0.8 Three ways this differs from normal computer vision

| Difference | What it means for you |
| --- | --- |
| **Extreme class imbalance** | Tumor is <1% of voxels. In COCO or Cityscapes your target classes fill a decent chunk of the frame. Here they don’t, so standard losses fail |
| **Noisy ground truth** | A cat in ImageNet is definitely a cat. A tumor boundary is one doctor’s opinion. Your model’s ceiling is set by label noise, not architecture |
| **3D + tiny datasets** | ~10M voxels per scan means you train on small 3D **patches**, not whole volumes. And BraTS 2021 has ~1,251 cases vs ImageNet’s 1.2 million — **1000× less data** |

---

## 0.9 DL terms used in these papers

You know CNNs, attention and transformers. These are the less-common ones that appear:

| Term | Plain explanation |
| --- | --- |
| **Disentanglement** | Splitting a feature vector so different parts mean different things. “This half encodes *which modality this is*, that half encodes *what the anatomy looks like*” |
| **Contrastive learning** | Train by pulling similar things together and pushing different things apart in feature space. Needs no labels |
| **Knowledge distillation** | A strong **teacher** model trains a weaker **student** by making the student copy the teacher’s internal features |
| **Hypernetwork** | A network that **generates another network’s weights**. Instead of storing 15 models, store 1 generator that produces the right weights on demand |
| **Mixture of Experts (MoE)** | Several specialist sub-networks plus a small **gating network** that picks which specialist handles each input |
| **Mamba (State Space Model)** | Alternative to attention. Gets global context at **linear** cost instead of attention’s **quadratic** cost. Matters enormously in 3D |
| **MAE (Masked Autoencoder)** | Self-supervised training: hide chunks of the input, make the model reconstruct them. Forces it to learn structure. No labels needed |
| **Deformable convolution** | A convolution whose sampling grid can bend and shift to follow object shape, instead of being a rigid square |

# PART 1 — YOUR TOP 5 PAPERS

**How I ranked them:** journal tier (IEEE TMI > TNNLS ≈ JBHI > TAI), how brain-tumor-specific they are, conceptual novelty, and citation count.

| Rank | Paper | Journal · Year | Why it made the top 5 |
| --- | --- | --- | --- |
|  **1. Samiul** | **HFF-Net** | IEEE TMI · 2025 | Most original idea in your entire set. Changes *how the image is represented*, not just the architecture |
| **2. Sneha**  | **MoME** | IEEE TMI · 2025 | The “one model for everything” direction. Tested on 9 datasets, 5 modalities, 8 lesion types |
| **3. Lakhay** | **M²FTrans** | IEEE JBHI · 2023 | The cleanest, safest solution to missing modalities. Widely used baseline, code public |
| **4. Krish** | **D²-Net** | IEEE TMI · 2022 | Most-cited in your set (107). The paper that started the disentanglement approach |
| **5. Vikash** | **MBUNeXt** | IEEE TNNLS · 2025 | Current SOTA with hard numbers, and the only one tested on a genuinely different population |

---

## ★ PAPER 1 — HFF-Net

### *Rethinking Brain Tumor Segmentation From the Frequency Domain Perspective*

**Minye Shao et al. · IEEE Transactions on Medical Imaging · 2025**`github.com/VinyehShaw/HFF`

### ❓ What problem is it solving?

**Everyone is bad at the ET region specifically.** (Remember: ET = the aggressive rim that glows in T1ce.)

The paper asks *why*, and gives an answer nobody else gave: existing methods **ignore what MRI images actually look like.** MRI tumors have complex textures and **directional** patterns — edges running in particular directions. A standard CNN just processes raw pixel values and never explicitly separates “smooth shape” from “fine texture.”

### 💡 The core idea

**Stop looking only at pixels. Split the image into frequencies.**

> **What does “frequency” mean for an image?**
> 
> 
> Think of music. A song contains **bass** (slow, smooth, low-frequency) and **treble** (fast, detailed, high-frequency). An equaliser lets you handle them separately.
> 
> Images work the same way:
> - **Low frequency** = slow, smooth changes = **the overall shape and contour** of the tumor
> - **High frequency** = rapid changes = **edges, fine texture, directional detail**
> 
> A normal CNN mashes these together. HFF-Net separates them, so each can be processed the way it deserves.
> 

### ⚙️ How it works — three modules

| Module | Full name | What it does, plainly |
| --- | --- | --- |
| **FDD** | Frequency Domain Decomposition | The equaliser. Splits the MRI into **low-frequency** (smooth tumor outlines) and **high-frequency** (textures and directional edges) parts |
| **ALC** | Adaptive Laplacian Convolution | The **Laplacian** is a classic edge-detector filter from old-school image processing. Normally it’s a fixed kernel. Here the kernel **updates itself during training**, so it learns to emphasise whichever edges matter for *this* image |
| **FDCA** | Frequency Domain Cross-Attention | Merges everything back together, combining three kinds of information: **what** it is (semantic), **where** it is (positional), and **which slice** it’s on (this is 3D, slices differ) |

!image.png

```
MRI scan
   ↓
FDD  ── splits like an audio equaliser ──┐
   ├──► LOW frequency  → smooth tumor contours
   └──► HIGH frequency → texture + directional edges
   ↓
ALC ── self-updating Laplacian edge filter
       sharpens the high-frequency details that matter
   ↓
FDCA ── cross-attention merges everything back
        (semantic + positional + slice info)
   ↓
Final segmentation: ET / TC / WT
```

### 📊 Results

Tested on **four public datasets**:

| Measure | Improvement |
| --- | --- |
| Mean Dice, all three regions | **+4.48%** relative (range 2.39–7.72%) |
| **Enhancing tumor (ET) specifically** | **+7.33%** relative (range 5.96–8.64%) |
| Computational cost | Stayed efficient |

They also backed it up with visualisations and theoretical reasoning, not just a table of numbers.

### ⭐ Why it matters

**The improvement is bigger on ET (+7.33%) than on the overall average (+4.48%).** That’s the important detail. It means the gain lands exactly where the problem is genuinely hard, not where it was already easy. Most papers improve the easy parts and quietly average it out.

Also: almost every other paper in your set is rearranging attention blocks and skip connections. **This one changes the representation domain itself.** That’s a different and rarer kind of contribution.

> 🎤 **If you present one paper as “the interesting one,” make it this.**
> 

Details

---

## ★ PAPER 2 — MoME

### *A Foundation Model for Lesion Segmentation on Brain MRI With Mixture of Modality Experts*

**Xinru Zhang et al. · IEEE Transactions on Medical Imaging · 2025**

### ❓ What problem is it solving?

Not accuracy — **usability**.

Today’s models are **task-specific**. One model for gliomas on T1ce. A different model for stroke lesions on FLAIR. A third for something else.

**To use them, you must already know what you’re looking for.** You have to know the lesion type and the modality *before* you can pick the right model.

That’s fine in a research paper. In a hospital it’s backwards — the scan arrives and you *don’t yet know* what’s in it. That’s the whole reason you’re looking.

### 💡 The core idea

**Build one universal model that handles any lesion type on any modality, automatically.**

The mechanism is **Mixture of Experts**.

> **What is Mixture of Experts?**
> 
> 
> Instead of one network trying to be good at everything, you train several **specialist** networks plus a small **gating network** that decides which specialist should handle each input.
> 
> **Analogy:** a hospital front desk. You don’t see one GP who treats everything badly. You describe your problem, and the desk routes you to the right specialist. The “front desk” is the gating network.
> 

### ⚙️ How it works

| Component | What it does |
| --- | --- |
| **Multiple expert networks** | Each expert specialises in a different imaging modality. One handles T1-like inputs, another FLAIR-like, and so on |
| **Hierarchical gating network** | Combines the experts’ outputs. **“Hierarchical”** means it combines at several levels rather than one final vote — so experts **collaborate** instead of one winning outright |
| **Curriculum learning** | ⚠ **The clever bit.** Without it, the experts **degenerate** — they all drift toward learning the same thing, and you’ve lost the entire point of having experts. Curriculum learning (train on easy examples first, hard later) keeps each expert **specialised** |
| **MoME+** | An extension with a **soft dispatch network** that routes *combinations* of modalities, not just single ones |

```
Input: any brain MRI. Unknown lesion type. Unknown modality.
   ↓
   ┌── Expert 1 ──┐
   ├── Expert 2 ──┤   each specialises in a
   ├── Expert 3 ──┤   different modality
   └── Expert N ──┘
   ↓
HIERARCHICAL GATING NETWORK
   combines expert outputs at multiple levels
   → experts collaborate, not compete
   ↓
(during training, curriculum learning stops
 the experts from collapsing into clones)
   ↓
Segmentation — no need to tell it anything in advance
```

### 📊 Results

- Tested on **9 brain lesion datasets**
- Covering **5 imaging modalities** and **8 lesion types**
- Beat state-of-the-art universal models
- Showed **promising generalisation to datasets it had never seen**

### ⭐ Why it matters

This is the **“foundation model”** direction — the same shift that happened in NLP (one GPT instead of a separate model per task) arriving in 3D medical imaging.

The **curriculum learning detail is the tell that they actually built this thing.** Expert degeneration is a failure you only discover by running it, not by theorising. Papers that mention their own failure modes are usually the honest ones.

---

## ★ PAPER 3 — M²FTrans

### *Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation*

**Junjie Shi et al. · IEEE Journal of Biomedical and Health Informatics · 2023**`github.com/Jun-Jie-Shi/M2FTrans`

### ❓ What problem is it solving?

**Hospitals often don’t have all four scans.**

Why this happens constantly:
- The patient is **allergic to the injected dye** → no T1ce
- The scan got **corrupted**, or the patient moved
- The hospital **skipped a sequence** to save scanner time or money
- Different hospitals just follow **different protocols**

**The scale of the problem:** with 4 modalities there are **15 possible combinations** a patient could show up with. Training 15 separate models is absurd. You need **one model that handles all 15**.

**And you can’t just fill the missing one with zeros** — because zero is a *real value* in an MRI. It means dark tissue. The model can’t tell “this scan is absent” from “this scan is present and dark.”

The paper identifies two specific technical failures:

1. **Attention breaks with holes in the input.** Feed self-attention a sequence with missing pieces and the attention weights get distorted.
2. **The model becomes addicted to T1ce.** T1ce is so informative for ET that the model over-relies on it — then collapses badly when it’s gone.

### 💡 The core idea

**Tell the attention mechanism explicitly which scans exist. And rebalance so no single scan dominates.**

### ⚙️ How it works

**Fix 1 — fusion tokens + masked attention**

> **Analogy:** imagine a weekly meeting where different people show up each time.
> 
> 
> ❌ **Bad approach:** everyone must talk to everyone. Breaks the moment someone’s absent — people wait for a person who isn’t coming.
> 
> ✅ **M²FTrans approach:** appoint a **chairperson**. Whoever shows up reports to the chair. The chair combines everything. Absence is handled naturally — the chair just gets fewer reports.
> 
> The **“fusion token”** is the chairperson. It’s a learnable vector that everything reports into.
> 

The **mask** is a simple binary vector like `[1, 0, 1, 1]` meaning “T1ce is missing.” It zeroes out attention to absent modalities so they never corrupt the weights.

**Fix 2 — rebalancing**
- **Spatial weight attention** — *where* in the image does each modality contribute?
- **Channel-wise fusion transformer** — *which feature channels* from each modality actually matter?
- Together: remove duplicated information and stop any one modality from dominating.

```
INPUT: whatever scans exist
       + MASK vector [1, 0, 1, 1]   ← "T1ce is missing"
   ↓
Separate encoder per available modality
   (absent ones are SKIPPED, not zero-filled)
   ↓
REBALANCING
   spatial weight attention + channel-wise fusion
   → strip redundancy, prevent T1ce addiction
   ↓
MASKED FUSION TRANSFORMER
   learnable fusion tokens (the "chairperson")
   + masked self-attention (mask blocks absent modalities)
   ↓
Decoder → ET / TC / WT
```

**Training trick:** randomly drop modalities during training, so the model practises many of the 15 combinations.

### 📊 Results

Tested on **BraTS 2018, 2020 and 2021**. Beat state-of-the-art by large margins across the incomplete-modality settings — with **one model** covering all combinations instead of 15.

### ⭐ Why it matters

**It never invents data.** Some competing methods *generate* the missing scan with a GAN. That’s risky — generative models can **hallucinate** structures that aren’t there, which is genuinely dangerous in a diagnostic setting. M²FTrans just handles absence gracefully instead. Safer, and it’s where the field has largely gone.

> ⚠ **The honest limit you should know:** ET is *defined* by dye uptake in T1ce. Remove T1ce and ET Dice collapses from ~0.80 to ~0.60 — across **every method anyone has ever tried**. Dozens of papers, same ceiling.
> 
> 
> **That’s an information limit, not a modelling limit.** No architecture can recover information that was never captured. The other three scans simply don’t contain it.
> 
> 🎤 *Saying this shows you can tell the difference between “researchers haven’t been clever enough yet” and “the data physically isn’t there.” That distinction is what separates understanding a field from understanding its papers.*
> 

---

## ★ PAPER 4 — D²-Net

### *Dual Disentanglement Network for Brain Tumor Segmentation With Missing Modalities*

**Qiushi Yang et al. · IEEE Transactions on Medical Imaging · 2022 · 107 citations**`github.com/CityU-AIM-Group/D2Net`

### ❓ What problem is it solving?

Same missing-modality problem, but the paper spots a flaw in **how everyone was solving it**.

Existing methods **blend all four modalities into one shared representation.** The paper argues this throws away something crucial:

**Different modalities matter for different tumor regions.**
- T1ce tells you about **ET**
- FLAIR tells you about **swelling**

If you blend everything into one mush, you lose track of *which scan was carrying which piece of information*. So when one goes missing, **you don’t know what specifically you lost.**

### 💡 The core idea

**Don’t blend. Deliberately separate things out — twice.**

Hence “**dual** disentanglement”: one stage separates the *modalities*, another separates the *tumor regions*.

> **What is disentanglement?**
> 
> 
> Imagine a smoothie made from banana, mango and spinach. Once blended, you can’t get the banana back out.
> 
> Disentanglement is the opposite approach: keep the ingredients **in separate labelled containers**. Now if the mango is unavailable, you know exactly what’s missing and what you still have.
> 

### ⚙️ How it works — two stages

**STAGE 1 — MD-Stage (Modality Disentanglement)**

*Goal:* explicitly pull out “what makes T1ce look like T1ce” as a separable quantity.

*Method:* **spatial-frequency joint modality contrastive learning**
- **Contrastive learning** = pull same-modality features together, push different-modality features apart. No labels needed
- **Spatial-frequency joint** = do this in the normal image domain **and** the frequency domain, because modalities differ in textures that are clearer in frequency space *(same intuition as HFF-Net, Paper 1)*

**STAGE 2 — TD-Stage (Tumor-region Disentanglement)**

*Goal:* separate the knowledge about each tumor region.

*Method:* **affinity-guided dense knowledge distillation**
- A **teacher** network is trained on simple *binary* tasks — one region at a time. It becomes an expert on each region individually
- A **student** network does the full multi-region job in one go
- The student is trained to **match the teacher’s features**, inheriting the per-region expertise while still producing one unified output
- **“Affinity-guided”** = the matching uses *relationships between* features, not just raw values

```
MRI (some modalities missing)
   ↓
┌─ MD-STAGE ────────────────────────────┐
│  Contrastive learning in BOTH the     │
│  spatial and frequency domains        │
│  → "T1ce-ness" separated from         │
│    "FLAIR-ness" as distinct things    │
└───────────────────────────────────────┘
   ↓
┌─ TD-STAGE ────────────────────────────┐
│  TEACHER (expert on one region        │
│           at a time)                  │
│      │ distils knowledge              │
│      ▼                                │
│  STUDENT (does all regions at once)   │
└───────────────────────────────────────┘
   ↓
Segmentation
```

### 📊 Results

Extensive experiments on **BraTS 2018**, outperforming state-of-the-art in missing-modality situations. Code released.

### ⭐ Why it matters

**Most-cited paper in your set (107 citations)** and it established **disentanglement** as a legitimate strategy for missing modalities.

Its real contribution is a **reframing**. Everyone else asked *“how do we cope with less input?”* This paper asked *“what specifically did we lose, and can we isolate it?”* That’s a better question, and later papers built on it.

---

## ★ PAPER 5 — MBUNeXt

### *Multibranch Encoder Aggregation Network Based on Layer-Fusion Strategy*

**Qinghao Liu et al. · IEEE Transactions on Neural Networks and Learning Systems · 2025**`github.com/liuqinghao2018/MBUNeXt`

### ❓ What problem is it solving?

This one assumes you **do** have all four scans, and asks how to fuse them better. It names two specific failures:

**1. Information redundancy.** The four scans overlap heavily — they’re all pictures of the same brain. Naive fusion carries a lot of duplicated signal, which **dilutes** the genuinely useful parts.

**2. The tumor regions are wildly different sizes.** WT is large and smooth. ET is small and detailed. **A single fixed receptive field cannot serve both well.** Tune your kernels for the small detailed region and you can’t see the big one, and vice versa.

### 💡 The core idea

**Three modules, each fixing one specific thing:** filter out redundancy → fuse properly → cover multiple scales at once.

### ⚙️ How it works

| Module | Full name | What it does, plainly |
| --- | --- | --- |
| **MFA** | Multimodal Feature Attention | Uses attention plus **Gaussian modulation** to keep what’s *complementary* between modalities and discard what’s *duplicated*. It’s a redundancy filter |
| **MEA** | Multibranch Encoder Aggregation | Each modality gets its **own encoder branch**. MEA then finds correlations between branches and merges them using a **layer-fusion strategy** — combining at several depths, not just at the end |
| **LCS** | Large-kernel Convolution Skip | Skip connections built from **depthwise-separable convolutions with large kernels**, so the network sees several scales at once |

> **Why “large kernel + depthwise-separable”?**
> 
> - **Large kernel** = big receptive field = good for the large WT region
> - **Depthwise-separable** = splits one expensive convolution into two cheap ones = keeps parameter count low
> 
> You get wide-scale coverage without paying the full computational price. That combination is the practical trick.
> 

```
4 modalities → 4 separate encoder branches
   ↓
MFA: attention + Gaussian modulation
     → filters out duplicated information
   ↓
MEA: layer-fusion at multiple depths
     → merges the branches properly
   ↓
LCS skip connections:
     large-kernel depthwise-separable convs
     → handles the WT-is-huge / ET-is-tiny gap
   ↓
Segmentation
```

### 📊 Results

Tested on roughly **5,000 3-D brain scans**:

| Dataset | Mean Dice |
| --- | --- |
| BraTS 2019 | **85.84%** |
| BraTS 2021 | **91.11%** |
| **BraTS-Africa 2024** | **Performs well despite low image quality** |

### ⭐ Why it matters

**The BraTS-Africa result matters more than the headline Dice.**

That dataset exists for a specific reason: models are trained overwhelmingly on data from wealthy Western hospitals, with particular scanner brands and imaging protocols. **Nobody knew whether they work elsewhere.** BraTS-Africa is the stress test.

Holding up on low-quality images from a different population is a **robustness** claim, and that’s rarer and harder than an accuracy claim. Most papers in your set only ever test on standard BraTS.

The paper is also explicitly framed around **surgical robots and navigation systems** — aimed at deployment, not just leaderboard position.

# PART 2 — THE REMAINING 6 PAPERS

*Shorter treatment. Same structure.*

---

## PAPER 6 — Unified Hyper-Network

### *Learning Unified Hyper-Network for Multi-Modal MR Image Synthesis and Tumor Segmentation With Missing Modalities*

**Heran Yang et al. · IEEE TMI · 2023 · 49 citations**

### ❓ Problem

A **practical inefficiency**. Existing missing-modality methods do one of two bad things:
- Train **15 separate networks**, one per combination → huge computational burden
- Train **one fixed network** for all cases → a compromise, so sub-optimal for every individual case

### 💡 Core idea

**Two moves:**

**(a) Generate the missing scan first, then segment normally.** If T1ce is missing, synthesise a plausible T1ce and proceed.

**(b) Use a hypernetwork so one network adapts to any combination.**

> **What’s a hypernetwork?** A network that **generates another network’s weights**.
> 
> 
> **Analogy:** instead of owning 15 different tools, you own one **3D printer** that prints whichever tool you need. Store the printer, not 15 tools.
> 

### ⚙️ How it works

Split each modality’s features into **common** (shared anatomy — it’s the same brain) and **modality-specific** (what makes T1ce look like T1ce).

| Component | Job |
| --- | --- |
| **Shared hyper-encoder** | Embeds whichever modality is available; its weights are generated based on which modality it’s handling |
| **Graph-attention fusion** | Treats modalities as **nodes in a graph** and learns which should influence which |
| **Shared hyper-decoder** | Reconstructs the missing images |
| **Adversarial common feature constraint** | A discriminator forces the “common” features from different modalities to genuinely land in the *same* space — otherwise “common” is common in name only |
| **Hypernet modulation module** | ⚠ Decides how much to trust **real** scans vs **synthesised** ones. Synthetic data is less reliable, so weight it lower |

### 📊 Results

Synthesises reasonable multi-modal MR images **and** achieves state-of-the-art segmentation with missing modalities.

### ⭐ Why it matters — and the criticism

It’s the **synthesis** answer, done carefully. The modulation module is the honest part — most synthesis papers treat generated data as if it were real.

⚠ **But generative models can hallucinate structures that don’t exist.** In a diagnostic setting that’s genuinely dangerous. This risk is inherent to the approach, and it’s exactly why the field has largely shifted toward masking (Paper 3) instead.

---

## PAPER 7 — M²SegMamba

### *Mamba-Based Incomplete Multimodal Learning for Brain Tumor Segmentation With Few Samples*

**Xinyue Zhang et al. · IEEE JBHI · 2025**

### ❓ Problem

**Two problems at once** — that’s what makes it distinctive:
1. **Missing modalities**
2. **Very few labelled samples** — because a real hospital might have 50 annotated cases, not 1,251

Most missing-modality papers still quietly assume plenty of labelled data. This one doesn’t.

### 💡 Core idea

Combine **self-supervised learning** (survive few labels) + **Mamba** (handle 3D cheaply) + **cross-modal interaction** (handle missing inputs).

> **Why Mamba?** Attention costs **O(n²)**. A 128×128×128 volume has ~2 million voxels — completely unaffordable. Mamba gets similar long-range modelling at **O(n)**. That’s why state space models arrived in 3D medical imaging around 2024.
> 

### ⚙️ How it works

Trained two ways simultaneously:
- **Supervised** on the few labelled cases
- **Self-supervised** via **MAE** — hide chunks of the volume, reconstruct them. No labels needed, so unlabelled scans become usable

| Piece | What it does |
| --- | --- |
| **Multi-traversal inter-modal interaction** | ⚠ Mamba reads a **sequence**, so it needs an order. But a 3D volume has **no natural order** — do you scan left-right? top-down? front-back? Solution: scan along **multiple paths** and combine |
| **Cross-modal interaction** | Lets modalities exchange information during encoding |
| **TSmamba blocks in skip connections** | Mamba blocks placed **inside the U-Net skip connections**, so the detail passed from encoder to decoder is also globally contextualised, not just raw local features |

### ⭐ Why it matters

Closest to real hospital conditions: incomplete scans **and** few annotations, together.

It also illustrates a **genuine Mamba weakness** worth knowing: sequence models need an ordering, and 3D volumes don’t have one. Everything else is engineering around that.

---

## PAPER 8 — MMTSeg

### *A Multimodal Feature Distillation With Mamba-Transformer Network for Brain Tumor Segmentation With Incomplete Modalities*

**Ming Kang et al. · IEEE Transactions on Artificial Intelligence · 2024**

### ❓ Problem

Missing modalities again, plus two additions:
- **Attention is too expensive** on 3D volumes
- **Boundaries are under-used** — and boundaries are exactly where segmentation errors concentrate

### 💡 Core idea

Three things combined: **knowledge distillation** + **Mamba** (linear cost) + a **boundary-specific loss**.

### ⚙️ How it works

| Module | What it does |
| --- | --- |
| **Multimodal feature distillation** | A **teacher** trained on all 4 modalities teaches a **student** that only sees some. The student learns to produce teacher-quality features from incomplete input |
| **Unimodal feature enhancement** | Strengthen each modality’s own representation *before* fusing, so even a single-modality input carries strong signal |
| **Cross-modal fusion** | Combine whatever is present |

All on a **Mamba-Transformer hybrid** backbone, trained with a **boundary-wise loss** that punishes errors near tumor edges more heavily.

```
Full modalities ──► TEACHER (strong)
                        │ distil knowledge
Missing modalities ─► STUDENT
   ↓
Unimodal enhancement → Cross-modal fusion
   ↓
Mamba-Transformer backbone (linear-cost global context)
   ↓
Boundary-wise loss → sharper edges
```

### ⭐ Why it matters

Represents the **third architecture wave**: CNN → Transformer → **Mamba**.

Useful context: independent comparisons find Mamba variants are strong at **detecting** small lesions but not clearly better at **outlining** them than a well-tuned CNN. So Mamba isn’t a replacement — it’s good at a different part of the job.

---

## PAPER 9 — Medical Transformer

### *Universal Encoder for 3-D Brain MRI Analysis*

**Eunji Jun et al. · IEEE TNNLS · 2023**

### ❓ Problem

**Labelled 3D medical data is desperately scarce.** Annotating one scan means an expert tracing across ~150 slices — hours per case, specialist only. BraTS 2021 has ~1,251 cases; ImageNet has 1.2 million.

So how do you get transfer-learning benefits when there’s no big labelled 3D medical dataset to pretrain on?

### 💡 Core idea

**(a) Treat a 3D volume as sequences of 2D slices — from three viewing directions.**

> 3D convolutions and 3D transformers are hugely expensive. Slicing lets you use cheap 2D machinery. But slicing along one axis loses information.
> 
> 
> **Fix: use all three anatomical planes** — axial (top-down), coronal (front-back), sagittal (side-to-side) — and combine. You recover most of the 3D relationships at a fraction of the cost.
> 
> **Analogy:** you can’t fully describe a building from one photo. But a front view, a side view and a top view together tell you almost everything.
> 

**(b) Pretrain with self-supervision on large-scale HEALTHY brain MRI.** The proxy task is **masked encoding vector prediction** — hide part of the representation, predict it. Same principle as BERT. **No labels needed**, so huge unlabelled healthy datasets become usable.

### 📊 Results

- Beat state-of-the-art transfer learning methods
- **~92% fewer parameters** for classification/regression, **~97% fewer** for segmentation
- Worked well **even using only partial training samples** — which is the actual point
- One encoder served **three different task types**: disease diagnosis, brain age prediction, tumor segmentation

### ⭐ Why it matters

Pretraining on **healthy** brains is the smart insight: **to recognise abnormality you first need a strong model of normality**, and healthy scans are far more abundant.

The ~97% parameter reduction also matters for anyone deploying on constrained hardware.

---

## PAPER 10 — SwinPA-Net ⚠ *not a brain tumor paper*

### *Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network*

**Hao Du et al. · IEEE TNNLS · 2022 · 138 citations**

> ⚠ **Tested on polyps, skin lesions and laparoscopic images — not brain tumors.** Cite it for **methodology**, not domain results.
> 

### ❓ Problem

Two general medical-imaging difficulties:
1. **Huge variation in lesion size and type**
2. **Lesions look like their surroundings** — similar shape and colour, so very low contrast

### 💡 Core idea

Add two aggregation modules to a Swin Transformer backbone.

> **Swin recap:** a transformer that computes attention only **inside small windows**, then **shifts** the windows between layers so information crosses boundaries. Global reach at linear cost instead of quadratic.
> 

### ⚙️ How it works

| Module | What it does |
| --- | --- |
| **DMC** — Dense Multiplicative Connection | Fuses multi-scale features by **multiplying** rather than adding. ⚠ **Why that matters:** multiplication kills anything weak in *any* scale, so **background noise gets suppressed instead of accumulating.** Addition would let noise pile up |
| **LPA** — Local Pyramid Attention | Merges **global** and **local** attention to steer the network toward the region of interest. Tackles the low-contrast problem |

### 📊 Results

Beat the second-best method by **1.68% / 0.8% / 1.2%** mean Dice on polyp, skin lesion and laparoscopic segmentation.

### ⭐ Why it’s in your set

The **multiplicative-fusion trick for noise suppression** is genuinely transferable to brain MRI. That’s the takeaway — the mechanism, not the results.

---

## PAPER 11 — DCTC-Net ⚠ *not a brain tumor paper*

### *Dual-Branch Cross-Fusion Transformer–CNN Architecture*

**Rui Sun · IEEE TNNLS · 2025**

> ⚠ **General medical image segmentation.** Cite for architecture design, not brain tumor results.
> 

### ❓ Problem

Hybrid CNN+Transformer models are everywhere, but the paper identifies **two flaws in how they’re normally built**:

1. **The CNN branch uses vanilla convolution** — a fixed square kernel, with the **same weights applied to every input**. It can’t adapt to what it’s actually looking at.
2. **The Transformer branch uses ordinary self-attention** — which only models **spatial** relationships and **ignores channel and cross-dimensional** ones. Bad for medical images with cluttered backgrounds.

### 💡 Core idea

Fix both branches: make the convolution **adaptive**, make the attention **multi-dimensional**.

### ⚙️ How it works

**CNN branch → DDConv (Dynamic Deformable Convolution)**
- **Deformable** = the sampling grid **bends** to follow object shape instead of staying a rigid square
- **Dynamic** = kernel weights are **generated per input** rather than shared across all inputs
- Together: it adapts both *where it looks* and *how it filters*, per image

> **Analogy:** a normal convolution is a rigid cookie cutter — same shape every time. A deformable one is a flexible loop of wire you can bend around whatever you’re tracing.
> 

**Transformer branch → (S)W-ACAM**
- *(Shifted)-Window Adaptive Complementary Attention Module*
- Attends across **spatial + channel + cross-dimensional** relationships
- Plus **compact convolutional projection** to cut cost

### 📊 Results

Beat state-of-the-art CNN and Transformer networks with **fewer parameters, lower compute, and no pretraining required.**

> ⭐ That last point is a real practical advantage — most transformer methods need heavy pretraining to work at all.
> 

# PART 3 — PUTTING IT TOGETHER

## Master table

| # | Paper | Journal · Year | What it’s really about | The one-line novelty |
| --- | --- | --- | --- | --- |
| **1** | **HFF-Net** | TMI · 2025 | Better ET accuracy | Splits the image into frequencies, like an audio equaliser |
| **2** | **MoME** | TMI · 2025 | One model for everything | Mixture of experts, with curriculum learning to stop experts becoming clones |
| **3** | **M²FTrans** | JBHI · 2023 | Missing scans | Tells attention explicitly what’s absent; never invents data |
| **4** | **D²-Net** | TMI · 2022 | Missing scans | Separates modality info from region info instead of blending |
| **5** | **MBUNeXt** | TNNLS · 2025 | Better fusion | Filters redundancy; only paper tested on a different population |
| 6 | Hyper-Network | TMI · 2023 | Missing scans | One network generates weights for all 15 cases; synthesises missing scans |
| 7 | M²SegMamba | JBHI · 2025 | Missing scans + few labels | Mamba inside skip connections; multi-path 3D scanning |
| 8 | MMTSeg | TAI · 2024 | Missing scans | Teacher-student distillation on a Mamba backbone + boundary loss |
| 9 | Medical Transformer | TNNLS · 2023 | Not enough labels | Three viewing planes + self-supervised pretraining on healthy brains |
| 10 | SwinPA-Net ⚠ | TNNLS · 2022 | General backbone | Multiplicative fusion suppresses background noise |
| 11 | DCTC-Net ⚠ | TNNLS · 2025 | General backbone | Fixes both halves of the standard CNN+Transformer hybrid |

## Your set’s structure

**5 of your 11 papers are about missing modalities.** That’s your research theme, whether you planned it or not.

Even better — those five use **four different strategies** for the same problem. Being able to lay this out is the most valuable thing in your whole reading list:

| Strategy | The idea | Papers | Trade-off |
| --- | --- | --- | --- |
| **Disentangle** | Keep the ingredients separate, so you know what you lost | D²-Net | Complicated two-stage training |
| **Synthesise** | Generate the missing scan, then segment normally | Hyper-Network | ⚠ Can hallucinate — risky for diagnosis |
| **Mask** | Just tell attention what’s absent | M²FTrans | Safest — never invents anything |
| **Distil** | Full-modality teacher trains a reduced-modality student | MMTSeg, M²SegMamba | Needs full-modality data to train the teacher |

## How the field evolved (your papers, chronologically)

```
2022 ─────────► 2023 ─────────► 2024 ─────────► 2025
CNN era         Transformer     Mamba era       New directions
                era             (attention
D²-Net          M²FTrans        too costly      HFF-Net (frequency)
SwinPA-Net      Hyper-Network   in 3D)          MoME (mixture of experts)
                Med.Transformer MMTSeg          MBUNeXt, M²SegMamba
                                                DCTC-Net
```

**What the field chased, year by year:**
- **2022** — how do we fuse four scans well?
- **2023** — global context (transformers); coping with absence
- **2024** — efficiency (Mamba), because attention is too expensive in 3D
- **2025** — generalisation, robustness, and new ways of representing the image

## 7 facts to have memorised

1. **T1ce = injected dye = the aggressive tumor.** FLAIR = swelling. The other two are structural.
2. **ET ⊂ TC ⊂ WT.** ET always scores lowest — that’s expected, not a failure.
3. **Two human doctors only agree at Dice 0.74–0.85.** Anything above 0.95 = broken measurement.
4. **4 modalities = 15 possible combinations.** That’s why one adaptive model beats 15 separate ones.
5. **Remove T1ce → ET collapses to ~0.60 for every method ever tried.** Information limit, not modelling limit.
6. **Mamba exists because attention costs O(n²)** and a 128³ volume has ~2 million voxels.
7. **SwinPA-Net and DCTC-Net are not brain tumor papers.** Cite them for methodology only.

## Gaps you could build research on

1. **The ET/T1ce ceiling.** Every method collapses to ~0.60 without T1ce. Nobody has beaten it. Even *characterising* the theoretical bound would be publishable.
2. **Frequency + missing modality has never been combined.** HFF-Net is nearly alone in the frequency domain, and five of your papers do missing modalities. **The combination is untried and both halves are in your reading list.**
3. **Mamba’s ordering problem in 3D.** M²SegMamba’s multi-traversal is one fix. It’s an open question with room for better answers.
4. **No shared efficiency benchmark.** Papers report parameters, or FLOPs, or inference time — never consistently, never on stated hardware. Speed claims can’t be compared.
5. **Almost nobody tests generalisation.** Only MBUNeXt tried a genuinely different population (BraTS-Africa). Everyone else tests on BraTS alone.