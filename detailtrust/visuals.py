"""MRI, preprocessing and prediction figures with explicit RAS slice axes."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import MODALITIES
from .paper_figures import save_paper_figure
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_subject(images, labels, path, title="MRI and released segmentation"):
    """Middle superior-axis slice chosen without finding a favourable lesion."""
    index = images.shape[-1] // 2
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, name in enumerate(MODALITIES):
        axes[i].imshow(images[i, :, :, index].T, cmap="gray", origin="lower")
        axes[i].set_title(name.upper())
    axes[4].imshow(labels[:, :, index].T, cmap="tab10", vmin=0, vmax=3, origin="lower")
    axes[4].set_title("Internal classes 0–3")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_corrections(images, truth, output, path, modality=3):
    """The same signed colour scale shows actual logit changes, not gate magnitude."""
    index = images.shape[-1] // 2
    base = output["base"].argmax(0).numpy()
    final = output["logits"].argmax(0).numpy()
    removed = (output["logits"] - output["contributions"][modality]).argmax(0).numpy()
    contribution = output["contributions"][modality, 3].numpy()
    limit = max(float(np.quantile(np.abs(contribution), .99)), 1e-5)
    panels = [(images[modality], "MRI", "gray", None), (truth, "Reference", "tab10", 3),
              (base, "Base", "tab10", 3), (final, "Final", "tab10", 3),
              (contribution, "ET logit contribution", "coolwarm", limit),
              (removed, "This detail removed", "tab10", 3)]
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for ax, (array, name, cmap, maximum) in zip(axes, panels):
        kwargs = {} if maximum is None else dict(vmin=-maximum if cmap == "coolwarm" else 0, vmax=maximum)
        ax.imshow(array[:, :, index].T, cmap=cmap, origin="lower", **kwargs)
        ax.set_title(name); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_qualitative_paper(images, truth, output, directory, subject, spacing):
    """First prespecified validation patient; slice centres use reference tumour only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tumour = np.argwhere(truth > 0)
    centre = np.median(tumour, axis=0).astype(int) if len(tumour) else np.asarray(truth.shape)//2
    base, final = output['base'].argmax(0).numpy(), output['logits'].argmax(0).numpy()
    colours = ListedColormap(['#000000', '#CC79A7', '#E69F00', '#009E73'])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    source = dict(subject=subject, partition='validation', selection='first validation subject in saved split',
                  slice_selection='median reference tumour coordinates; volume centre when no tumour',
                  ras_indices=centre.tolist(), spacing_mm=list(spacing), classes={'0':'background','1':'NCR/NET','2':'edema','3':'ET'},
                  interpretation='Held-out illustration, not a representative-cohort claim or clinical explanation')
    native = {}
    fig, axes = plt.subplots(3, 4, figsize=(11, 9), layout='constrained')
    for row, (axis, label, xy) in enumerate(((2,'Axial',(0,1)), (1,'Coronal',(0,2)), (0,'Sagittal',(1,2)))):
        extent = (0, truth.shape[xy[0]]*spacing[xy[0]], 0, truth.shape[xy[1]]*spacing[xy[1]])
        for col, (array, title) in enumerate(((images[3], 'FLAIR'), (truth,'Reference'), (base,'Base'), (final,'Final'))):
            plane = np.take(array, centre[axis], axis=axis).T
            native[label+'_'+title] = plane
            kwargs = dict(cmap='gray') if col == 0 else dict(cmap=colours, norm=norm)
            im = axes[row,col].imshow(plane, origin='lower', extent=extent, interpolation='nearest', **kwargs)
            axes[row,col].set(title=title if row == 0 else '', xlabel='Distance (mm)')
            if col == 0:
                axes[row,col].set_ylabel(label+'; distance (mm)')
    bar = fig.colorbar(im, ax=axes[:,1:].ravel().tolist(), ticks=[0,1,2,3], shrink=.6)
    bar.ax.set_yticklabels(['background','NCR/NET','edema','ET'])
    artifacts = {'segmentation_views': save_paper_figure(fig, directory, subject+'_segmentation_views',
        'Three orthogonal views of the prespecified validation subject. Reference, base and final use identical class colours. Distances use voxel spacing; orientation is canonical RAS. Reference masks only select display slices, never inference features.', source)}
    index = centre[2]
    extent = (0, truth.shape[0]*spacing[0], 0, truth.shape[1]*spacing[1])
    for key, name, caption in (
        ('gates', 'spatial_gates', 'Window-averaged gate scores, not calibrated quality probabilities.'),
        ('contributions', 'et_contribution_maps', 'Actual additive ET logit contributions; positive adds ET evidence and negative subtracts it. Magnitude does not establish benefit.')):
        maps = output[key][:, 0 if key == 'gates' else 3, :, :, index].numpy()
        native[name] = maps
        bound = max(float(np.abs(maps).max()), 1e-8)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), layout='constrained')
        for ax, modality, plane in zip(axes, MODALITIES, maps):
            im = ax.imshow(plane.T, origin='lower', interpolation='nearest', extent=extent,
                           cmap='viridis' if key == 'gates' else 'RdBu_r',
                           vmin=0 if key == 'gates' else -bound, vmax=1 if key == 'gates' else bound)
            ax.set(title=modality.upper(), xlabel='R distance (mm)', ylabel='A distance (mm)')
        fig.colorbar(im, ax=axes, label='Gate score' if key == 'gates' else 'ET logit change', shrink=.7)
        artifacts[name] = save_paper_figure(fig, directory, subject+'_'+name,
            caption+' Same axial slice for all modalities; shared colour limits without percentile clipping.', source)
    np.savez_compressed(directory/(subject+'_display_source.npz'), **native)
    return artifacts
