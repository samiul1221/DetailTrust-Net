"""MRI, preprocessing and prediction figures with explicit RAS slice axes."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import MODALITIES


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
