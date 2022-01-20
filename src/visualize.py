import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def draw_samples(points, labels, idx_to_class, n_rows, n_cols):
    plt.figure(figsize=(16, 16))
    n_images = n_rows * n_cols
    for i, (p, l) in enumerate(zip(points[:n_images], labels[:n_images])):
        ax = plt.subplot(n_rows, n_cols, i + 1, projection='3d')
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1)
        ax.set_title(f'class ({l.item()}): ' + idx_to_class[l.item()])
    plt.tight_layout()
    fig = plt.gcf()
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((int(height), int(width), 3))
    image = torch.tensor(image).permute(2, 0, 1)
    return image