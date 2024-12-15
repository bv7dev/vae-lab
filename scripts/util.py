import torch
import matplotlib.pyplot as plt

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_image_grid(n_rows:int, n_cols:int, images: torch.Tensor, labels: torch.Tensor = None):
    """
    Plots a grid (n_rows * n_cols) of images with corresponding labels on top, if provided.
    """
    _, axes = plt.subplots(n_rows, n_cols)
    axes = axes.ravel()

    for i in range(n_rows*n_cols):
        image = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(image)
        axes[i].axis('off')
        if (labels is not None):
            axes[i].set_title(labels[i])

    plt.tight_layout()
    plt.show()