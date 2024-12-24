import torch
import matplotlib.pyplot as plt

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_image_grid(n_rows:int, n_cols:int, images: torch.Tensor, labels: torch.Tensor = None):
    """
    Plots a grid (n_rows * n_cols) of images with corresponding labels on top, if provided.
    """
    if n_rows == 1 and n_cols == 1:
        plt.imshow(images[0].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return

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