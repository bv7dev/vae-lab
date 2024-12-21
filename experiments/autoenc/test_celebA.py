import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.CenterCrop((120, 120)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CelebA dataset
dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# Get a batch of 10 images
images, labels = next(iter(dataloader))

print(labels[:,20])

def plot_image_grid(images: torch.Tensor, labels: torch.Tensor):
    """
    Plots a 5x2 grid of images with corresponding labels on top.

    Args:
        images (torch.Tensor): A batch of 10 images with shape (10, 3, 28, 28).
        labels (torch.Tensor): A tensor of shape (10, 40) corresponding to the labels.
    """
    # Ensure the input tensor has the correct shape
    assert images.shape == (10, 3, 64, 64), "Images tensor must have shape (10, 3, 28, 28)"
    assert labels.shape == (10, 40), "Labels tensor must have shape (10, 40)"

    # Convert labels to strings for display
    labels = ["".join(map(str, label.tolist())) for label in labels]

    # Create a figure with 5x2 grid
    fig, axes = plt.subplots(2, 5, figsize=(8, 10))
    axes = axes.ravel()  # Flatten the 2D array of axes into a 1D array for easy iteration

    for i in range(10):
        # Rearrange image shape from (3, 28, 28) to (28, 28, 3)
        image = images[i].permute(1, 2, 0).numpy()
        
        # Plot the image
        axes[i].imshow(image)
        axes[i].axis('off')
        
        # Add label as the title
        axes[i].set_title(labels[i][20])

    # Adjust spacing
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming you have a tensor `batch_images` of shape (10, 3, 28, 28)
# and a tensor of labels `batch_labels` of shape (10, 40)
# batch_images = torch.rand(10, 3, 28, 28)
# batch_labels = torch.randint(0, 2, (10, 40))
# plot_image_grid(batch_images, batch_labels)


plot_image_grid(images, labels)


# Denormalize the images
images = images / 2 + 0.5

# Create a grid of images
images_grid = utils.make_grid(images, nrow=5)

# Convert to numpy array for display
np_image = images_grid.numpy()

# Display images
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(np_image, (1, 2, 0)))
plt.axis('off')
plt.show()
