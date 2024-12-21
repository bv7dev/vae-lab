import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 8)  # Latent representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Output scaled to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Weight initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


# Data preparation
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model, loss, and optimizer
model = Autoencoder()
model.apply(weights_init)

criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    for images, _ in train_loader:
        # Flatten the images
        images = images.view(images.size(0), -1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Visualize a sample reconstruction
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            sample = images[0].view(1, -1)
            reconstructed = model(sample).view(28, 28).numpy()
            plt.imshow(reconstructed, cmap='gray')
            plt.title(f"Reconstruction Epoch {epoch+1}")
            plt.show()

# Testing and visualization
test_loader = data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_images, _ = next(iter(test_loader))

# Flatten and pass through the model
test_images_flat = test_images.view(test_images.size(0), -1)
with torch.no_grad():
    reconstructed = model(test_images_flat)

# Reshape for visualization
test_images = test_images.numpy()
reconstructed = reconstructed.view(-1, 1, 28, 28).numpy()

# Plot original and reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
for i in range(10):
    # Original images
    axes[0, i].imshow(test_images[i][0], cmap='gray')
    axes[0, i].axis('off')
    # Reconstructed images
    axes[1, i].imshow(reconstructed[i][0], cmap='gray')
    axes[1, i].axis('off')
plt.show()
