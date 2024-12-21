# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %%
latent_dim = 10

# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 528),
            nn.LeakyReLU(),
            nn.Linear(528, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 336),
            nn.LeakyReLU(),
            nn.Linear(336, 288),
            nn.LeakyReLU(),
            # nn.Linear(288, 280),
            # nn.LeakyReLU(),
            # nn.Linear(280, 32),
            # nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(288, latent_dim)  # Mean of latent space
        self.fc_log_var = nn.Linear(288, latent_dim)  # Log-variance of latent space

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 288),
            nn.LeakyReLU(),
            # nn.Linear(32, 280),
            # nn.LeakyReLU(),
            # nn.Linear(280, 288),
            # nn.LeakyReLU(),
            nn.Linear(288, 336),
            nn.LeakyReLU(),
            nn.Linear(336, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 528),
            nn.LeakyReLU(),
            nn.Linear(528, 784),
            nn.Sigmoid()  # Output scaled to [0, 1]
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample epsilon from standard normal
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var


# Loss function (Reconstruction + KL Divergence)
def vae_loss(reconstructed, original, mu, log_var):
    # Reconstruction loss
    recon_loss = nn.MSELoss(reduction='sum')(reconstructed, original)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss

# %%
# Data preparation
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model, optimizer, and training setup
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# %%
# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.view(images.size(0), -1)

        # Forward pass
        reconstructed, mu, log_var = model(images)
        loss = vae_loss(reconstructed, images, mu, log_var)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.2f}")

# %%
# Visualize reconstructions
model.eval()
test_loader = data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_images, _ = next(iter(test_loader))
test_images_flat = test_images.view(test_images.size(0), -1)

with torch.no_grad():
    reconstructed, _, _ = model(test_images_flat)

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


# %%
# Random latent gen
with torch.no_grad():
    z = torch.randn(10, model.fc_mu.out_features)  # Random latent vectors
    generated = model.decoder(z).view(-1, 1, 28, 28).numpy()

# Plot generated images
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(generated[i][0], cmap='gray')
    axes[i].axis('off')
plt.show()

# %%
# Latent space vis

# Ensure the latent dimension is 2 for visualization
# assert model.fc_mu.out_features == 2, "Latent dimension must be 2 for 2D visualization!"

# Get latent space representation for the entire dataset
model.eval()
latents = []
labels = []

with torch.no_grad():
    for images, label in train_loader:
        images = images.view(images.size(0), -1)
        _, mu, _ = model(images)
        latents.append(mu)
        labels.append(label)

latents = torch.cat(latents).numpy()
labels = torch.cat(labels).numpy()

# Plot the latent space
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, label="Digit Label")
plt.title("Latent Space Visualization")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()

# %%
# Ensure the latent dimension is 3 for visualization
# assert model.fc_mu.out_features == 3, "Latent dimension must be 3 for 3D visualization!"

# Get latent space representation
model.eval()
latents = []
labels = []

with torch.no_grad():
    for images, label in train_loader:
        images = images.view(images.size(0), -1)
        _, mu, _ = model(images)
        latents.append(mu)
        labels.append(label)

latents = torch.cat(latents).numpy()
labels = torch.cat(labels).numpy()

# Plot the 3D latent space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], c=labels, cmap='tab10', s=10, alpha=0.7)
fig.colorbar(scatter, ax=ax, label="Digit Label")
ax.set_title("3D Latent Space Visualization")
ax.set_xlabel("Latent Dimension 1")
ax.set_ylabel("Latent Dimension 2")
ax.set_zlabel("Latent Dimension 3")
plt.show()



