import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Hyperparameters
epochs = 16
batch_size = 64
latent_dim = 64
learning_rate = 1e-3

# Data preparation (MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Convolutional Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=1)  # (28x28) -> (14x14)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1)  # (14x14) -> (7x7)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)  # (7x7) -> (4x4)
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(8 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(8 * 4 * 4, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

# # Convolutional Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 8 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1)  # (4x4) -> (7x7)
        self.deconv2 = nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)  # (7x7) -> (14x14)
        self.deconv3 = nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1)  # (14x14) -> (28x28)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 8, 4, 4)  # Reshape for ConvTranspose2d
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

# VAE Model
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

# Loss function
def vae_loss(x, reconstructed, z_mean, z_log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return (reconstruction_loss + kl_loss) / x.size(0)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
vae = VAE(encoder, decoder).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch in train_loader:
        x, _ = batch
        x = x.to(device)
        optimizer.zero_grad()
        reconstructed, z_mean, z_log_var = vae(x)
        loss = vae_loss(x, reconstructed, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}")

# Latent space visualization
vae.eval()
latent_vectors, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        z_mean, _ = vae.encoder(x)
        latent_vectors.append(z_mean.cpu())
        labels.append(y)
latent_vectors = torch.cat(latent_vectors)
labels = torch.cat(labels)

# Use t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab10", s=5)
plt.colorbar(scatter, label="Class")
plt.title("Latent Space Visualization with t-SNE")
plt.show()

# Reconstruction visualization
vae.eval()
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_dataset[i][0].numpy().squeeze(), cmap="gray")
    plt.axis("off")

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    reconstructed, _, _ = vae(test_dataset[i][0].unsqueeze(0).to(device))
    plt.imshow(reconstructed.cpu().detach().numpy().squeeze(), cmap="gray")
    plt.axis("off")
plt.show()
