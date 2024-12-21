import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from jax import random
from flax import linen as nn
from sklearn.manifold import TSNE

# Define the VAE model
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)
        return mean, log_var


class Decoder(nn.Module):
    input_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(128)(z)
        z = nn.relu(z)
        x_reconstructed = nn.Dense(self.input_dim)(z)
        return nn.sigmoid(x_reconstructed)


class VAE(nn.Module):
    latent_dim: int
    input_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.input_dim)

    def __call__(self, x):
        mean, log_var = self.encoder(x)
        std = jnp.exp(0.5 * log_var)
        z = mean + std * random.normal(random.PRNGKey(0), mean.shape)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, log_var, z


# Reconstruction + KL Divergence Loss
def vae_loss(model, params, x, rng):
    x_reconstructed, mean, log_var, _ = model.apply(params, x, rngs={'dropout': rng})
    reconstruction_loss = jnp.mean(jnp.square(x - x_reconstructed))
    kl_loss = -0.5 * jnp.mean(1 + log_var - jnp.square(mean) - jnp.exp(log_var))
    return reconstruction_loss + kl_loss


# Training Step
@jax.jit
def train_step(optimizer, params, x, rng):
    def loss_fn(params):
        return vae_loss(vae, params, x, rng)

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, optimizer.state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


# Visualization with t-SNE
def visualize_latent_space(latent_vectors, labels):
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label="Class")
    plt.title("Latent Space Visualization with t-SNE")
    plt.show()


# Example Workflow
if __name__ == "__main__":
    # Data Preparation (e.g., MNIST)
    from tensorflow.keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(-1, 28 * 28) / 255.0
    test_images = test_images.reshape(-1, 28 * 28) / 255.0

    # Model and Optimizer Initialization
    latent_dim = 2
    input_dim = 28 * 28
    rng = random.PRNGKey(0)
    vae = VAE(latent_dim, input_dim)
    params = vae.init(rng, jnp.ones((1, input_dim)))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # Training Loop
    for epoch in range(10):
        for i in range(0, len(train_images), 128):
            batch = train_images[i : i + 128]
            params, opt_state = train_step(optimizer, params, batch, rng)

        # Logging
        loss = vae_loss(vae, params, train_images, rng)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Latent Space Visualization
    _, _, _, latent_vectors = vae.apply(params, test_images, rngs={'dropout': rng})
    visualize_latent_space(latent_vectors, test_labels)
