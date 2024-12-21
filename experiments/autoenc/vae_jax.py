import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Initialize random key
key = jax.random.PRNGKey(0)

# Define helper functions for dense layers
def dense(inputs, weights, biases):
    return jnp.dot(inputs, weights) + biases

def relu(x):
    return jnp.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

# Initialize weights and biases
def initialize_params(layer_sizes, key):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        weight = jax.random.normal(subkey, (layer_sizes[i], layer_sizes[i + 1])) * 0.01
        bias = jnp.zeros(layer_sizes[i + 1])
        params.append((weight, bias))
    return params

# Define the encoder
def encoder(params, x):
    for w, b in params[:-1]:
        x = relu(dense(x, w, b))
    mean, log_var = dense(x, params[-1][0], params[-1][1]), dense(x, params[-2][0], params[-2][1])
    return mean, log_var

# Define the decoder
def decoder(params, z):
    for w, b in params[:-1]:
        z = relu(dense(z, w, b))
    return sigmoid(dense(z, params[-1][0], params[-1][1]))

# Sampling from latent space
def sample_latent(mean, log_var, key):
    std = jnp.exp(0.5 * log_var)
    eps = jax.random.normal(key, mean.shape)
    return mean + eps * std

# VAE forward pass
def forward_pass(params, x, key):
    encoder_params, decoder_params = params
    mean, log_var = encoder(encoder_params, x)
    z = sample_latent(mean, log_var, key)
    x_reconstructed = decoder(decoder_params, z)
    return x_reconstructed, mean, log_var, z

# Loss function
def vae_loss(params, x, key):
    x_reconstructed, mean, log_var, _ = forward_pass(params, x, key)
    reconstruction_loss = jnp.mean((x - x_reconstructed) ** 2)
    kl_divergence = -0.5 * jnp.mean(1 + log_var - mean ** 2 - jnp.exp(log_var))
    return reconstruction_loss + kl_divergence

# Training step
@jax.jit
def train_step(params, x, key, learning_rate):
    loss, grads = jax.value_and_grad(vae_loss)(params, x, key)
    new_params = [(w - learning_rate * dw, b - learning_rate * db)
                  for (w, b), (dw, db) in zip(params[0], grads[0])]  # Encoder
    new_params.append([(w - learning_rate * dw, b - learning_rate * db)
                       for (w, b), (dw, db) in zip(params[1], grads[1])])  # Decoder
    return loss, new_params

# Latent space visualization
def visualize_latent_space(latents, labels):
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latents)
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

    # Initialize model parameters
    latent_dim = 2
    input_dim = 28 * 28
    encoder_layers = [input_dim, 128, latent_dim, latent_dim]
    decoder_layers = [latent_dim, 128, input_dim]

    key, subkey = jax.random.split(key)
    encoder_params = initialize_params(encoder_layers, subkey)
    key, subkey = jax.random.split(key)
    decoder_params = initialize_params(decoder_layers, subkey)

    params = (encoder_params, decoder_params)

    # Training loop
    learning_rate = 1e-3
    epochs = 10
    batch_size = 128

    for epoch in range(epochs):
        # Shuffle and batch the data
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, len(train_images))
        train_images = train_images[indices]

        for i in range(0, len(train_images), batch_size):
            batch = train_images[i:i + batch_size]
            key, subkey = jax.random.split(key)
            loss, params = train_step(params, batch, subkey, learning_rate)

        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Latent space visualization
    _, _, _, latent_vectors = forward_pass(params, test_images, key)
    visualize_latent_space(latent_vectors, test_labels)
