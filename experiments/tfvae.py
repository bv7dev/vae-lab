import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt

# Reparameterization trick
def sampling(args):
    mu, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * epsilon

# Hyperparameters
latent_dim = 2  # Size of the latent space
batch_size = 128
epochs = 1
learning_rate = 0.001

# Load and preprocess MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Encoder
input_img = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same')(input_img)
x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
mu = Dense(latent_dim, name="mu")(x)
log_var = Dense(latent_dim, name="log_var")(x)
z = Lambda(sampling, name="z")([mu, log_var])

# Decoder
decoder_input = Input(shape=(latent_dim,))
x = Dense(7 * 7 * 128, activation='relu')(decoder_input)
x = Reshape((7, 7, 128))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
output_img = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

# Models
encoder = Model(input_img, [mu, log_var, z], name="Encoder")
decoder = Model(decoder_input, output_img, name="Decoder")
vae_output = decoder(encoder(input_img)[2])
vae = Model(input_img, vae_output, name="VAE")

# Loss Function
reconstruction_loss_fn = MeanSquaredError()
def vae_loss(x, x_reconstructed, mu, log_var):
    reconstruction_loss = reconstruction_loss_fn(x, x_reconstructed)
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return reconstruction_loss + kl_loss

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Training Loop
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        mu, log_var, z = encoder(x)
        x_reconstructed = decoder(z)
        loss = vae_loss(x, x_reconstructed, mu, log_var)
    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    return loss

# Training
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step, x_batch in enumerate(tf.data.Dataset.from_tensor_slices(x_train).shuffle(1024).batch(batch_size)):
        loss = train_step(x_batch)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

# Visualize Latent Space
z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(8, 6))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=np.argmax(x_test, axis=1), cmap="viridis", s=1)
plt.colorbar()
plt.title("Latent Space Visualization")
plt.show()

# Generate Images from Latent Space
n = 15  # Number of images per axis
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)
figure = np.zeros((28 * n, 28 * n))

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="Greys_r")
plt.title("Generated Images from Latent Space")
plt.axis("off")
plt.show()