"""
Experiment 29: VAE/GAN Sample Generation
CO5 | Packages: TensorFlow, Keras, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 29: VAE / GAN Sample Generation")
print("=" * 60)

np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import mnist

    tf.random.set_seed(42)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat  = X_test.reshape(-1, 784)

    LATENT_DIM = 16

    # ══════════════════════════════════════════════════════════
    # VAE
    # ══════════════════════════════════════════════════════════
    print("\n[1] Building VAE")

    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            eps = tf.random.normal(tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * eps

    # Encoder
    enc_input  = keras.Input(shape=(784,))
    x = layers.Dense(256, activation='relu')(enc_input)
    x = layers.Dense(128, activation='relu')(x)
    z_mean    = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(enc_input, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    dec_input = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(128, activation='relu')(dec_input)
    x = layers.Dense(256, activation='relu')(x)
    dec_out = layers.Dense(784, activation='sigmoid')(x)
    decoder = keras.Model(dec_input, dec_out, name='decoder')

    # VAE
    class VAE(keras.Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name='total_loss')

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                recon_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction), axis=-1
                    )
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                total_loss = recon_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            return {'loss': self.total_loss_tracker.result()}

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    print("  Training VAE...")
    h_vae = vae.fit(X_train_flat, epochs=30, batch_size=256, verbose=0)
    print(f"  Final VAE Loss: {h_vae.history['loss'][-1]:.2f}")

    # ══════════════════════════════════════════════════════════
    # GAN
    # ══════════════════════════════════════════════════════════
    print("\n[2] Building GAN")

    def build_generator(latent_dim):
        return keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(784, activation='sigmoid')
        ], name='generator')

    def build_discriminator():
        return keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(512, activation='leaky_relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='leaky_relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

    GAN_LATENT = 64
    generator     = build_generator(GAN_LATENT)
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    gan_input = keras.Input(shape=(GAN_LATENT,))
    discriminator.trainable = False
    gan_output = discriminator(generator(gan_input))
    gan = keras.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    print("  Training GAN...")
    d_losses, g_losses = [], []
    batch_size = 256
    n_batches = len(X_train_flat) // batch_size

    for epoch in range(20):
        d_loss_ep, g_loss_ep = [], []
        for _ in range(n_batches):
            # Train discriminator
            real = X_train_flat[np.random.randint(0, len(X_train_flat), batch_size)]
            noise = np.random.randn(batch_size, GAN_LATENT).astype(np.float32)
            fake = generator.predict(noise, verbose=0)
            X_comb = np.vstack([real, fake])
            y_comb = np.hstack([np.ones(batch_size)*0.9, np.zeros(batch_size)])
            d_loss = discriminator.train_on_batch(X_comb, y_comb)[0]
            # Train generator
            noise2 = np.random.randn(batch_size, GAN_LATENT).astype(np.float32)
            g_loss = gan.train_on_batch(noise2, np.ones(batch_size))
            d_loss_ep.append(d_loss); g_loss_ep.append(g_loss)
        d_losses.append(np.mean(d_loss_ep))
        g_losses.append(np.mean(g_loss_ep))
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: D_loss={d_losses[-1]:.4f}, G_loss={g_losses[-1]:.4f}")

    tf_ok = True

except ImportError:
    print("  TensorFlow not installed — using NumPy VAE demo.")
    tf_ok = False

    # Simple NumPy VAE proof of concept
    class SimpleVAE:
        def __init__(self, input_dim, latent_dim, lr=1e-3):
            self.input_dim = input_dim; self.latent_dim = latent_dim; self.lr = lr
            self.W_enc = np.random.randn(input_dim, 128) * 0.01
            self.W_mu  = np.random.randn(128, latent_dim) * 0.01
            self.W_lv  = np.random.randn(128, latent_dim) * 0.01
            self.W_dec = np.random.randn(latent_dim, input_dim) * 0.01
            self.losses = []

        def encode(self, x):
            h = np.maximum(0, x @ self.W_enc)
            return h @ self.W_mu, h @ self.W_lv

        def sample(self, mu, log_var):
            eps = np.random.randn(*mu.shape)
            return mu + np.exp(0.5*log_var) * eps

        def decode(self, z):
            return 1/(1+np.exp(-z @ self.W_dec))

        def forward(self, x):
            mu, lv = self.encode(x)
            z = self.sample(mu, lv)
            return self.decode(z), mu, lv

    n = 500; d = 10; latent = 2
    X_demo = np.random.randn(n, d)
    vae_demo = SimpleVAE(d, latent, lr=1e-3)
    losses = []
    for _ in range(100):
        idx = np.random.choice(n, 32)
        xb = X_demo[idx]
        x_rec, mu, lv = vae_demo.forward(xb)
        x_rec = np.clip(x_rec, 1e-7, 1-1e-7)
        x_norm = (xb - xb.min())/(xb.max()-xb.min()+1e-8)
        loss = -np.mean(x_norm*np.log(x_rec)+(1-x_norm)*np.log(1-x_rec))
        kl = -0.5*np.mean(1+lv-mu**2-np.exp(lv))
        losses.append(loss+kl)
    print(f"  NumPy VAE demo: final loss={losses[-1]:.4f}")
    h_vae = type('h', (), {'history': {'loss': losses}})()
    d_losses = g_losses = losses

# ── Visualisation ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

if tf_ok:
    # VAE loss
    ax = axes[0, 0]
    ax.plot(h_vae.history['loss'], 'b-', linewidth=2)
    ax.set_title('VAE Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('ELBO Loss'); ax.grid(True)

    # VAE generated samples
    ax = axes[0, 1]
    z_samples = np.random.randn(25, LATENT_DIM).astype(np.float32)
    gen_imgs = decoder.predict(z_samples, verbose=0).reshape(25, 28, 28)
    grid = np.block([[gen_imgs[i*5+j] for j in range(5)] for i in range(5)])
    ax.imshow(grid, cmap='gray'); ax.axis('off')
    ax.set_title('VAE Generated Samples (25)')

    # VAE latent space
    ax = axes[0, 2]
    z_means, _, _ = encoder.predict(X_test_flat[:1000], verbose=0)
    sc = ax.scatter(z_means[:,0], z_means[:,1], c=y_test[:1000], cmap='tab10', s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Digit')
    ax.set_title('VAE Latent Space (dim 0,1)'); ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]'); ax.grid(True)

    # GAN losses
    ax = axes[1, 0]
    ax.plot(d_losses, 'b-', label='D Loss', linewidth=2)
    ax.plot(g_losses, 'r-', label='G Loss', linewidth=2)
    ax.set_title('GAN Training Losses'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss'); ax.legend(); ax.grid(True)

    # GAN generated samples
    ax = axes[1, 1]
    noise = np.random.randn(25, GAN_LATENT).astype(np.float32)
    gan_imgs = generator.predict(noise, verbose=0).reshape(25, 28, 28)
    grid2 = np.block([[gan_imgs[i*5+j] for j in range(5)] for i in range(5)])
    ax.imshow(grid2, cmap='gray'); ax.axis('off')
    ax.set_title('GAN Generated Samples (25)')

else:
    ax = axes[0, 0]
    ax.plot(h_vae.history['loss'], 'b-', linewidth=2)
    ax.set_title('VAE Training Loss'); ax.grid(True)
    for axi in [axes[0,1],axes[0,2],axes[1,0],axes[1,1]]:
        axi.axis('off')

# Architecture text
ax = axes[1, 2]
ax.axis('off')
arch_text = """
VAE Architecture:
  Encoder: x → [μ, log σ²]
  Reparameterization: z = μ + σ·ε, ε~N(0,I)
  Decoder: z → x̂
  Loss: Recon + β·KL(q(z|x)||p(z))

GAN Architecture:
  Generator G:  z ~ N(0,I) → x̂
  Discriminator D: x → P(real)
  
  Min-Max Game:
    D: max E[log D(x)] + E[log(1-D(G(z)))]
    G: min E[log(1-D(G(z)))]
  
  G tries to fool D;
  D tries to distinguish real vs fake.

Key Differences:
  VAE: explicit latent space (encoder+decoder)
       smooth interpolation, blurry samples
  GAN: implicit distribution (generator only)
       sharp samples, training instability
"""
ax.text(0.02, 0.98, arch_text, transform=ax.transAxes, va='top',
        fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('VAE vs GAN Summary')

plt.suptitle('Experiment 29: VAE/GAN Sample Generation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp29.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
