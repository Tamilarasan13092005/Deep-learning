"""
Experiment 28: Deep Boltzmann Machine / Convolutional Boltzmann Machine Simulation
CO5 | Packages: NumPy, SciPy, TensorFlow, Keras, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

print("=" * 60)
print("Experiment 28: DBM / Convolutional Boltzmann Machine Simulation")
print("=" * 60)

np.random.seed(42)

# ── 1. Simplified DBM (2-layer) ───────────────────────────────
class DBM:
    """Simplified Deep Boltzmann Machine with 2 hidden layers."""

    def __init__(self, n_v, n_h1, n_h2, lr=0.01):
        self.W1 = np.random.randn(n_v,  n_h1) * 0.01
        self.W2 = np.random.randn(n_h1, n_h2) * 0.01
        self.bv = np.zeros(n_v)
        self.b1 = np.zeros(n_h1)
        self.b2 = np.zeros(n_h2)
        self.lr = lr
        self.losses = []

    def mean_field(self, v, n_iter=5):
        """Variational mean-field inference."""
        h1 = sigmoid(v @ self.W1 + self.b1)
        h2 = sigmoid(h1 @ self.W2 + self.b2)
        for _ in range(n_iter):
            h1 = sigmoid(v @ self.W1 + h2 @ self.W2.T + self.b1)
            h2 = sigmoid(h1 @ self.W2 + self.b2)
        return h1, h2

    def gibbs_step(self, v):
        h1_p = sigmoid(v @ self.W1 + self.b1)
        h1 = (np.random.rand(*h1_p.shape) < h1_p).astype(float)
        h2_p = sigmoid(h1 @ self.W2 + self.b2)
        h2 = (np.random.rand(*h2_p.shape) < h2_p).astype(float)
        v_p = sigmoid(h1 @ self.W1.T + self.bv)
        v_new = (np.random.rand(*v_p.shape) < v_p).astype(float)
        return v_new, h1, h2

    def fit(self, X, epochs=50, batch_size=32, k=1):
        n = len(X)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            ep_loss = []
            for i in range(0, n, batch_size):
                v0 = X[idx[i:i+batch_size]]
                # Positive phase (mean field)
                h1_pos, h2_pos = self.mean_field(v0)
                # Negative phase (k Gibbs steps)
                vk, h1k, h2k = v0.copy(), h1_pos.copy(), h2_pos.copy()
                for _ in range(k):
                    vk, h1k, h2k = self.gibbs_step(vk)
                m = len(v0)
                self.W1  += self.lr * (v0.T@h1_pos - vk.T@h1k) / m
                self.W2  += self.lr * (h1_pos.T@h2_pos - h1k.T@h2k) / m
                self.bv  += self.lr * (v0 - vk).mean(0)
                self.b1  += self.lr * (h1_pos - h1k).mean(0)
                self.b2  += self.lr * (h2_pos - h2k).mean(0)
                h1_recon, _ = self.mean_field(vk)
                v_recon = sigmoid(h1_recon @ self.W1.T + self.bv)
                ep_loss.append(np.mean((v0 - v_recon)**2))
            self.losses.append(np.mean(ep_loss))
            if (ep+1) % 25 == 0:
                print(f"  Epoch {ep+1}: loss={self.losses[-1]:.4f}")

    def reconstruct(self, v):
        h1, _ = self.mean_field(v)
        return sigmoid(h1 @ self.W1.T + self.bv)

# ── 2. Train DBM on Digits ────────────────────────────────────
print("\n[2] Training DBM on Digits (64→32→16)")
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

digits = load_digits()
X = MinMaxScaler().fit_transform(digits.data)
X_bin = (X > 0.5).astype(float)

dbm = DBM(n_v=64, n_h1=32, n_h2=16, lr=0.03)
dbm.fit(X_bin, epochs=100, batch_size=64, k=1)

X_recon_dbm = dbm.reconstruct(X_bin[:20])
print(f"\n[3] DBM Reconstruction MSE: {np.mean((X_bin[:20]-X_recon_dbm)**2):.4f}")

# ── 3. TF-based Convolutional Boltzmann Machine (approximation) ─
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)
    print("\n[4] Convolutional Autoencoder (approximating ConvBM)")

    # Convolutional Autoencoder as ConvBM approximation
    X_imgs = digits.data.reshape(-1, 8, 8, 1).astype('float32') / 16.0
    split = int(0.8*len(X_imgs))
    X_tr, X_te = X_imgs[:split], X_imgs[split:]

    encoder = keras.Sequential([
        layers.Input(shape=(8,8,1)),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(8, 3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
    ])
    decoder_input = keras.Input(shape=(16,))
    x = layers.Dense(4*4*8, activation='relu')(decoder_input)
    x = layers.Reshape((4,4,8))(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    decoder = keras.Model(decoder_input, x)

    autoencoder = keras.Model(encoder.input, decoder(encoder.output))
    autoencoder.compile(optimizer='adam', loss='mse')
    h_ae = autoencoder.fit(X_tr, X_tr, epochs=50, batch_size=32,
                            validation_split=0.1, verbose=0)
    mse_ae = autoencoder.evaluate(X_te, X_te, verbose=0)
    print(f"  ConvAE Test MSE: {mse_ae:.4f}")
    tf_ok = True

except ImportError:
    tf_ok = False
    print("  TensorFlow not installed — skipping ConvBM.")

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 4a. DBM training loss
ax = axes[0, 0]
ax.plot(dbm.losses, 'b-', linewidth=2)
ax.set_title('DBM Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.grid(True)

# 4b. DBM reconstruction
ax = axes[0, 1]
n_show = 5
fig2, ax2_arr = plt.subplots(2, n_show, figsize=(10,4))
for i in range(n_show):
    ax2_arr[0,i].imshow(X_bin[i].reshape(8,8), cmap='gray'); ax2_arr[0,i].axis('off')
    ax2_arr[1,i].imshow(X_recon_dbm[i].reshape(8,8), cmap='gray'); ax2_arr[1,i].axis('off')
ax2_arr[0,0].set_ylabel('Original'); ax2_arr[1,0].set_ylabel('DBM Recon')
fig2.suptitle('DBM Reconstruction', fontsize=11); fig2.tight_layout()
fig2.savefig('output_exp28_dbm_recon.png', dpi=100, bbox_inches='tight'); plt.close(fig2)
ax.axis('off')
ax.text(0.5,0.5,'DBM Reconstruction\n(see output_exp28_dbm_recon.png)',
        ha='center', va='center', transform=ax.transAxes, fontsize=11)
ax.set_title('DBM Digit Reconstruction')

# 4c. DBM W1 weights
ax = axes[0, 2]
grid = np.zeros((4*8, 8*8))
for idx in range(16):
    r, c = divmod(idx, 8)
    w = dbm.W1[:, idx].reshape(8,8)
    grid[r*8:(r+1)*8, c*8:(c+1)*8] = w
ax.imshow(grid, cmap='RdBu', aspect='auto')
ax.set_title('DBM W1 Filters (16/32 shown)'); ax.axis('off')

# 4d. ConvAE training (if TF)
ax = axes[1, 0]
if tf_ok:
    ax.plot(h_ae.history['loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(h_ae.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_title('ConvAE Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.legend(); ax.grid(True)
else:
    ax.axis('off')

# 4e. DBM architecture text
ax = axes[1, 1]
ax.axis('off')
dbm_text = """
Deep Boltzmann Machine (DBM)

Architecture:
  v (visible)  — 64 units
  h1 (hidden1) — 32 units
  h2 (hidden2) — 16 units

Energy Function:
  E(v,h1,h2) = -vᵀW1h1 - h1ᵀW2h2
               -aᵀv - b1ᵀh1 - b2ᵀh2

Inference (mean field):
  μ_h1 = σ(W1ᵀv + W2μ_h2 + b1)
  μ_h2 = σ(W2ᵀμ_h1 + b2)
  (iterate until convergence)

vs RBM: symmetric inter-layer coupling
  RBM: v↔h (1 hidden layer)
  DBM: v↔h1↔h2 (multiple hidden layers)
  — richer representations
  — harder training (mean-field needed)
"""
ax.text(0.02, 0.98, dbm_text, transform=ax.transAxes, va='top',
        fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('DBM Architecture & Theory')

# 4f. Hidden activation distribution
ax = axes[1, 2]
h1_acts, h2_acts = dbm.mean_field(X_bin)
ax.hist(h1_acts.flatten(), bins=30, alpha=0.6, color='blue', label='h1 activations', density=True)
ax.hist(h2_acts.flatten(), bins=30, alpha=0.6, color='red', label='h2 activations', density=True)
ax.set_title('Hidden Unit Activation Distributions')
ax.set_xlabel('Activation'); ax.set_ylabel('Density'); ax.legend(); ax.grid(True)

plt.suptitle('Experiment 28: DBM / Convolutional Boltzmann Machine Simulation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp28.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
