"""
Experiment 27: Deep Belief Network Representation Learning
CO5 | Packages: NumPy, SciPy, TensorFlow, Keras, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

print("=" * 60)
print("Experiment 27: Deep Belief Network (DBN) Representation Learning")
print("=" * 60)

np.random.seed(42)

# ── 1. Greedy Layer-Wise RBM Training ────────────────────────
class RBM_Layer:
    def __init__(self, n_v, n_h, lr=0.05, k=1):
        self.W  = np.random.randn(n_v, n_h) * 0.01
        self.bv = np.zeros(n_v)
        self.bh = np.zeros(n_h)
        self.lr = lr; self.k = k
        self.losses = []

    def p_h(self, v): return sigmoid(v @ self.W + self.bh)
    def p_v(self, h): return sigmoid(h @ self.W.T + self.bv)
    def sample(self, p): return (np.random.rand(*p.shape) < p).astype(float)

    def transform(self, v): return self.p_h(v)

    def fit(self, X, epochs=50, batch_size=64):
        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            ep_losses = []
            for i in range(0, len(X), batch_size):
                v0 = X[idx[i:i+batch_size]]
                h0 = self.p_h(v0); h0s = self.sample(h0)
                vk, hk = v0, h0s
                for _ in range(self.k):
                    vk_p = self.p_v(hk); vk = self.sample(vk_p)
                    hk   = self.p_h(vk)
                m = len(v0)
                self.W  += self.lr * (v0.T@h0 - vk.T@hk) / m
                self.bv += self.lr * (v0 - vk).mean(0)
                self.bh += self.lr * (h0 - hk).mean(0)
                ep_losses.append(np.mean((v0 - self.p_v(h0))**2))
            self.losses.append(np.mean(ep_losses))
        return self


class DBN:
    def __init__(self, layer_sizes, lr=0.05):
        self.rbms = []
        for i in range(len(layer_sizes)-1):
            rbm = RBM_Layer(layer_sizes[i], layer_sizes[i+1], lr=lr)
            self.rbms.append(rbm)
        self.losses_per_layer = []

    def fit(self, X, epochs=50):
        H = X.copy()
        for layer_idx, rbm in enumerate(self.rbms):
            print(f"  Training RBM Layer {layer_idx+1} ({rbm.W.shape[0]}→{rbm.W.shape[1]})...")
            rbm.fit(H, epochs=epochs)
            H = rbm.transform(H)
            self.losses_per_layer.append(rbm.losses[:])
            print(f"    Final loss: {rbm.losses[-1]:.4f}")
        return self

    def transform(self, X):
        H = X.copy()
        for rbm in self.rbms:
            H = rbm.transform(H)
        return H

    def reconstruct(self, X):
        H = self.transform(X)
        for rbm in reversed(self.rbms):
            H = rbm.p_v(H)
        return H


# ── 2. Train DBN on Digits ────────────────────────────────────
print("\n[2] Training DBN on Digits (64→32→16→8)")
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

digits = load_digits()
X = MinMaxScaler().fit_transform(digits.data)

dbn = DBN(layer_sizes=[64, 32, 16, 8], lr=0.05)
dbn.fit(X, epochs=80)

# ── 3. Reconstruction ─────────────────────────────────────────
X_recon = dbn.reconstruct(X[:20])
recon_mse = np.mean((X[:20] - X_recon)**2)
print(f"\n[3] DBN Reconstruction MSE: {recon_mse:.4f}")

# ── 4. TF Fine-tuning (optional) ──────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)

    # Use DBN features for classification
    X_feat = dbn.transform(X)
    y = digits.target
    split = int(0.8*len(X))
    X_tr_f, X_te_f = X_feat[:split], X_feat[split:]
    y_tr, y_te = y[:split], y[split:]

    clf = keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h_tf = clf.fit(X_tr_f, y_tr, epochs=50, verbose=0, validation_split=0.1)
    _, acc = clf.evaluate(X_te_f, y_te, verbose=0)
    print(f"\n[4] DBN Features → Classifier Accuracy: {acc*100:.1f}%")
    tf_ok = True

except ImportError:
    tf_ok = False
    print("\n[4] TensorFlow not installed — skipping fine-tuning.")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 5a. Layer-wise training losses
ax = axes[0, 0]
colors = ['blue', 'red', 'green']
for i, (losses, color) in enumerate(zip(dbn.losses_per_layer, colors)):
    ax.plot(losses, color=color, linewidth=2, label=f'RBM Layer {i+1}')
ax.set_title('DBN Greedy Layer-Wise Training'); ax.set_xlabel('Epoch')
ax.set_ylabel('Reconstruction Error'); ax.legend(); ax.grid(True)

# 5b. Original vs Reconstructed
ax = axes[0, 1]
fig2, ax2_arr = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    ax2_arr[0,i].imshow(X[i].reshape(8,8), cmap='gray'); ax2_arr[0,i].axis('off')
    ax2_arr[1,i].imshow(X_recon[i].reshape(8,8), cmap='gray'); ax2_arr[1,i].axis('off')
ax2_arr[0,0].set_ylabel('Original')
ax2_arr[1,0].set_ylabel('DBN Recon')
fig2.suptitle('DBN Reconstruction', fontsize=11)
fig2.tight_layout()
fig2.savefig('output_exp27_reconstruction.png', dpi=100, bbox_inches='tight')
plt.close(fig2)
ax.axis('off')
ax.text(0.5, 0.5, 'DBN Reconstruction\n(see output_exp27_reconstruction.png)',
        ha='center', va='center', transform=ax.transAxes, fontsize=11)
ax.set_title('Digit Reconstruction')

# 5c. Latent space (8-dim → 2D via PCA)
ax = axes[0, 2]
from sklearn.decomposition import PCA
X_lat = dbn.transform(X)
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_lat)
sc = ax.scatter(X_2d[:,0], X_2d[:,1], c=digits.target, cmap='tab10', s=15, alpha=0.6)
plt.colorbar(sc, ax=ax, label='Digit')
ax.set_title('DBN Latent Space (PCA 2D)'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True)

# 5d. Weight visualizations (RBM 1)
ax = axes[1, 0]
w1 = dbn.rbms[0].W  # shape (64, 32)
n_show_w = 16
grid = np.zeros((2*8, 8*8))
for idx in range(n_show_w):
    r, c = divmod(idx, 8)
    grid[r*8:(r+1)*8, c*8:(c+1)*8] = w1[:, idx].reshape(8,8)
ax.imshow(grid, cmap='RdBu', aspect='auto')
ax.set_title('RBM Layer 1 Weight Filters (16 of 32)'); ax.axis('off')

# 5e. DBN architecture diagram
ax = axes[1, 1]
ax.axis('off')
arch_text = """
Deep Belief Network (DBN) Architecture

Input (64)
  ↕  RBM Layer 1 (Contrastive Divergence)
Hidden 1 (32)
  ↕  RBM Layer 2 (CD)
Hidden 2 (16)
  ↕  RBM Layer 3 (CD)
Latent (8)
  ↓  Fine-tuning (backpropagation)
Softmax Output (10 classes)

Training Strategy:
  Phase 1: Greedy layer-wise pre-training
    Each RBM trained independently
  Phase 2: Fine-tune with labeled data
    Unrolled as autoencoder or classifier
"""
ax.text(0.02, 0.98, arch_text, transform=ax.transAxes, va='top',
        fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('DBN Architecture')

# 5f. Classification accuracy (if TF available)
ax = axes[1, 2]
if tf_ok:
    ax.plot(h_tf.history['accuracy'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(h_tf.history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    ax.set_title('Fine-tuning Classifier (DBN Features)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(True)
else:
    ax.axis('off')
    ax.text(0.5, 0.5, 'TF not installed\nSkipped fine-tuning',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

plt.suptitle('Experiment 27: Deep Belief Network Representation Learning',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp27.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
