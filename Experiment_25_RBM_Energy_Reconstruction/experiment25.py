"""
Experiment 25: Restricted Boltzmann Machine Energy and Reconstruction
CO5 | Packages: NumPy, SciPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

print("=" * 60)
print("Experiment 25: Restricted Boltzmann Machine (RBM)")
print("=" * 60)

np.random.seed(42)

# ── 1. RBM Implementation ─────────────────────────────────────
class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.01, k=1):
        self.n_v = n_visible
        self.n_h = n_hidden
        self.lr = lr
        self.k = k
        # Weights & biases
        self.W  = np.random.randn(n_visible, n_hidden) * 0.01
        self.bv = np.zeros(n_visible)
        self.bh = np.zeros(n_hidden)
        self.train_loss = []

    def sigmoid(self, x):
        return sigmoid(x)

    def sample_h_given_v(self, v):
        p_h = self.sigmoid(v @ self.W + self.bh)
        return (np.random.rand(*p_h.shape) < p_h).astype(float), p_h

    def sample_v_given_h(self, h):
        p_v = self.sigmoid(h @ self.W.T + self.bv)
        return (np.random.rand(*p_v.shape) < p_v).astype(float), p_v

    def contrastive_divergence(self, v0):
        # Positive phase
        h0_sample, h0_prob = self.sample_h_given_v(v0)
        # Negative phase (k-step Gibbs)
        hk, vk = h0_sample, v0
        for _ in range(self.k):
            vk, vk_prob = self.sample_v_given_h(hk)
            hk, hk_prob = self.sample_h_given_v(vk)
        # Gradient updates
        m = v0.shape[0]
        self.W  += self.lr * (v0.T @ h0_prob - vk.T @ hk_prob) / m
        self.bv += self.lr * (v0 - vk).mean(0)
        self.bh += self.lr * (h0_prob - hk_prob).mean(0)
        return vk, vk_prob

    def free_energy(self, v):
        wx_b = v @ self.W + self.bh
        return -v @ self.bv - np.sum(np.log(1 + np.exp(wx_b)), axis=1)

    def reconstruct(self, v):
        _, h_prob  = self.sample_h_given_v(v)
        _, v_recon = self.sample_v_given_h(h_prob)
        return v_recon

    def fit(self, X, epochs=50, batch_size=32):
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            recon_errors = []
            for i in range(0, n, batch_size):
                batch = X[idx[i:i+batch_size]]
                v_recon, _ = self.contrastive_divergence(batch)
                recon_errors.append(np.mean((batch - v_recon)**2))
            self.train_loss.append(np.mean(recon_errors))
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}: recon_error={self.train_loss[-1]:.4f}")

# ── 2. Train on MNIST-like Binary Data ────────────────────────
print("\n[2] Training RBM on Binary Data")
from sklearn.datasets import load_digits

digits = load_digits()
X_bin = (digits.data > 8).astype(float)   # binarize
X_bin = X_bin / X_bin.max()

rbm = RBM(n_visible=64, n_hidden=32, lr=0.05, k=1)
rbm.fit(X_bin, epochs=100, batch_size=64)

# ── 3. Reconstruction Quality ─────────────────────────────────
X_recon = rbm.reconstruct(X_bin[:20])
recon_mse = np.mean((X_bin[:20] - X_recon)**2)
print(f"\n[3] Reconstruction MSE on 20 samples: {recon_mse:.4f}")

# ── 4. Free Energy ────────────────────────────────────────────
fe_train = rbm.free_energy(X_bin[:100])
print(f"\n[4] Free Energy Statistics")
print(f"  Mean: {fe_train.mean():.4f}")
print(f"  Std:  {fe_train.std():.4f}")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 5a. Training Loss
ax = axes[0, 0]
ax.plot(rbm.train_loss, 'b-', linewidth=2)
ax.set_title('RBM Training: Reconstruction Error'); ax.set_xlabel('Epoch')
ax.set_ylabel('MSE'); ax.grid(True)

# 5b. Original vs Reconstructed Digits
ax = axes[0, 1]
ax.axis('off')
n_show = 5
fig2, ax2_arr = plt.subplots(2, n_show, figsize=(10, 4))
for i in range(n_show):
    ax2_arr[0, i].imshow(X_bin[i].reshape(8,8), cmap='gray'); ax2_arr[0,i].axis('off')
    ax2_arr[1, i].imshow(X_recon[i].reshape(8,8), cmap='gray'); ax2_arr[1,i].axis('off')
ax2_arr[0,0].set_ylabel('Original', fontsize=10)
ax2_arr[1,0].set_ylabel('Reconstructed', fontsize=10)
fig2.suptitle('RBM Digit Reconstruction', fontsize=11)
fig2.tight_layout()
fig2.savefig('output_exp25_reconstruction.png', dpi=100, bbox_inches='tight')
plt.close(fig2)
ax.text(0.5, 0.5, 'See output_exp25_reconstruction.png\nfor digit reconstructions',
        ha='center', va='center', fontsize=12, transform=ax.transAxes)
ax.set_title('Digit Reconstructions')

# 5c. Weight Matrix Heatmap
ax = axes[0, 2]
im = ax.imshow(rbm.W, cmap='RdBu', aspect='auto')
ax.set_title('RBM Weight Matrix W (64 × 32)'); ax.set_xlabel('Hidden Units')
ax.set_ylabel('Visible Units'); plt.colorbar(im, ax=ax)

# 5d. Free Energy Distribution
ax = axes[1, 0]
ax.hist(fe_train, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax.set_title('Free Energy Distribution'); ax.set_xlabel('Free Energy')
ax.set_ylabel('Count'); ax.grid(True)

# 5e. Hidden Activations
ax = axes[1, 1]
_, h_probs = rbm.sample_h_given_v(X_bin[:50])
im2 = ax.imshow(h_probs.T, aspect='auto', cmap='hot')
ax.set_title('Hidden Unit Activations (50 samples)'); ax.set_xlabel('Sample')
ax.set_ylabel('Hidden Unit'); plt.colorbar(im2, ax=ax, label='P(h=1|v)')

# 5f. Energy Landscape (2D RBM toy)
ax = axes[1, 2]
v_space = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
W_toy = np.array([[0.5, -0.5], [-0.5, 0.5]])
bv_toy = np.array([0.1, -0.1])
bh_toy = np.array([0.2, -0.2])
energies = []
for v in v_space:
    wx_b = v @ W_toy + bh_toy
    fe = -(v @ bv_toy) - np.sum(np.log(1 + np.exp(wx_b)))
    energies.append(fe)
labels = ['v=[0,0]','v=[0,1]','v=[1,0]','v=[1,1]']
bars = ax.bar(labels, energies, color=['steelblue','coral','green','purple'])
ax.set_title('RBM Free Energy (Toy 2-bit)'); ax.set_ylabel('Free Energy')
for bar, val in zip(bars, energies):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}', ha='center', fontsize=10)
ax.grid(True, axis='y')

plt.suptitle('Experiment 25: RBM Energy and Reconstruction',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp25.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
