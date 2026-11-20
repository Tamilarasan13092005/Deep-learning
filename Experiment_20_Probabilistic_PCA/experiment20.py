"""
Experiment 20: Probabilistic PCA and Reconstruction Analysis
CO4 | Packages: NumPy, Scikit-learn, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

print("=" * 60)
print("Experiment 20: Probabilistic PCA and Reconstruction Analysis")
print("=" * 60)

np.random.seed(42)

# ── 1. Standard vs Probabilistic PCA ─────────────────────────
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

# Standard PCA
pca_std = PCA(n_components=20)
X_reduced_std = pca_std.fit_transform(X)
X_recon_std   = pca_std.inverse_transform(X_reduced_std)
mse_std = np.mean((X - X_recon_std)**2)

# Probabilistic (MLE) PCA: sklearn supports 'mle' for n_components
pca_mle = PCA(n_components='mle')
X_reduced_mle = pca_mle.fit_transform(X)
X_recon_mle   = pca_mle.inverse_transform(X_reduced_mle)
mse_mle = np.mean((X - X_recon_mle)**2)

print(f"\n[1] Standard PCA (20 components): MSE={mse_std:.4f}")
print(f"    Probabilistic PCA (MLE n={pca_mle.n_components_}): MSE={mse_mle:.4f}")

# ── 2. Noise Level Effect ─────────────────────────────────────
print("\n[2] Reconstruction MSE vs Noise Level")
noise_levels = [0, 0.1, 0.5, 1.0, 2.0]
n_pca = 20
mse_noisy = []
for noise in noise_levels:
    X_noisy = X + noise * np.random.randn(*X.shape)
    pca_n = PCA(n_components=n_pca)
    X_red = pca_n.fit_transform(X_noisy)
    X_rec = pca_n.inverse_transform(X_red)
    mse_noisy.append(np.mean((X - X_rec)**2))
    print(f"  noise={noise:.1f}: MSE={mse_noisy[-1]:.4f}")

# ── 3. Reconstruction at Different Component Counts ───────────
print("\n[3] Reconstruction Quality vs # Components")
comp_counts = [1, 2, 5, 10, 20, 32, 40, 64]
recon_mses = []
for nc in comp_counts:
    pca_nc = PCA(n_components=min(nc, X.shape[1]))
    X_r = pca_nc.fit_transform(X)
    X_rec = pca_nc.inverse_transform(X_r)
    recon_mses.append(np.mean((X - X_rec)**2))
    print(f"  PCA({nc:3d}): MSE={recon_mses[-1]:.4f}")

# ── 4. Manual PPCA (EM-based) ─────────────────────────────────
class PPCA:
    """Probabilistic PCA via EM algorithm."""
    def __init__(self, n_components, n_iter=50):
        self.q = n_components
        self.n_iter = n_iter

    def fit(self, X):
        n, d = X.shape
        self.mu = X.mean(0)
        X_c = X - self.mu
        W = np.random.randn(d, self.q) * 0.1
        sigma2 = 1.0

        for _ in range(self.n_iter):
            # E-step
            M = W.T @ W + sigma2 * np.eye(self.q)
            M_inv = np.linalg.inv(M)
            Z = X_c @ W @ M_inv

            # M-step
            W = (X_c.T @ Z) @ np.linalg.inv(Z.T @ Z + sigma2 * M_inv * n)
            sigma2 = (np.sum((X_c - Z @ W.T)**2) + sigma2 * np.trace(M_inv @ W.T @ W) * n) / (n*d)

        self.W = W
        self.sigma2 = sigma2
        return self

    def transform(self, X):
        X_c = X - self.mu
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.q)
        return X_c @ self.W @ np.linalg.inv(M)

    def inverse_transform(self, Z):
        return Z @ self.W.T + self.mu

ppca = PPCA(n_components=10, n_iter=100)
ppca.fit(X)
Z = ppca.transform(X)
X_recon_ppca = ppca.inverse_transform(Z)
mse_ppca = np.mean((X - X_recon_ppca)**2)
print(f"\n[4] Manual PPCA (10 components): MSE={mse_ppca:.4f}, sigma2={ppca.sigma2:.4f}")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 5a. Reconstruction comparison (digit images)
ax = axes[0, 0]
n_show = 5
orig   = digits.data[:n_show].reshape(-1, 8, 8)
recon_std_imgs  = pca_std.inverse_transform(pca_std.transform(X))[:n_show]
recon_ppca_imgs = X_recon_ppca[:n_show]
fig2, ax2_arr = plt.subplots(3, n_show, figsize=(10, 5))
for i in range(n_show):
    ax2_arr[0, i].imshow(orig[i], cmap='gray'); ax2_arr[0, i].axis('off')
    ax2_arr[1, i].imshow(recon_std_imgs[i].reshape(8,8), cmap='gray'); ax2_arr[1, i].axis('off')
    ax2_arr[2, i].imshow(recon_ppca_imgs[i].reshape(8,8), cmap='gray'); ax2_arr[2, i].axis('off')
ax2_arr[0,0].set_ylabel('Original')
ax2_arr[1,0].set_ylabel('PCA Recon')
ax2_arr[2,0].set_ylabel('PPCA Recon')
fig2.suptitle('Digit Reconstruction Comparison', fontsize=12)
fig2.tight_layout()
fig2.savefig('output_exp20_digits.png', dpi=100, bbox_inches='tight')
plt.close(fig2)

ax.imshow(orig[0], cmap='gray'); ax.axis('off')
ax.set_title(f'Sample Digit\n(see output_exp20_digits.png)')

# 5b. MSE vs #Components
ax = axes[0, 1]
ax.plot(comp_counts, recon_mses, 'b-o', linewidth=2, markersize=8)
ax.axhline(mse_ppca, color='red', linestyle='--', label=f'PPCA(10): {mse_ppca:.3f}')
ax.set_title('Reconstruction MSE vs # Components')
ax.set_xlabel('# PCA Components'); ax.set_ylabel('MSE')
ax.legend(); ax.grid(True)

# 5c. Noise effect
ax = axes[0, 2]
ax.plot(noise_levels, mse_noisy, 'rs-', linewidth=2, markersize=8)
ax.set_title('Reconstruction MSE vs Noise Level')
ax.set_xlabel('Noise Std'); ax.set_ylabel('MSE'); ax.grid(True)
for x, y in zip(noise_levels, mse_noisy):
    ax.text(x, y+0.01, f'{y:.3f}', ha='center', fontsize=9)

# 5d. Latent space (2D) comparison
ax = axes[1, 0]
pca2 = PCA(n_components=2).fit(X)
Z2 = pca2.transform(X)
ppca2 = PPCA(n_components=2, n_iter=50)
ppca2.fit(X)
Z2_ppca = ppca2.transform(X)
ax.scatter(Z2[:, 0], Z2[:, 1], c=digits.target, cmap='tab10', s=8, alpha=0.5, label='PCA')
ax.set_title('PCA Latent Space (2D)'); ax.set_xlabel('Z1'); ax.set_ylabel('Z2'); ax.grid(True)

ax = axes[1, 1]
sc = ax.scatter(Z2_ppca[:, 0], Z2_ppca[:, 1], c=digits.target, cmap='tab10', s=8, alpha=0.5)
plt.colorbar(sc, ax=ax, label='Digit')
ax.set_title('PPCA Latent Space (2D)'); ax.set_xlabel('Z1'); ax.set_ylabel('Z2'); ax.grid(True)

# 5e. Noise variance (sigma^2) text box
ax = axes[1, 2]
ax.axis('off')
info = f"""
Probabilistic PCA (PPCA)

Model: X = Wz + μ + ε
  z ~ N(0, I_q)
  ε ~ N(0, σ²I_d)

EM Results (q=10):
  σ² (noise variance) = {ppca.sigma2:.4f}

Comparison:
  Standard PCA(20):   MSE = {mse_std:.4f}
  PPCA(MLE, q={pca_mle.n_components_}):   MSE = {mse_mle:.4f}
  Manual PPCA(10):    MSE = {mse_ppca:.4f}

Key insight:
  PPCA provides a probabilistic
  framework — σ² captures the
  unexplained (noise) variance.
"""
ax.text(0.05, 0.95, info, transform=ax.transAxes, va='top',
        fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('PPCA Summary')

plt.suptitle('Experiment 20: Probabilistic PCA & Reconstruction Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp20.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
