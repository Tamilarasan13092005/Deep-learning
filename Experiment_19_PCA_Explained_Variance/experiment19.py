"""
Experiment 19: PCA and Explained Variance Analysis
CO4 | Packages: NumPy, Scikit-learn, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, load_iris

print("=" * 60)
print("Experiment 19: PCA and Explained Variance Analysis")
print("=" * 60)

np.random.seed(42)

# ── 1. Iris Dataset ───────────────────────────────────────────
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

pca_iris = PCA()
pca_iris.fit(X_scaled)

print("\n[1] PCA on Iris Dataset (4 features)")
print(f"  Explained variance ratios: {np.round(pca_iris.explained_variance_ratio_, 4)}")
print(f"  Cumulative variance:       {np.round(np.cumsum(pca_iris.explained_variance_ratio_), 4)}")
print(f"  Components for 95% var:   {np.argmax(np.cumsum(pca_iris.explained_variance_ratio_)>=0.95)+1}")

# ── 2. Digits Dataset ─────────────────────────────────────────
digits = load_digits()
X_digits = StandardScaler().fit_transform(digits.data)
pca_digits = PCA(n_components=0.95)
X_pca95 = pca_digits.fit_transform(X_digits)
print(f"\n[2] PCA on Digits (64 features)")
print(f"  Original shape: {X_digits.shape}")
print(f"  PCA(95%) shape: {X_pca95.shape}")
print(f"  Components used: {pca_digits.n_components_}")

# ── 3. 2D Projection ──────────────────────────────────────────
pca2 = PCA(n_components=2)
X_iris_2d   = pca2.fit_transform(X_scaled)
X_digits_2d = pca2.fit_transform(X_digits)

# ── 4. Reconstruction ─────────────────────────────────────────
n_components_list = [5, 15, 30, 64]
reconstructions = {}
pca_full = PCA(n_components=64)
pca_full.fit(X_digits)
for n in n_components_list:
    pca_n = PCA(n_components=n)
    X_reduced = pca_n.fit_transform(X_digits)
    X_recon   = pca_n.inverse_transform(X_reduced)
    mse = np.mean((X_digits - X_recon)**2)
    reconstructions[n] = {'data': X_recon, 'mse': mse}
    print(f"  PCA({n:3d} components): reconstruction MSE = {mse:.4f}")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 5a. Iris 2D projection
ax = axes[0, 0]
colors = ['red', 'green', 'blue']
pca2_iris = PCA(n_components=2)
X_i2 = pca2_iris.fit_transform(X_scaled)
for i, (name, c) in enumerate(zip(iris.target_names, colors)):
    mask = y_iris == i
    ax.scatter(X_i2[mask, 0], X_i2[mask, 1], c=c, label=name, alpha=0.7, s=40)
ax.set_title('Iris: PCA 2D Projection'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(); ax.grid(True)

# 5b. Iris explained variance
ax = axes[0, 1]
ax.bar(range(1, 5), pca_iris.explained_variance_ratio_, color='steelblue', label='Individual')
ax.plot(range(1, 5), np.cumsum(pca_iris.explained_variance_ratio_),
        'ro-', linewidth=2, label='Cumulative')
ax.set_title('Iris: Explained Variance Ratio'); ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Ratio'); ax.legend(); ax.grid(True)

# 5c. Digits 2D projection
ax = axes[0, 2]
pca2_dg = PCA(n_components=2)
X_dg2 = pca2_dg.fit_transform(X_digits)
scatter = ax.scatter(X_dg2[:, 0], X_dg2[:, 1], c=digits.target,
                     cmap='tab10', s=10, alpha=0.5)
plt.colorbar(scatter, ax=ax, label='Digit Class')
ax.set_title('Digits: PCA 2D Projection'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

# 5d. Scree plot (digits)
ax = axes[1, 0]
pca_all = PCA()
pca_all.fit(X_digits)
ax.plot(range(1, 65), pca_all.explained_variance_ratio_, 'b-o', markersize=3, linewidth=1.5)
ax.axvline(pca_digits.n_components_, color='red', linestyle='--',
           label=f'95% @ {pca_digits.n_components_} PCs')
ax.set_title('Scree Plot (Digits)'); ax.set_xlabel('Component')
ax.set_ylabel('Explained Variance Ratio'); ax.legend(); ax.grid(True)

# 5e. Reconstruction quality
ax = axes[1, 1]
comps = list(reconstructions.keys())
mses  = [reconstructions[n]['mse'] for n in comps]
ax.plot(comps, mses, 'rs-', linewidth=2, markersize=8)
ax.set_title('Reconstruction MSE vs # Components'); ax.set_xlabel('# PCA Components')
ax.set_ylabel('MSE'); ax.grid(True)
for x, y in zip(comps, mses):
    ax.text(x, y+0.001, f'{y:.3f}', ha='center', fontsize=9)

# 5f. PCA components (loadings) heatmap for Iris
ax = axes[1, 2]
im = ax.imshow(pca_iris.components_, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels(iris.feature_names, rotation=20, fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels([f'PC{i+1}' for i in range(4)])
plt.colorbar(im, ax=ax)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{pca_iris.components_[i,j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_title('PCA Loadings (Iris)')

plt.suptitle('Experiment 19: PCA and Explained Variance Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp19.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp19.png")
