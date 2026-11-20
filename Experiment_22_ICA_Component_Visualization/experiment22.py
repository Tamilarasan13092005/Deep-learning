"""
Experiment 22: Independent Component Analysis and Component Visualization
CO4 | Packages: Scikit-learn, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("Experiment 22: Independent Component Analysis (ICA)")
print("=" * 60)

np.random.seed(42)

# ── 1. Cocktail Party Problem ─────────────────────────────────
n = 2000
t = np.linspace(0, 8, n)
S1 = np.sin(2 * np.pi * 1.5 * t)              # sine wave
S2 = np.sign(np.sin(2 * np.pi * 1.0 * t))     # square wave
S3 = np.random.laplace(0, 1, n)               # laplacian noise

S = np.column_stack([S1, S2, S3])
S = S / S.std(axis=0)

# Mixing matrix
A = np.array([[1.0, 0.5, 0.2],
              [0.3, 1.0, 0.4],
              [0.5, 0.2, 1.0]])

X_mixed = S @ A.T
print("\n[1] Source Signals + Mixing")
print(f"  Source shape: {S.shape}, Mixed shape: {X_mixed.shape}")
print(f"  Mixing matrix A:\n{A}")

# ── 2. Apply ICA ──────────────────────────────────────────────
X_scaled = StandardScaler().fit_transform(X_mixed)
ica = FastICA(n_components=3, random_state=42, max_iter=1000)
S_recovered = ica.fit_transform(X_scaled)

# Match recovered signals to sources (by correlation)
def match_sources(S_true, S_rec):
    corr_matrix = np.abs(np.corrcoef(S_true.T, S_rec.T)[:3, 3:])
    return corr_matrix

corr = match_sources(S, S_recovered)
print("\n[2] ICA Recovery Correlation Matrix")
print(np.round(corr, 3))

# ── 3. Compare with PCA ───────────────────────────────────────
pca = PCA(n_components=3)
S_pca = pca.fit_transform(X_scaled)
print("\n[3] PCA Recovery Correlation Matrix")
corr_pca = match_sources(S, S_pca)
print(np.round(corr_pca, 3))

# ── 4. 2D ICA Demo ───────────────────────────────────────────
n2d = 500
s1 = np.random.uniform(-1, 1, n2d)
s2 = np.random.uniform(-1, 1, n2d)
S_2d = np.column_stack([s1, s2])

A2 = np.array([[2, 1], [1, 2]])
X_2d = S_2d @ A2.T

ica2d = FastICA(n_components=2, random_state=42)
S_2d_rec = ica2d.fit_transform(X_2d)
pca2d = PCA(n_components=2)
S_2d_pca = pca2d.fit_transform(X_2d)

print("\n[4] 2D ICA Unmixing Matrix (W):")
print(np.round(ica2d.components_, 3))

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(15, 14))
n_plot = 500

# Row 0: Original sources
for i, (sig, name) in enumerate(zip([S1, S2, S3], ['Sine', 'Square', 'Laplacian'])):
    axes[0, i].plot(t[:n_plot], sig[:n_plot], linewidth=1.5)
    axes[0, i].set_title(f'Source {i+1}: {name}'); axes[0, i].grid(True)
axes[0, 0].set_ylabel('Original Sources', fontsize=10, fontweight='bold')

# Row 1: Mixed signals
for i in range(3):
    axes[1, i].plot(t[:n_plot], X_mixed[:n_plot, i], color='coral', linewidth=1.5)
    axes[1, i].set_title(f'Mixed Signal {i+1}'); axes[1, i].grid(True)
axes[1, 0].set_ylabel('Mixed Signals', fontsize=10, fontweight='bold')

# Row 2: ICA recovered
for i in range(3):
    axes[2, i].plot(t[:n_plot], S_recovered[:n_plot, i], color='green', linewidth=1.5)
    axes[2, i].set_title(f'ICA Recovered {i+1}'); axes[2, i].grid(True)
axes[2, 0].set_ylabel('ICA Recovered', fontsize=10, fontweight='bold')

# Row 3: 2D joint distributions
axes[3, 0].scatter(S_2d[:, 0], S_2d[:, 1], alpha=0.3, s=5, color='blue')
axes[3, 0].set_title('2D Original Sources'); axes[3, 0].grid(True)

axes[3, 1].scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, s=5, color='red')
axes[3, 1].set_title('2D Mixed Signals'); axes[3, 1].grid(True)

axes[3, 2].scatter(S_2d_rec[:, 0], S_2d_rec[:, 1], alpha=0.3, s=5, color='green')
axes[3, 2].set_title('2D ICA Recovered'); axes[3, 2].grid(True)

plt.suptitle('Experiment 22: Independent Component Analysis (ICA)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp22.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp22.png")
