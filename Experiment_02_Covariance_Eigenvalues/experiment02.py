"""
Experiment 02: Covariance, Eigenvalues and Eigenvectors Analysis
CO1 | Packages: NumPy, Pandas, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 02: Covariance, Eigenvalues & Eigenvectors")
print("=" * 60)

np.random.seed(42)

# ── 1. Sample Dataset ─────────────────────────────────────────
n = 100
x1 = np.random.randn(n)
x2 = 0.8 * x1 + 0.2 * np.random.randn(n)
x3 = np.random.randn(n)
X = np.column_stack([x1, x2, x3])
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])

print("\n[1] Dataset Statistics")
print(df.describe().round(3))

# ── 2. Covariance Matrix ──────────────────────────────────────
cov_matrix = np.cov(X.T)
print("\n[2] Covariance Matrix")
print(np.round(cov_matrix, 4))

# ── 3. Correlation Matrix ─────────────────────────────────────
corr_matrix = df.corr()
print("\n[3] Correlation Matrix")
print(corr_matrix.round(4))

# ── 4. Eigenvalues & Eigenvectors ─────────────────────────────
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\n[4] Eigenvalues")
print(np.round(eigenvalues, 4))
print("\n[4] Eigenvectors (columns)")
print(np.round(eigenvectors, 4))

# Explained variance ratio
total_var = eigenvalues.sum()
explained_ratio = eigenvalues / total_var
print("\n[4] Explained Variance Ratio")
for i, (ev, er) in enumerate(zip(eigenvalues, explained_ratio)):
    print(f"  PC{i+1}: eigenvalue={ev:.4f}, explained={er*100:.2f}%")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a. Scatter x1 vs x2 with eigenvectors
ax = axes[0, 0]
ax.scatter(x1, x2, alpha=0.5, color='steelblue', s=20)
origin = [x1.mean(), x2.mean()]
small_cov = np.cov(np.stack([x1, x2]))
evals2, evecs2 = np.linalg.eig(small_cov)
for ev, vec in zip(evals2, evecs2.T):
    ax.quiver(*origin, vec[0]*ev, vec[1]*ev, scale=1,
              scale_units='xy', angles='xy', color='red', width=0.008)
ax.set_title('Scatter with Eigenvectors (x1 vs x2)'); ax.grid(True)
ax.set_xlabel('Feature1'); ax.set_ylabel('Feature2')

# 5b. Covariance Heatmap
ax = axes[0, 1]
im = ax.imshow(cov_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im, ax=ax)
labels = ['F1', 'F2', 'F3']
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cov_matrix[i,j]:.2f}', ha='center', va='center')
ax.set_title('Covariance Matrix Heatmap')

# 5c. Eigenvalue bar chart
ax = axes[1, 0]
ax.bar([f'PC{i+1}' for i in range(len(eigenvalues))], eigenvalues, color='coral')
ax.set_title('Eigenvalues (Variance Explained)'); ax.set_ylabel('Eigenvalue')
for i, v in enumerate(eigenvalues):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

# 5d. Cumulative explained variance
ax = axes[1, 1]
sorted_ratios = np.sort(explained_ratio)[::-1]
cumulative = np.cumsum(sorted_ratios)
ax.plot(range(1, len(cumulative)+1), cumulative * 100, 'bo-', linewidth=2)
ax.axhline(90, color='red', linestyle='--', label='90% threshold')
ax.set_title('Cumulative Explained Variance')
ax.set_xlabel('Number of Components'); ax.set_ylabel('Cumulative Variance (%)')
ax.legend(); ax.grid(True)

plt.suptitle('Experiment 02: Covariance, Eigenvalues & Eigenvectors', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp02.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp02.png")
