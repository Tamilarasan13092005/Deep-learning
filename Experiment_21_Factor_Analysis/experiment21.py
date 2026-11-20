"""
Experiment 21: Factor Analysis and Latent Factor Interpretation
CO4 | Packages: Scikit-learn, NumPy, Pandas, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("Experiment 21: Factor Analysis & Latent Factor Interpretation")
print("=" * 60)

np.random.seed(42)

# ── 1. Synthetic Data with Known Factor Structure ─────────────
n = 500
# 2 latent factors
F1 = np.random.randn(n)  # e.g., "general intelligence"
F2 = np.random.randn(n)  # e.g., "verbal ability"
noise = np.random.randn(n, 6) * 0.4

X_latent = np.column_stack([
    0.9*F1 + 0.1*F2 + noise[:,0],   # math score
    0.8*F1 + 0.2*F2 + noise[:,1],   # science score
    0.7*F1 + 0.3*F2 + noise[:,2],   # reasoning
    0.2*F1 + 0.9*F2 + noise[:,3],   # reading
    0.1*F1 + 0.85*F2 + noise[:,4],  # writing
    0.15*F1 + 0.8*F2 + noise[:,5],  # vocabulary
])

cols = ['Math', 'Science', 'Reasoning', 'Reading', 'Writing', 'Vocabulary']
df = pd.DataFrame(X_latent, columns=cols)
print("\n[1] Correlation Matrix")
print(df.corr().round(3))

# ── 2. Factor Analysis ────────────────────────────────────────
X_scaled = StandardScaler().fit_transform(X_latent)

for n_factors in [1, 2, 3]:
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa.fit(X_scaled)
    score = fa.score(X_scaled)
    print(f"\n[2] Factor Analysis (n_factors={n_factors})")
    print(f"  Log-likelihood: {score:.4f}")
    if n_factors <= 2:
        loadings = pd.DataFrame(
            fa.components_.T,
            index=cols,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )
        print(f"  Loadings:\n{loadings.round(3)}")

fa2 = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa2.fit_transform(X_scaled)

# ── 3. Compare with PCA ───────────────────────────────────────
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)
print(f"\n[3] PCA vs FA: Explained variance (PCA): {pca2.explained_variance_ratio_.round(3)}")

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 4a. Correlation heatmap
ax = axes[0, 0]
corr = df.corr()
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(6)); ax.set_yticklabels(cols, fontsize=9)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax)
ax.set_title('Correlation Heatmap')

# 4b. Factor loadings (heatmap)
ax = axes[0, 1]
loadings_mat = fa2.components_.T
im2 = ax.imshow(loadings_mat, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Factor 1\n(Quantitative)', 'Factor 2\n(Verbal)'])
ax.set_yticks(range(6)); ax.set_yticklabels(cols)
for i in range(6):
    for j in range(2):
        ax.text(j, i, f'{loadings_mat[i,j]:.3f}', ha='center', va='center', fontsize=10)
plt.colorbar(im2, ax=ax)
ax.set_title('Factor Loadings (n=2)')

# 4c. Scatter: Factor 1 vs Factor 2
ax = axes[0, 2]
ax.scatter(X_fa[:, 0], X_fa[:, 1], c=F1, cmap='coolwarm', alpha=0.5, s=15)
ax.set_title('Factor Analysis Latent Space'); ax.set_xlabel('Factor 1'); ax.set_ylabel('Factor 2')
ax.grid(True)

# 4d. FA vs PCA scatter
ax = axes[1, 0]
ax.scatter(X_fa[:, 0], X_pca[:, 0], alpha=0.3, s=15, color='steelblue')
ax.set_xlabel('FA Factor 1'); ax.set_ylabel('PCA PC1')
ax.set_title('FA Factor 1 vs PCA PC1'); ax.grid(True)
corr_val = np.corrcoef(X_fa[:, 0], X_pca[:, 0])[0, 1]
ax.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=ax.transAxes, fontsize=12)

# 4e. Loading comparison bar chart
ax = axes[1, 1]
x_pos = np.arange(6)
ax.bar(x_pos - 0.2, loadings_mat[:, 0], 0.4, label='Factor 1', color='steelblue')
ax.bar(x_pos + 0.2, loadings_mat[:, 1], 0.4, label='Factor 2', color='coral')
ax.set_xticks(x_pos); ax.set_xticklabels(cols, rotation=20, fontsize=9)
ax.set_title('Factor Loadings Comparison'); ax.legend(); ax.grid(True, axis='y')
ax.axhline(0, color='black', linewidth=0.8)

# 4f. Uniqueness (noise variance per variable)
ax = axes[1, 2]
noise_var = fa2.noise_variance_
ax.bar(cols, noise_var, color='salmon', edgecolor='black')
ax.set_title('Factor Analysis: Uniqueness (Noise Variance)')
ax.set_ylabel('Noise Variance'); ax.set_xticklabels(cols, rotation=20, fontsize=9)
for i, v in enumerate(noise_var):
    ax.text(i, v+0.005, f'{v:.3f}', ha='center', fontsize=9)
ax.grid(True, axis='y')

plt.suptitle('Experiment 21: Factor Analysis & Latent Factor Interpretation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp21.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp21.png")
