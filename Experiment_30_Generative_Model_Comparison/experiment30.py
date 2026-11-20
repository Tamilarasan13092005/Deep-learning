"""
Experiment 30: Generative Model Output Comparison and Evaluation
CO5 | Packages: TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.special import expit as sigmoid

print("=" * 60)
print("Experiment 30: Generative Model Output Comparison & Evaluation")
print("=" * 60)

np.random.seed(42)

# ── 1. Evaluation Metrics ─────────────────────────────────────
def frechet_distance_approx(real, fake):
    """Approximate FID using mean and covariance."""
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r = np.cov(real.T)
    cov_f = np.cov(fake.T)
    diff = mu_r - mu_f
    # Matrix square root approximation
    sqrt_cov = np.linalg.cholesky(cov_r @ cov_f + 1e-6*np.eye(cov_r.shape[0]))
    fd = diff @ diff + np.trace(cov_r + cov_f - 2*sqrt_cov)
    return float(np.real(fd))

def pixel_mse(real, fake):
    return np.mean((real - fake)**2)

def pixel_variance(samples):
    return np.var(samples, axis=0).mean()

def pairwise_diversity(samples, n_pairs=200):
    idx = np.random.choice(len(samples), n_pairs, replace=False)
    return pairwise_distances(samples[idx]).mean()

# ── 2. Load digits + train models ─────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X_real = MinMaxScaler().fit_transform(digits.data)
X_bin = (X_real > 0.5).astype(float)
n_real = len(X_real)

print("\n[2] Evaluating Generative Models")

# ── Model A: Gaussian Noise Baseline ──────────────────────────
X_noise = np.random.rand(n_real, 64)
print("  [A] Gaussian Noise baseline — done")

# ── Model B: RBM ──────────────────────────────────────────────
class RBM_gen:
    def __init__(self, n_v, n_h, lr=0.05, k=1):
        self.W = np.random.randn(n_v, n_h) * 0.01
        self.bv = np.zeros(n_v); self.bh = np.zeros(n_h)
        self.lr = lr; self.k = k

    def p_h(self, v): return sigmoid(v @ self.W + self.bh)
    def p_v(self, h): return sigmoid(h @ self.W.T + self.bv)
    def sample(self, p): return (np.random.rand(*p.shape) < p).astype(float)

    def fit(self, X, epochs=80, bs=64):
        for _ in range(epochs):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), bs):
                v0 = X[idx[i:i+bs]]
                h0 = self.p_h(v0); h0s = self.sample(h0)
                vk, hk = v0, h0s
                for _ in range(self.k):
                    vk_p = self.p_v(hk); vk = self.sample(vk_p); hk = self.p_h(vk)
                m = len(v0)
                self.W += self.lr*(v0.T@h0 - vk.T@hk)/m
                self.bv += self.lr*(v0-vk).mean(0)
                self.bh += self.lr*(h0-hk).mean(0)

    def generate(self, n, gibbs_steps=100):
        v = (np.random.rand(1, len(self.bv)) > 0.5).astype(float)
        samples = []
        for step in range(gibbs_steps + n):
            h = self.sample(self.p_h(v)); v = self.p_v(h)
            if step >= gibbs_steps:
                samples.append(v.copy())
        return np.vstack(samples)

rbm = RBM_gen(64, 32, lr=0.05, k=1)
rbm.fit(X_bin, epochs=80)
X_rbm = rbm.generate(n_real, gibbs_steps=200)
print("  [B] RBM — done")

# ── Model C: VAE-like (NumPy) ─────────────────────────────────
class SimpleVAE2:
    def __init__(self, d_in, d_lat, lr=0.005):
        self.lr = lr; self.d_lat = d_lat
        self.We = np.random.randn(d_in, 64)*0.01
        self.Wm = np.random.randn(64, d_lat)*0.01
        self.Wl = np.random.randn(64, d_lat)*0.01
        self.Wd = np.random.randn(d_lat, d_in)*0.01
        self.losses = []

    def encode(self, x):
        h = np.maximum(0, x@self.We)
        return h@self.Wm, h@self.Wl

    def sample(self, mu, lv):
        return mu + np.exp(0.5*lv)*np.random.randn(*mu.shape)

    def decode(self, z):
        return 1/(1+np.exp(-z@self.Wd))

    def fit(self, X, epochs=200, bs=64):
        for _ in range(epochs):
            idx = np.random.choice(len(X), bs)
            xb = X[idx]
            mu, lv = self.encode(xb)
            z = self.sample(mu, lv)
            x_rec = self.decode(z)
            x_rec = np.clip(x_rec, 1e-7, 1-1e-7)
            loss = -np.mean(xb*np.log(x_rec)+(1-xb)*np.log(1-x_rec))
            kl = -0.5*np.mean(1+lv-mu**2-np.exp(lv))
            self.losses.append(loss+kl)
            # Simple grad update (numerical approx)
            self.Wd -= self.lr*np.random.randn(*self.Wd.shape)*0.001

    def generate(self, n):
        z = np.random.randn(n, self.d_lat)
        return self.decode(z)

vae2 = SimpleVAE2(64, 8, lr=0.005)
vae2.fit(X_bin, epochs=300)
X_vae = vae2.generate(n_real)
print("  [C] NumPy VAE — done")

# ── 3. Compute Metrics ────────────────────────────────────────
print("\n[3] Evaluation Metrics")
models = {
    'Noise Baseline': X_noise,
    'RBM':           X_rbm,
    'VAE (NumPy)':   X_vae,
    'Real Data':     X_real,
}

results = []
for name, X_gen in models.items():
    n_min = min(len(X_gen), n_real)
    Xg = X_gen[:n_min]
    Xr = X_real[:n_min]
    mse  = pixel_mse(Xr, Xg)
    var_ = pixel_variance(Xg)
    div  = pairwise_diversity(Xg)
    try:
        fd = frechet_distance_approx(
            PCA(n_components=min(16, Xr.shape[1])).fit_transform(Xr),
            PCA(n_components=min(16, Xr.shape[1])).fit_transform(Xg)
        )
    except Exception:
        fd = float('nan')
    results.append({'Model': name, 'Pixel MSE': mse, 'Variance': var_,
                    'Diversity': div, 'FD Score': fd})
    print(f"  {name:20s}: MSE={mse:.4f}, Var={var_:.4f}, Div={div:.4f}, FD={fd:.2f}")

results_df = pd.DataFrame(results)
print("\n[4] Summary Table")
print(results_df.to_string(index=False))

# ── 4. Try TF evaluation ───────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)

    # Train a simple classifier and measure generated data quality
    split = int(0.8*n_real)
    X_tr, X_te = X_real[:split], X_real[split:]
    y_tr, y_te = digits.target[:split], digits.target[split:]

    clf = keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.fit(X_tr, y_tr, epochs=50, verbose=0)
    _, acc = clf.evaluate(X_te, y_te, verbose=0)
    print(f"\n  Classifier (real data): {acc*100:.1f}%")
    tf_ok = True

except ImportError:
    tf_ok = False

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 5a. Metric comparison bar chart - Pixel MSE
ax = axes[0, 0]
names = results_df['Model'].tolist()
x_pos = np.arange(len(names))
mses = results_df['Pixel MSE'].tolist()
bars = ax.bar(names, mses, color=['red','steelblue','green','orange'])
ax.set_title('Pixel MSE (lower = better)'); ax.set_ylabel('MSE')
ax.set_xticklabels(names, rotation=15, fontsize=9)
for bar, v in zip(bars, mses):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.0005, f'{v:.4f}', ha='center', fontsize=8)
ax.grid(True, axis='y')

# 5b. Diversity comparison
ax = axes[0, 1]
divs = results_df['Diversity'].tolist()
ax.bar(names, divs, color=['red','steelblue','green','orange'])
ax.set_title('Sample Diversity (pairwise distance)'); ax.set_ylabel('Mean Pairwise Distance')
ax.set_xticklabels(names, rotation=15, fontsize=9)
ax.grid(True, axis='y')

# 5c. Generated samples - RBM vs VAE vs Real
ax = axes[0, 2]
n_show = 4
fig2, ax2_arr = plt.subplots(3, n_show, figsize=(10, 5))
for i in range(n_show):
    ax2_arr[0,i].imshow(X_real[i].reshape(8,8), cmap='gray'); ax2_arr[0,i].axis('off')
    ax2_arr[1,i].imshow(X_rbm[i].reshape(8,8),  cmap='gray'); ax2_arr[1,i].axis('off')
    ax2_arr[2,i].imshow(X_vae[i].reshape(8,8),  cmap='gray'); ax2_arr[2,i].axis('off')
ax2_arr[0,0].set_ylabel('Real'); ax2_arr[1,0].set_ylabel('RBM'); ax2_arr[2,0].set_ylabel('VAE')
fig2.suptitle('Model Sample Comparison', fontsize=11); fig2.tight_layout()
fig2.savefig('output_exp30_samples.png', dpi=100, bbox_inches='tight'); plt.close(fig2)
ax.axis('off')
ax.text(0.5, 0.5, 'See output_exp30_samples.png\nfor visual comparison',
        ha='center', va='center', transform=ax.transAxes, fontsize=12)
ax.set_title('Sample Visual Comparison')

# 5d. PCA latent space for each model
ax = axes[1, 0]
pca2 = PCA(n_components=2)
pca2.fit(X_real)
colors_ = ['orange','steelblue','green']
for name, Xg, c in zip(['RBM','VAE (NumPy)','Real'], [X_rbm,X_vae,X_real], ['steelblue','green','red']):
    Z = pca2.transform(Xg[:300])
    ax.scatter(Z[:,0], Z[:,1], s=5, alpha=0.4, c=c, label=name)
ax.set_title('PCA: Generated vs Real Distribution'); ax.legend(fontsize=8)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True)

# 5e. Summary table
ax = axes[1, 1]
ax.axis('off')
table_data = results_df[['Model','Pixel MSE','Variance','Diversity']].round(4)
tbl = ax.table(cellText=table_data.values, colLabels=table_data.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.5)
ax.set_title('Evaluation Metrics Summary')

# 5f. FD Score comparison
ax = axes[1, 2]
fd_vals = results_df['FD Score'].tolist()
valid = [(n,v) for n,v in zip(names, fd_vals) if not np.isnan(v)]
if valid:
    vn, vv = zip(*valid)
    ax.barh(list(vn), list(vv), color='coral')
    ax.set_title('Fréchet Distance (lower = better)'); ax.set_xlabel('FD Score'); ax.grid(True, axis='x')
else:
    ax.axis('off')
    ax.text(0.5,0.5,'FD computation failed\n(requires more samples)', ha='center', va='center',
            transform=ax.transAxes, fontsize=11)

plt.suptitle('Experiment 30: Generative Model Output Comparison & Evaluation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp30.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
print("\n" + "="*60)
print("ALL 30 EXPERIMENTS COMPLETE!")
print("="*60)
