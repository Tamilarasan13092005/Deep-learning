"""
Experiment 26: Gibbs Sampling and Contrastive Divergence Analysis
CO5 | Packages: NumPy, SciPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

print("=" * 60)
print("Experiment 26: Gibbs Sampling & Contrastive Divergence Analysis")
print("=" * 60)

np.random.seed(42)

# ── 1. CD-k Comparison ────────────────────────────────────────
class RBM_CDk:
    def __init__(self, n_v, n_h, lr=0.05):
        self.W  = np.random.randn(n_v, n_h) * 0.01
        self.bv = np.zeros(n_v)
        self.bh = np.zeros(n_h)
        self.lr = lr
        self.losses = []

    def p_h(self, v): return sigmoid(v @ self.W + self.bh)
    def p_v(self, h): return sigmoid(h @ self.W.T + self.bv)
    def sample(self, probs): return (np.random.rand(*probs.shape) < probs).astype(float)

    def cd_step(self, v0, k=1):
        h0 = self.p_h(v0); h0s = self.sample(h0)
        vk, hk = v0, h0s
        for _ in range(k):
            vk = self.p_v(hk); vk = self.sample(vk)
            hk = self.p_h(vk)
        return v0, h0, vk, hk

    def fit_step(self, X, k=1):
        v0, h0, vk, hk = self.cd_step(X, k)
        m = len(X)
        self.W  += self.lr * (v0.T@h0 - vk.T@hk) / m
        self.bv += self.lr * (v0 - vk).mean(0)
        self.bh += self.lr * (h0 - hk).mean(0)
        recon_v = self.p_v(h0)
        return np.mean((X - recon_v)**2)

    def fit(self, X, epochs=50, batch_size=32, k=1):
        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            ep_loss = []
            for i in range(0, len(X), batch_size):
                batch = X[idx[i:i+batch_size]]
                ep_loss.append(self.fit_step(batch, k))
            self.losses.append(np.mean(ep_loss))

# ── 2. Train on digits ────────────────────────────────────────
print("\n[2] CD-k Comparison (k=1, 5, 10)")
from sklearn.datasets import load_digits
digits = load_digits()
X_bin = (digits.data > 8).astype(float)

results_cdk = {}
for k in [1, 5, 10]:
    rbm = RBM_CDk(n_v=64, n_h=32, lr=0.05)
    rbm.fit(X_bin, epochs=100, batch_size=64, k=k)
    results_cdk[f'CD-{k}'] = rbm.losses[:]
    print(f"  CD-{k:2d}: final_loss={rbm.losses[-1]:.4f}")

# ── 3. Gibbs Chain Analysis ───────────────────────────────────
print("\n[3] Gibbs Chain: Markov Chain Statistics")
rbm_final = RBM_CDk(64, 32, lr=0.05)
rbm_final.fit(X_bin, epochs=100, k=1)

# Run Gibbs chain from random init
v = (np.random.rand(1, 64) > 0.5).astype(float)
chain_samples, chain_energies = [], []

def free_energy(rbm, v):
    wx_b = v @ rbm.W + rbm.bh
    return -v @ rbm.bv - np.sum(np.log(1 + np.exp(wx_b)), axis=1)

for step in range(500):
    h = rbm_final.sample(rbm_final.p_h(v))
    v = rbm_final.sample(rbm_final.p_v(h))
    chain_samples.append(v.copy())
    chain_energies.append(free_energy(rbm_final, v)[0])

chain_samples = np.array(chain_samples).squeeze()
chain_energies = np.array(chain_energies)
print(f"  Chain mean energy: {chain_energies.mean():.4f}")
print(f"  Chain std energy:  {chain_energies.std():.4f}")

# ── 4. Learning Rate Comparison ───────────────────────────────
print("\n[4] Learning Rate Effect on CD-1")
lr_results = {}
for lr in [0.001, 0.01, 0.05, 0.1]:
    rbm_lr = RBM_CDk(64, 32, lr=lr)
    rbm_lr.fit(X_bin, epochs=50, k=1)
    lr_results[f'lr={lr}'] = rbm_lr.losses[:]
    print(f"  lr={lr}: final_loss={rbm_lr.losses[-1]:.4f}")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 5a. CD-k comparison
ax = axes[0, 0]
for name, losses in results_cdk.items():
    ax.plot(losses, linewidth=2, label=name)
ax.set_title('CD-k Comparison'); ax.set_xlabel('Epoch')
ax.set_ylabel('Reconstruction Error'); ax.legend(); ax.grid(True)

# 5b. Learning rate comparison
ax = axes[0, 1]
for name, losses in lr_results.items():
    ax.plot(losses, linewidth=2, label=name)
ax.set_title('Learning Rate Effect (CD-1)'); ax.set_xlabel('Epoch')
ax.set_ylabel('Reconstruction Error'); ax.legend(fontsize=8); ax.grid(True)

# 5c. Gibbs chain: energy trace
ax = axes[0, 2]
ax.plot(chain_energies, 'coral', linewidth=1.5)
ax.set_title('Gibbs Chain: Free Energy Trace'); ax.set_xlabel('Step')
ax.set_ylabel('Free Energy'); ax.grid(True)

# 5d. Gibbs chain: sample diversity
ax = axes[1, 0]
ax.imshow(chain_samples[::5].T, aspect='auto', cmap='gray')
ax.set_title('Gibbs Chain Samples (every 5 steps)'); ax.set_xlabel('Step (÷5)')
ax.set_ylabel('Visible Unit')

# 5e. CD-1 vs CD-10 final reconstructions
ax = axes[1, 1]
rbm_cd1  = RBM_CDk(64, 32, lr=0.05); rbm_cd1.fit(X_bin, epochs=100, k=1)
rbm_cd10 = RBM_CDk(64, 32, lr=0.05); rbm_cd10.fit(X_bin, epochs=100, k=10)

n_show = 4
fig2, ax2_arr = plt.subplots(3, n_show, figsize=(8, 5))
for i in range(n_show):
    r1  = rbm_cd1.p_v(rbm_cd1.sample(rbm_cd1.p_h(X_bin[i:i+1])))[0]
    r10 = rbm_cd10.p_v(rbm_cd10.sample(rbm_cd10.p_h(X_bin[i:i+1])))[0]
    ax2_arr[0,i].imshow(X_bin[i].reshape(8,8),  cmap='gray'); ax2_arr[0,i].axis('off')
    ax2_arr[1,i].imshow(r1.reshape(8,8),         cmap='gray'); ax2_arr[1,i].axis('off')
    ax2_arr[2,i].imshow(r10.reshape(8,8),        cmap='gray'); ax2_arr[2,i].axis('off')
ax2_arr[0,0].set_ylabel('Original'); ax2_arr[1,0].set_ylabel('CD-1'); ax2_arr[2,0].set_ylabel('CD-10')
fig2.suptitle('CD-1 vs CD-10 Reconstructions', fontsize=11)
fig2.tight_layout()
fig2.savefig('output_exp26_reconstruction.png', dpi=100, bbox_inches='tight')
plt.close(fig2)
ax.axis('off')
ax.text(0.5, 0.5, 'See output_exp26_reconstruction.png', ha='center', va='center',
        transform=ax.transAxes, fontsize=11)
ax.set_title('CD Reconstruction Comparison')

# 5f. Contrastive Divergence Concept
ax = axes[1, 2]
ax.axis('off')
cd_text = """
Contrastive Divergence (CD-k):

Objective: minimize KL(p_data || p_model)

Exact gradient: expensive (sum over all configs)

CD-k approximation:
 1. Start with data sample v₀
 2. Run k Gibbs steps: v₀ → h₀ → v₁ → ... → vₖ
 3. Approximate gradient:
    ΔW ≈ ⟨v₀h₀ᵀ⟩_data - ⟨vₖhₖᵀ⟩_model

CD-1: fast, biased, works well in practice
CD-k: less biased, slower, better mixing

Gibbs Sampling:
  P(h_j=1|v) = σ(Σᵢ W_ij vᵢ + b_j)
  P(v_i=1|h) = σ(Σⱼ W_ij h_j + a_i)
  (all units in each layer updated in parallel)
"""
ax.text(0.02, 0.98, cd_text, transform=ax.transAxes, va='top',
        fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('CD Algorithm Summary')

plt.suptitle('Experiment 26: Gibbs Sampling & Contrastive Divergence Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp26.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
