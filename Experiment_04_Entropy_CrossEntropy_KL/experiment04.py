"""
Experiment 04: Entropy, Cross-Entropy and KL Divergence
CO1 | Packages: NumPy, SciPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kl_div
from scipy.stats import entropy as sp_entropy

print("=" * 60)
print("Experiment 04: Entropy, Cross-Entropy & KL Divergence")
print("=" * 60)

# ── 1. Shannon Entropy ────────────────────────────────────────
def shannon_entropy(p):
    p = np.array(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

print("\n[1] Shannon Entropy")
distributions = {
    'Uniform (4 classes)': [0.25, 0.25, 0.25, 0.25],
    'Skewed':              [0.7,  0.1,  0.1,  0.1],
    'Deterministic':       [1.0,  0.0,  0.0,  0.0],
    'Moderate':            [0.5,  0.3,  0.1,  0.1],
}
for name, dist in distributions.items():
    h = shannon_entropy(dist)
    print(f"  {name:28s}: H = {h:.4f} bits")

# ── 2. Cross-Entropy ──────────────────────────────────────────
def cross_entropy(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    q = np.clip(q, 1e-10, 1.0)
    return -np.sum(p * np.log(q))

print("\n[2] Cross-Entropy H(P, Q)")
P_true = np.array([1, 0, 0, 0], dtype=float)
Q_pred_good  = np.array([0.9, 0.05, 0.03, 0.02])
Q_pred_bad   = np.array([0.1, 0.4,  0.3,  0.2])

print(f"  P (true)         = {P_true}")
print(f"  Q good prediction: H(P,Q) = {cross_entropy(P_true, Q_pred_good):.4f}")
print(f"  Q bad  prediction: H(P,Q) = {cross_entropy(P_true, Q_pred_bad):.4f}")

# ── 3. KL Divergence ─────────────────────────────────────────
def kl_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    q = np.clip(q, 1e-10, 1.0)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

print("\n[3] KL Divergence KL(P || Q)")
P = np.array([0.4, 0.3, 0.2, 0.1])
Q1 = np.array([0.4, 0.3, 0.2, 0.1])  # same
Q2 = np.array([0.25, 0.25, 0.25, 0.25])
Q3 = np.array([0.1, 0.2, 0.3, 0.4])

print(f"  P       = {P}")
print(f"  KL(P||P)= {kl_divergence(P, Q1):.6f}  (same distribution)")
print(f"  KL(P||uniform)= {kl_divergence(P, Q2):.4f}")
print(f"  KL(P||reverse)= {kl_divergence(P, Q3):.4f}")
print(f"  Note: KL is asymmetric: KL(P||Q)≠KL(Q||P)")
print(f"  KL(Q2||P)= {kl_divergence(Q2, P):.4f}")

# ── 4. Gaussian KL Divergence ─────────────────────────────────
def gaussian_kl(mu1, sigma1, mu2, sigma2):
    return (np.log(sigma2/sigma1) +
            (sigma1**2 + (mu1-mu2)**2) / (2*sigma2**2) - 0.5)

print("\n[4] Gaussian KL Divergence KL(N(mu1,s1) || N(mu2,s2))")
print(f"  KL(N(0,1)||N(0,1))   = {gaussian_kl(0,1,0,1):.6f}")
print(f"  KL(N(0,1)||N(1,1))   = {gaussian_kl(0,1,1,1):.4f}")
print(f"  KL(N(0,1)||N(0,2))   = {gaussian_kl(0,1,0,2):.4f}")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a. Entropy vs probability (binary)
ax = axes[0, 0]
p_vals = np.linspace(0.001, 0.999, 300)
h_vals = -p_vals * np.log2(p_vals) - (1-p_vals) * np.log2(1-p_vals)
ax.plot(p_vals, h_vals, 'b-', linewidth=2)
ax.set_title('Binary Entropy H(p)'); ax.set_xlabel('p'); ax.set_ylabel('Entropy (bits)')
ax.axvline(0.5, color='red', linestyle='--', label='max H at p=0.5')
ax.legend(); ax.grid(True)

# 5b. Cross-entropy vs predictions
ax = axes[0, 1]
q_vals = np.linspace(0.01, 0.99, 300)
ce_vals = -np.log(q_vals)
ax.plot(q_vals, ce_vals, 'coral', linewidth=2)
ax.set_title('Cross-Entropy Loss (true class)'); ax.set_xlabel('Predicted prob for true class')
ax.set_ylabel('Cross-Entropy'); ax.grid(True)

# 5c. KL Divergence between Gaussians
ax = axes[1, 0]
x_vals = np.linspace(-6, 6, 300)
p_gauss = (1 / (np.sqrt(2*np.pi))) * np.exp(-0.5 * x_vals**2)
shifts = [0, 1, 2, 3]
colors = ['blue', 'green', 'orange', 'red']
for shift, color in zip(shifts, colors):
    q_gauss = (1 / (np.sqrt(2*np.pi))) * np.exp(-0.5 * (x_vals - shift)**2)
    kl_val = gaussian_kl(0, 1, shift, 1)
    ax.plot(x_vals, q_gauss, color=color, label=f'N({shift},1) KL={kl_val:.2f}')
ax.plot(x_vals, p_gauss, 'b-', linewidth=3, label='P=N(0,1)')
ax.set_title('KL Divergence: Shifting Q from P'); ax.legend(fontsize=7); ax.grid(True)

# 5d. Entropy of different distributions
ax = axes[1, 1]
n_classes = np.arange(2, 15)
entropies = [-np.sum(np.full(n, 1/n) * np.log2(np.full(n, 1/n))) for n in n_classes]
ax.plot(n_classes, entropies, 'go-', linewidth=2)
ax.set_title('Max Entropy vs Number of Classes')
ax.set_xlabel('Number of Classes'); ax.set_ylabel('Max Entropy (bits)'); ax.grid(True)

plt.suptitle('Experiment 04: Entropy, Cross-Entropy & KL Divergence',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp04.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp04.png")
