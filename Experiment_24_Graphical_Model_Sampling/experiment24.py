"""
Experiment 24: Graphical Model Sampling and Probabilistic Inference
CO4 | Packages: pgmpy, NumPy, Pandas, NetworkX, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 24: Graphical Model Sampling & Probabilistic Inference")
print("=" * 60)

np.random.seed(42)

# ── 1. Manual MCMC Sampling (Gibbs) on a Bayes Net ───────────
print("\n[1] Gibbs Sampling on Rain-Sprinkler-Grass Network")

# P(Rain=1) = 0.2
# P(Sprinkler=1 | Rain=0) = 0.4, P(Sprinkler=1 | Rain=1) = 0.01
# P(Wet Grass=1 | Sprinkler, Rain) lookup
P_G = {(0,0): 0.0, (0,1): 0.9, (1,0): 0.9, (1,1): 0.99}

def p_rain_given_rest(s, g):
    p1 = 0.2 * (P_G[(1,s)] if g else 1-P_G[(1,s)])
    p0 = 0.8 * (P_G[(0,s)] if g else 1-P_G[(0,s)])
    return p1 / (p0 + p1)

def p_sprinkler_given_rest(r, g):
    s1 = (0.01 if r else 0.4) * (P_G[(r,1)] if g else 1-P_G[(r,1)])
    s0 = (0.99 if r else 0.6) * (P_G[(r,0)] if g else 1-P_G[(r,0)])
    return s1 / (s0 + s1)

# Gibbs sampling
n_samples = 5000
burn_in = 500
r, s, g = 0, 0, 1  # observed grass=1

samples = []
for i in range(n_samples + burn_in):
    r = int(np.random.rand() < p_rain_given_rest(s, g))
    s = int(np.random.rand() < p_sprinkler_given_rest(r, g))
    if i >= burn_in:
        samples.append((r, s))

samples = np.array(samples)
p_rain_wet    = samples[:, 0].mean()
p_sprink_wet  = samples[:, 1].mean()
print(f"  Gibbs: P(Rain=1  | Grass=1) ≈ {p_rain_wet:.4f}")
print(f"  Gibbs: P(Sprinkler=1 | Grass=1) ≈ {p_sprink_wet:.4f}")

# ── 2. Try pgmpy ──────────────────────────────────────────────
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.sampling import BayesianModelSampling

    model = BayesianNetwork([('Rain','Sprinkler'),('Rain','Grass'),('Sprinkler','Grass')])
    cpd_R = TabularCPD('Rain', 2, [[0.8],[0.2]])
    cpd_S = TabularCPD('Sprinkler', 2, [[0.6,0.99],[0.4,0.01]],
                       evidence=['Rain'], evidence_card=[2])
    cpd_G = TabularCPD('Grass', 2,
                       [[1.0, 0.1, 0.1, 0.01],
                        [0.0, 0.9, 0.9, 0.99]],
                       evidence=['Rain','Sprinkler'], evidence_card=[2,2])
    model.add_cpds(cpd_R, cpd_S, cpd_G)
    assert model.check_model()

    infer = VariableElimination(model)
    q = infer.query(['Rain','Sprinkler'], evidence={'Grass': 1})
    print(f"\n[2] pgmpy Exact Inference P(Rain,Sprinkler|Grass=1):\n{q}")

    sampler = BayesianModelSampling(model)
    samples_pgmpy = sampler.forward_sample(size=2000)
    print(f"\n[3] pgmpy Forward Sampling (2000):")
    print(samples_pgmpy.head(5))
    pgmpy_ok = True

except ImportError:
    pgmpy_ok = False
    print("  pgmpy not installed — using Gibbs sampling only.")

# ── 3. Markov Chain Convergence Analysis ──────────────────────
print("\n[4] Markov Chain Convergence Analysis")
running_mean_r = np.cumsum(samples[:, 0]) / np.arange(1, len(samples)+1)
running_mean_s = np.cumsum(samples[:, 1]) / np.arange(1, len(samples)+1)

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 4a. Gibbs trace - Rain
ax = axes[0, 0]
ax.plot(samples[:500, 0], 'b-', linewidth=1, alpha=0.7)
ax.set_title('Gibbs Trace: Rain'); ax.set_xlabel('Sample')
ax.set_ylabel('Rain (0/1)'); ax.grid(True)

# 4b. Running mean convergence
ax = axes[0, 1]
ax.plot(running_mean_r, 'b-', linewidth=2, label='P(Rain|Grass=1)')
ax.plot(running_mean_s, 'r-', linewidth=2, label='P(Sprinkler|Grass=1)')
ax.axhline(p_rain_wet, color='blue', linestyle='--', alpha=0.5, label=f'P(R)={p_rain_wet:.3f}')
ax.axhline(p_sprink_wet, color='red', linestyle='--', alpha=0.5, label=f'P(S)={p_sprink_wet:.3f}')
ax.set_title('Gibbs MCMC Convergence'); ax.set_xlabel('Sample')
ax.set_ylabel('Running Mean P'); ax.legend(fontsize=8); ax.grid(True)
ax.set_ylim(0, 1)

# 4c. Sample histogram
ax = axes[0, 2]
joint_counts = pd.DataFrame(samples, columns=['Rain','Sprinkler']).groupby(['Rain','Sprinkler']).size()
labels = [f'R={r},S={s}' for (r,s) in joint_counts.index]
ax.bar(labels, joint_counts.values / joint_counts.sum(), color='steelblue', edgecolor='black')
ax.set_title('Joint P(Rain, Sprinkler | Grass=1)'); ax.set_ylabel('Probability')
ax.grid(True, axis='y')

# 4d. Variable Elimination concept
ax = axes[1, 0]
ax.axis('off')
ve_text = """
Variable Elimination (VE) Algorithm:

Goal: P(X | evidence E)

1. Start with all factors (CPDs) φ_i
2. Set evidence: reduce factors
3. For each hidden variable Z (in order):
   a. Collect all factors containing Z
   b. Multiply them: ψ = Π φ_i(Z, ...)
   c. Marginalize out Z: τ = Σ_Z ψ
   d. Add τ back to factor list
4. Multiply remaining factors
5. Normalize to get probability

Gibbs Sampling Alternative:
  - Initialize all variables randomly
  - Repeat: sample X_i | all others
  - Collect samples after burn-in
  - Empirical frequencies → probabilities
"""
ax.text(0.02, 0.98, ve_text, transform=ax.transAxes, va='top',
        fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('Inference Methods')

# 4e. Autocorrelation of Gibbs chain
ax = axes[1, 1]
chain = samples[:, 0].astype(float) - samples[:, 0].mean()
max_lag = 50
acf = [np.corrcoef(chain[:-lag], chain[lag:])[0,1] if lag>0 else 1.0 for lag in range(max_lag)]
ax.bar(range(max_lag), acf, color='coral')
ax.axhline(0, color='black'); ax.axhline(0.05, color='red', linestyle='--')
ax.axhline(-0.05, color='red', linestyle='--')
ax.set_title('Gibbs Chain Autocorrelation (Rain)'); ax.set_xlabel('Lag')
ax.set_ylabel('ACF'); ax.grid(True)

# 4f. CPD visualization
ax = axes[1, 2]
cpd_G_mat = np.array([[1.0, 0.1, 0.1, 0.01],
                       [0.0, 0.9, 0.9, 0.99]])
im = ax.imshow(cpd_G_mat, cmap='Greens', vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_xticklabels(['R=0,S=0','R=0,S=1','R=1,S=0','R=1,S=1'], rotation=20, fontsize=9)
ax.set_yticks([0,1]); ax.set_yticklabels(['Dry','Wet'])
ax.set_title('P(Grass | Rain, Sprinkler)')
plt.colorbar(im, ax=ax)
for i in range(2):
    for j in range(4):
        ax.text(j, i, f'{cpd_G_mat[i,j]:.2f}', ha='center', va='center', fontsize=11)

plt.suptitle('Experiment 24: Graphical Model Sampling & Probabilistic Inference',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp24.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp24.png")
