"""
Experiment 03: Probability Distributions and Bayes' Theorem
CO1 | Packages: NumPy, SciPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 60)
print("Experiment 03: Probability Distributions & Bayes' Theorem")
print("=" * 60)

np.random.seed(42)
x = np.linspace(-5, 5, 500)

# ── 1. Normal Distribution ────────────────────────────────────
mu, sigma = 0, 1
normal_pdf = stats.norm.pdf(x, mu, sigma)
print("\n[1] Normal Distribution N(0,1)")
print(f"  Mean={mu}, Std={sigma}")
print(f"  P(-1 < X < 1) = {stats.norm.cdf(1)-stats.norm.cdf(-1):.4f}")
print(f"  P(-2 < X < 2) = {stats.norm.cdf(2)-stats.norm.cdf(-2):.4f}")

# ── 2. Various Distributions ──────────────────────────────────
print("\n[2] Sampling from Distributions")
samples_normal = np.random.normal(0, 1, 1000)
samples_uniform = np.random.uniform(-3, 3, 1000)
samples_binomial = np.random.binomial(10, 0.5, 1000)
samples_poisson = np.random.poisson(3, 1000)
samples_exponential = np.random.exponential(1, 1000)

for name, s in [('Normal', samples_normal), ('Uniform', samples_uniform),
                ('Binomial', samples_binomial), ('Poisson', samples_poisson)]:
    print(f"  {name:12s}: mean={s.mean():.3f}, std={s.std():.3f}")

# ── 3. Bayes' Theorem ─────────────────────────────────────────
print("\n[3] Bayes' Theorem - Medical Test Example")
P_disease = 0.01          # Prior: 1% have disease
P_pos_given_disease = 0.95 # Sensitivity
P_pos_given_no_disease = 0.05  # 1 - Specificity
P_no_disease = 1 - P_disease

P_pos = (P_pos_given_disease * P_disease +
         P_pos_given_no_disease * P_no_disease)

P_disease_given_pos = (P_pos_given_disease * P_disease) / P_pos
print(f"  P(Disease)        = {P_disease}")
print(f"  P(+|Disease)      = {P_pos_given_disease}")
print(f"  P(+|No Disease)   = {P_pos_given_no_disease}")
print(f"  P(+)              = {P_pos:.4f}")
print(f"  P(Disease|+)      = {P_disease_given_pos:.4f}  ← Posterior")

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Normal PDF & CDF
ax = axes[0, 0]
ax.plot(x, normal_pdf, 'b-', linewidth=2, label='PDF N(0,1)')
ax.fill_between(x, normal_pdf, where=(x >= -1) & (x <= 1), alpha=0.3, color='blue', label='68%')
ax.set_title('Normal Distribution PDF'); ax.legend(); ax.grid(True)

# Multiple Normals
ax = axes[0, 1]
for mu_, sigma_, c in [(0, 1, 'blue'), (0, 2, 'red'), (1, 0.5, 'green')]:
    y_ = stats.norm.pdf(x, mu_, sigma_)
    ax.plot(x, y_, label=f'N({mu_},{sigma_})', color=c)
ax.set_title('Normal Distributions'); ax.legend(); ax.grid(True)

# Binomial
ax = axes[0, 2]
n_b, p_b = 20, 0.4
binom_x = np.arange(0, 21)
binom_pmf = stats.binom.pmf(binom_x, n_b, p_b)
ax.bar(binom_x, binom_pmf, color='coral', alpha=0.7)
ax.set_title(f'Binomial(n={n_b}, p={p_b})'); ax.grid(True, axis='y')

# Poisson
ax = axes[1, 0]
pois_x = np.arange(0, 15)
for lam, c in [(1, 'blue'), (3, 'red'), (6, 'green')]:
    ax.plot(pois_x, stats.poisson.pmf(pois_x, lam), 'o-', label=f'λ={lam}', color=c)
ax.set_title('Poisson Distributions'); ax.legend(); ax.grid(True)

# Exponential
ax = axes[1, 1]
x_exp = np.linspace(0, 6, 300)
for scale, c in [(0.5, 'blue'), (1, 'red'), (2, 'green')]:
    ax.plot(x_exp, stats.expon.pdf(x_exp, scale=scale), label=f'λ={1/scale}', color=c)
ax.set_title('Exponential Distributions'); ax.legend(); ax.grid(True)

# Bayes Visualisation
ax = axes[1, 2]
priors = np.linspace(0.001, 0.2, 200)
posteriors = [(P_pos_given_disease * p) /
              (P_pos_given_disease * p + P_pos_given_no_disease * (1-p))
              for p in priors]
ax.plot(priors, posteriors, 'purple', linewidth=2)
ax.axvline(P_disease, color='red', linestyle='--', label=f'P(D)={P_disease}')
ax.axhline(P_disease_given_pos, color='blue', linestyle='--',
           label=f'P(D|+)={P_disease_given_pos:.3f}')
ax.set_title("Bayes: Posterior vs Prior"); ax.set_xlabel("Prior P(Disease)")
ax.set_ylabel("Posterior P(Disease|+)"); ax.legend(); ax.grid(True)

plt.suptitle("Experiment 03: Probability Distributions & Bayes' Theorem",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp03.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp03.png")
