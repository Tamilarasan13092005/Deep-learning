"""
Experiment 23: Bayesian Network Construction and Visualization
CO4 | Packages: pgmpy, NetworkX, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 23: Bayesian Network Construction & Visualization")
print("=" * 60)

# ── 1. Try pgmpy ──────────────────────────────────────────────
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    print("\n[1] Building Student Bayesian Network")
    # Classic student network: D→G←I, I→S, G→L
    model = BayesianNetwork([
        ('Difficulty', 'Grade'),
        ('Intelligence', 'Grade'),
        ('Intelligence', 'SAT'),
        ('Grade', 'Letter')
    ])

    # CPDs
    cpd_D = TabularCPD('Difficulty', 2, [[0.6], [0.4]])
    cpd_I = TabularCPD('Intelligence', 2, [[0.7], [0.3]])
    cpd_G = TabularCPD('Grade', 3,
                       [[0.3, 0.05, 0.9, 0.5],
                        [0.4, 0.25, 0.08, 0.3],
                        [0.3, 0.70, 0.02, 0.2]],
                       evidence=['Difficulty', 'Intelligence'],
                       evidence_card=[2, 2])
    cpd_S = TabularCPD('SAT', 2,
                       [[0.95, 0.2],
                        [0.05, 0.8]],
                       evidence=['Intelligence'], evidence_card=[2])
    cpd_L = TabularCPD('Letter', 2,
                       [[0.1, 0.4, 0.99],
                        [0.9, 0.6, 0.01]],
                       evidence=['Grade'], evidence_card=[3])

    model.add_cpds(cpd_D, cpd_I, cpd_G, cpd_S, cpd_L)
    print(f"  Model valid: {model.check_model()}")

    infer = VariableElimination(model)

    # Query: P(Grade | Difficulty=1, Intelligence=0)
    q1 = infer.query(['Grade'], evidence={'Difficulty': 1, 'Intelligence': 0})
    print(f"\n[2] P(Grade | Hard course, Low intelligence):")
    print(f"  {q1}")

    q2 = infer.query(['Letter'], evidence={'Grade': 2})
    print(f"\n[3] P(Letter | Grade=A):")
    print(f"  {q2}")

    q3 = infer.query(['Intelligence'], evidence={'Grade': 0})
    print(f"\n[4] Posterior P(Intelligence | Grade=C):")
    print(f"  {q3}")

    pgmpy_ok = True

except ImportError:
    print("  pgmpy not installed — manual Bayesian network demo.")
    pgmpy_ok = False

# ── 2. NetworkX Visualization ─────────────────────────────────
try:
    import networkx as nx
    nx_ok = True
except ImportError:
    nx_ok = False

# ── 3. Manual Bayes Net (rain-sprinkler-grass) ────────────────
print("\n[5] Manual Bayesian Network: Rain-Sprinkler-Grass")
# P(Rain), P(Sprinkler|Rain), P(Grass|Rain,Sprinkler)
P_R  = {0: 0.8, 1: 0.2}        # P(Rain=0)=0.8
P_S_given_R  = {0: 0.4, 1: 0.01}  # P(Sprinkler=1|Rain)
P_G_given_RS = {(0,0): 0.0, (0,1): 0.9, (1,0): 0.9, (1,1): 0.99}  # P(Grass=1|R,S)

def joint(r, s, g):
    p_r = P_R[r]
    p_s = P_S_given_R[r] if s==1 else 1-P_S_given_R[r]
    p_g = P_G_given_RS[(r,s)] if g==1 else 1-P_G_given_RS[(r,s)]
    return p_r * p_s * p_g

# P(Rain=1 | Grass=1) — Bayes inference
p_grass1 = sum(joint(r,s,1) for r in [0,1] for s in [0,1])
p_rain1_grass1 = sum(joint(1,s,1) for s in [0,1]) / p_grass1
print(f"  P(Rain=1 | Grass wet) = {p_rain1_grass1:.4f}")

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# 4a. Student network graph
ax = axes[0]
if nx_ok:
    G = nx.DiGraph()
    G.add_edges_from([('Difficulty','Grade'),('Intelligence','Grade'),
                      ('Intelligence','SAT'),('Grade','Letter')])
    pos = {'Difficulty':  (0, 1), 'Intelligence': (2, 1),
           'Grade':       (1, 0), 'SAT':          (3, 0),
           'Letter':      (1,-1)}
    colors = ['lightblue','lightblue','lightyellow','lightyellow','lightgreen']
    nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=2000,
                     font_size=9, font_weight='bold', arrows=True,
                     arrowsize=25, edge_color='gray')
    ax.set_title('Student Bayesian Network')
    ax.axis('off')
else:
    ax.axis('off')
    ax.text(0.5, 0.5, 'Student BN\n(install networkx)', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)

# 4b. Rain-Sprinkler-Grass network
ax = axes[1]
if nx_ok:
    G2 = nx.DiGraph()
    G2.add_edges_from([('Rain','Sprinkler'),('Rain','Grass'),('Sprinkler','Grass')])
    pos2 = {'Rain':(1,2), 'Sprinkler':(0,0), 'Grass':(2,0)}
    nx.draw_networkx(G2, pos2, ax=ax, node_color='lightyellow',
                     node_size=3000, font_size=11, font_weight='bold',
                     arrows=True, arrowsize=30, edge_color='steelblue')
    ax.set_title('Rain-Sprinkler-Grass BN'); ax.axis('off')
else:
    ax.axis('off')

# 4c. CPD heatmap for P(Grade | Difficulty, Intelligence)
ax = axes[2]
cpd_data = np.array([[0.3, 0.05, 0.9, 0.5],
                     [0.4, 0.25, 0.08, 0.3],
                     [0.3, 0.70, 0.02, 0.2]])
im = ax.imshow(cpd_data, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_xticklabels(['D=Easy,I=Low','D=Hard,I=Low','D=Easy,I=High','D=Hard,I=High'],
                   rotation=25, ha='right', fontsize=8)
ax.set_yticks(range(3)); ax.set_yticklabels(['Grade C','Grade B','Grade A'])
ax.set_title('CPD: P(Grade | Difficulty, Intelligence)')
plt.colorbar(im, ax=ax, label='Probability')
for i in range(3):
    for j in range(4):
        ax.text(j, i, f'{cpd_data[i,j]:.2f}', ha='center', va='center', fontsize=10)

plt.suptitle('Experiment 23: Bayesian Network Construction & Visualization',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp23.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp23.png")
