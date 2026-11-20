"""
Experiment 14: Unfolded RNN Architecture Visualization
CO3 | Packages: TensorFlow, Keras, NetworkX, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

print("=" * 60)
print("Experiment 14: Unfolded RNN Architecture Visualization")
print("=" * 60)

np.random.seed(42)

# ── 1. Manual RNN Cell ────────────────────────────────────────
class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.Wx = np.random.randn(input_size, hidden_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b  = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        z = x @ self.Wx + h_prev @ self.Wh + self.b
        return np.tanh(z)

T, input_size, hidden_size = 5, 3, 4
cell = RNNCell(input_size, hidden_size)
X_seq = np.random.randn(T, input_size)
h = np.zeros(hidden_size)
hidden_states = []
print("\n[1] RNN Forward Pass through T=5 Timesteps")
for t in range(T):
    h = cell.forward(X_seq[t], h)
    hidden_states.append(h.copy())
    print(f"  t={t}: h={np.round(h[:3], 4)} ...")

# ── 2. Try NetworkX visualization ─────────────────────────────
try:
    import networkx as nx
    nx_ok = True
except ImportError:
    nx_ok = False
    print("  NetworkX not installed — using manual drawing.")

# ── 3. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 3a. Unfolded RNN diagram (manual)
ax = axes[0]
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-1, 3.5)
ax.axis('off')
ax.set_title('Unfolded RNN Over Time', fontsize=13, fontweight='bold')

colors = {'input': 'lightblue', 'hidden': 'lightyellow', 'output': 'lightgreen'}

for t in range(T):
    # Input node
    circ_x = ax.add_patch(plt.Circle((t, 0), 0.3, color=colors['input'], ec='navy', zorder=5))
    ax.text(t, 0, f'x{t}', ha='center', va='center', fontsize=10, fontweight='bold')

    # Hidden node
    circ_h = ax.add_patch(plt.Circle((t, 1.5), 0.3, color=colors['hidden'], ec='darkorange', zorder=5))
    ax.text(t, 1.5, f'h{t}', ha='center', va='center', fontsize=10, fontweight='bold')

    # Output node
    circ_o = ax.add_patch(plt.Circle((t, 3), 0.3, color=colors['output'], ec='darkgreen', zorder=5))
    ax.text(t, 3, f'o{t}', ha='center', va='center', fontsize=10, fontweight='bold')

    # x→h arrow
    ax.annotate('', xy=(t, 1.2), xytext=(t, 0.3),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
    # h→o arrow
    ax.annotate('', xy=(t, 2.7), xytext=(t, 1.8),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

    # h→h (recurrent)
    if t < T-1:
        ax.annotate('', xy=(t+1-0.3, 1.5), xytext=(t+0.3, 1.5),
                   arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

# Labels
ax.text(-0.5, 0, 'Input\n(xt)', ha='right', va='center', fontsize=9, color='navy')
ax.text(-0.5, 1.5, 'Hidden\n(ht)', ha='right', va='center', fontsize=9, color='darkorange')
ax.text(-0.5, 3, 'Output\n(ot)', ha='right', va='center', fontsize=9, color='darkgreen')

patches = [mpatches.Patch(color='lightblue', label='Input'),
           mpatches.Patch(color='lightyellow', label='Hidden State'),
           mpatches.Patch(color='lightgreen', label='Output')]
ax.legend(handles=patches, loc='lower right', fontsize=9)

# 3b. Hidden state heatmap over time
ax = axes[1]
H_matrix = np.array(hidden_states)
im = ax.imshow(H_matrix.T, aspect='auto', cmap='RdYlBu', vmin=-1, vmax=1)
ax.set_xlabel('Timestep t'); ax.set_ylabel('Hidden Unit')
ax.set_title('Hidden State Activations Over Time')
ax.set_xticks(range(T)); ax.set_xticklabels([f't={t}' for t in range(T)])
ax.set_yticks(range(hidden_size)); ax.set_yticklabels([f'h{i}' for i in range(hidden_size)])
plt.colorbar(im, ax=ax, label='Activation (tanh)')
for i in range(hidden_size):
    for j in range(T):
        ax.text(j, i, f'{H_matrix[j,i]:.2f}', ha='center', va='center', fontsize=9)

plt.suptitle('Experiment 14: Unfolded RNN Architecture Visualization',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp14.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp14.png")
