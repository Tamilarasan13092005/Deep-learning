"""
Experiment 15: BPTT Flow and Gradient Analysis
CO3 | Packages: TensorFlow, Keras, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 15: BPTT Flow and Gradient Analysis")
print("=" * 60)

np.random.seed(42)

# ── 1. Manual BPTT Implementation ────────────────────────────
class SimpleRNN_BPTT:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.Wx = np.random.randn(input_size, hidden_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Wy = np.random.randn(hidden_size, output_size) * 0.1
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)
        self.lr = lr

    def forward(self, X_seq):
        T = len(X_seq)
        self.h = [np.zeros(self.Wh.shape[0])]
        self.z = []
        self.outputs = []
        for t in range(T):
            z_t = X_seq[t] @ self.Wx + self.h[-1] @ self.Wh + self.bh
            h_t = np.tanh(z_t)
            o_t = h_t @ self.Wy + self.by
            self.z.append(z_t)
            self.h.append(h_t)
            self.outputs.append(o_t)
        return self.outputs

    def bptt(self, X_seq, y_seq):
        T = len(X_seq)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros(self.Wh.shape[0])

        grad_norms = []

        for t in reversed(range(T)):
            dy = self.outputs[t] - y_seq[t]
            dWy += np.outer(self.h[t+1], dy)
            dby += dy

            dh = dy @ self.Wy.T + dh_next
            dtanh = (1 - self.h[t+1]**2) * dh
            dWx += np.outer(X_seq[t], dtanh)
            dWh += np.outer(self.h[t], dtanh)
            dbh += dtanh
            dh_next = dtanh @ self.Wh.T
            grad_norms.append(np.linalg.norm(dh_next))

        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        self.Wy -= self.lr * dWy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby
        return list(reversed(grad_norms))

# ── 2. Gradient Vanishing/Exploding Study ─────────────────────
print("\n[2] Gradient Norm Analysis across Sequence Lengths")
T = 20
input_size, hidden_size, output_size = 2, 8, 1
rnn = SimpleRNN_BPTT(input_size, hidden_size, output_size, lr=0.001)

X_seq = [np.random.randn(input_size) for _ in range(T)]
y_seq = [np.random.randn(output_size) for _ in range(T)]

losses, grad_norms_per_epoch = [], []
for epoch in range(100):
    outputs = rnn.forward(X_seq)
    loss = sum((o - y)**2 for o, y in zip(outputs, y_seq)) / T
    losses.append(float(np.mean(loss)))
    grad_norms = rnn.bptt(X_seq, y_seq)
    grad_norms_per_epoch.append(grad_norms)

print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print(f"  Gradient norms at t=0 (last): {grad_norms_per_epoch[-1][0]:.6f}")
print(f"  Gradient norms at t=T (first): {grad_norms_per_epoch[-1][-1]:.6f}")

# ── 3. Vanishing Gradient with Different Activations ──────────
print("\n[3] Gradient Magnitude vs Time Step")
def simulate_gradient_propagation(activation, T=30):
    if activation == 'tanh':
        deriv = lambda h: 1 - h**2
        act = np.tanh
    elif activation == 'sigmoid':
        s = lambda x: 1/(1+np.exp(-x))
        deriv = lambda h: h*(1-h)
        act = s
    elif activation == 'relu':
        deriv = lambda h: (h > 0).astype(float)
        act = lambda x: np.maximum(0, x)

    Wh = np.random.randn(4, 4) * 0.9
    h = np.random.randn(4) * 0.1
    grad_norm = []
    dh = np.ones(4)
    for t in range(T):
        h = act(Wh @ h)
        d = deriv(h)
        dh = d * (Wh.T @ dh)
        grad_norm.append(np.linalg.norm(dh))
    return grad_norm

tanh_grads    = simulate_gradient_propagation('tanh')
sigmoid_grads = simulate_gradient_propagation('sigmoid')
relu_grads    = simulate_gradient_propagation('relu')

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a. Training loss
ax = axes[0, 0]
ax.plot(losses, 'b-', linewidth=2)
ax.set_title('BPTT Training Loss'); ax.set_xlabel('Epoch')
ax.set_ylabel('MSE'); ax.grid(True)

# 4b. Gradient norms over timesteps (last epoch)
ax = axes[0, 1]
final_grads = grad_norms_per_epoch[-1]
ax.bar(range(len(final_grads)), final_grads, color='coral')
ax.set_title('Gradient Norms vs Timestep (BPTT)'); ax.set_xlabel('Timestep (from end)')
ax.set_ylabel('||dh||'); ax.grid(True, axis='y')

# 4c. Gradient heatmap over training
ax = axes[1, 0]
grad_matrix = np.array(grad_norms_per_epoch[::10])
im = ax.imshow(grad_matrix, aspect='auto', cmap='hot')
ax.set_title('Gradient Norms Over Training (every 10 epochs)')
ax.set_xlabel('Timestep'); ax.set_ylabel('Epoch (×10)')
plt.colorbar(im, ax=ax)

# 4d. Vanishing gradient comparison
ax = axes[1, 1]
ax.plot(tanh_grads,    'b-o', linewidth=2, label='Tanh',    markersize=4)
ax.plot(sigmoid_grads, 'r-s', linewidth=2, label='Sigmoid', markersize=4)
ax.plot(relu_grads,    'g-^', linewidth=2, label='ReLU',    markersize=4)
ax.set_title('Vanishing Gradients: Activation Comparison')
ax.set_xlabel('Timestep'); ax.set_ylabel('Gradient Norm')
ax.legend(); ax.grid(True); ax.set_yscale('log')

plt.suptitle('Experiment 15: BPTT Flow and Gradient Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp15.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp15.png")
