"""
Experiment 08: Two-Layer Neural Network with Backpropagation
CO2 | Packages: NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 08: Two-Layer Neural Network with Backpropagation")
print("=" * 60)

np.random.seed(42)

# ── 1. Activation Functions ───────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# ── 2. Two-Layer Neural Network ───────────────────────────────
class TwoLayerNN:
    def __init__(self, n_input, n_hidden, n_output, lr=0.01):
        self.lr = lr
        # Xavier init
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0/n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0/n_hidden)
        self.b2 = np.zeros((1, n_output))
        self.train_loss = []

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        eps = 1e-10
        return -np.mean(y * np.log(y_hat + eps) + (1-y) * np.log(1-y_hat + eps))

    def backward(self, X, y):
        m = X.shape[0]
        dA2 = self.A2 - y
        dZ2 = dA2 * sigmoid_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = dZ2.mean(0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * sigmoid_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = dZ1.mean(0, keepdims=True)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y, y_hat)
            self.train_loss.append(loss)
            self.backward(X, y)
            if verbose and (epoch+1) % 200 == 0:
                acc = ((y_hat > 0.5) == y).mean()
                print(f"  Epoch {epoch+1:5d}: loss={loss:.4f}, acc={acc*100:.1f}%")
        return self

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# ── 3. XOR Problem ────────────────────────────────────────────
print("\n[3] XOR Problem (requires hidden layer)")
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)
nn_xor = TwoLayerNN(2, 4, 1, lr=1.0)
nn_xor.fit(X_xor, y_xor, epochs=5000, verbose=False)
preds = nn_xor.predict(X_xor)
print(f"  Predictions: {preds.flatten()}, Expected: {y_xor.flatten().astype(int)}")
print(f"  Accuracy: {(preds == y_xor).mean()*100:.1f}%")

# ── 4. 2D Classification ──────────────────────────────────────
print("\n[4] 2D Circular Classification")
n = 200
theta = np.linspace(0, 2*np.pi, n)
X_inner = np.column_stack([0.5*np.cos(theta) + 0.1*np.random.randn(n),
                            0.5*np.sin(theta) + 0.1*np.random.randn(n)])
X_outer = np.column_stack([2*np.cos(theta) + 0.2*np.random.randn(n),
                            2*np.sin(theta) + 0.2*np.random.randn(n)])
X2d = np.vstack([X_inner, X_outer])
y2d = np.hstack([np.ones(n), np.zeros(n)]).reshape(-1, 1)
perm = np.random.permutation(2*n)
X2d, y2d = X2d[perm], y2d[perm]

nn2d = TwoLayerNN(2, 8, 1, lr=0.5)
print("  Training...")
nn2d.fit(X2d, y2d, epochs=2000, verbose=True)
acc_2d = (nn2d.predict(X2d) == y2d).mean()
print(f"  Final Accuracy: {acc_2d*100:.1f}%")

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a. Loss curve XOR
ax = axes[0, 0]
ax.plot(nn_xor.train_loss, 'b-', linewidth=1.5)
ax.set_title('XOR Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
ax.grid(True)

# 5b. Decision boundary 2D
ax = axes[0, 1]
x_min, x_max = X2d[:,0].min()-0.5, X2d[:,0].max()+0.5
y_min, y_max = X2d[:,1].min()-0.5, X2d[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = nn2d.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.6)
ax.scatter(X2d[:,0], X2d[:,1], c=y2d.flatten(), cmap='RdYlBu',
           edgecolors='k', s=10, alpha=0.5)
ax.set_title('2D Decision Boundary (Neural Net)'); ax.grid(True)

# 5c. Loss curve 2D
ax = axes[1, 0]
ax.plot(nn2d.train_loss, 'r-', linewidth=1.5)
ax.set_title('2D Classification Training Loss'); ax.set_xlabel('Epoch')
ax.set_ylabel('BCE Loss'); ax.grid(True)

# 5d. Backprop gradient flow diagram (text)
ax = axes[1, 1]
ax.axis('off')
bprop_text = """
Backpropagation Chain Rule:

Forward Pass:
  Z1 = X·W1 + b1
  A1 = sigmoid(Z1)
  Z2 = A1·W2 + b2
  A2 = sigmoid(Z2)  ← output

Loss: L = BCE(y, A2)

Backward Pass:
  dL/dA2 = A2 - y
  dL/dW2 = A1ᵀ · (dL/dA2 · σ'(Z2))
  dL/db2 = mean(dL/dA2 · σ'(Z2))
  dL/dW1 = Xᵀ · (dA1 · σ'(Z1))
  dL/db1 = mean(dA1 · σ'(Z1))

Update: W -= lr * dL/dW
"""
ax.text(0.05, 0.95, bprop_text, transform=ax.transAxes, va='top',
        fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('Backpropagation Summary')

plt.suptitle('Experiment 08: Two-Layer NN with Backpropagation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp08.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp08.png")
