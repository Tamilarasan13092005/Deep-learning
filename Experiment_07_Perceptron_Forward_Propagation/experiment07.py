"""
Experiment 07: Perceptron and Forward Propagation
CO2 | Packages: NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 07: Perceptron and Forward Propagation")
print("=" * 60)

np.random.seed(42)

# ── 1. Perceptron Class ───────────────────────────────────────
class Perceptron:
    def __init__(self, lr=0.1, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_ = []

    def step(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_ = []
        for epoch in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                y_hat = self.step(z)
                update = self.lr * (yi - y_hat)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.step(z)

# ── 2. AND Gate ───────────────────────────────────────────────
print("\n[2] Perceptron on AND Gate")
X_and = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_and = np.array([0, 0, 0, 1])
p_and = Perceptron(lr=0.1, n_epochs=20)
p_and.fit(X_and, y_and)
print(f"  Weights: {p_and.weights}, Bias: {p_and.bias:.2f}")
print(f"  Predictions: {p_and.predict(X_and)}, Expected: {y_and}")
print(f"  Accuracy: {(p_and.predict(X_and)==y_and).mean()*100:.1f}%")

# ── 3. OR Gate ────────────────────────────────────────────────
print("\n[3] Perceptron on OR Gate")
y_or = np.array([0, 1, 1, 1])
p_or = Perceptron(lr=0.1, n_epochs=20)
p_or.fit(X_and, y_or)
print(f"  Predictions: {p_or.predict(X_and)}, Expected: {y_or}")

# ── 4. XOR (limitation) ───────────────────────────────────────
print("\n[4] XOR Gate — Not Linearly Separable")
y_xor = np.array([0, 1, 1, 0])
p_xor = Perceptron(lr=0.1, n_epochs=100)
p_xor.fit(X_and, y_xor)
print(f"  Predictions: {p_xor.predict(X_and)}, Expected: {y_xor}")
print(f"  Final Errors: {p_xor.errors_[-1]} (non-zero = not solvable by perceptron)")

# ── 5. 2D Classification ──────────────────────────────────────
print("\n[5] 2D Linearly Separable Dataset")
n = 100
c0 = np.random.randn(n, 2) - 2
c1 = np.random.randn(n, 2) + 2
X2d = np.vstack([c0, c1])
y2d = np.hstack([np.zeros(n), np.ones(n)])
p2d = Perceptron(lr=0.01, n_epochs=50)
p2d.fit(X2d, y2d)
acc = (p2d.predict(X2d) == y2d).mean()
print(f"  Accuracy on 2D data: {acc*100:.1f}%")

# ── 6. Forward Propagation Demo ───────────────────────────────
print("\n[6] Forward Propagation through 1 Neuron")
x_in = np.array([0.5, -0.3, 0.8])
w = np.array([0.4, 0.2, -0.1])
b = 0.1
z = np.dot(w, x_in) + b
sigmoid = lambda z: 1 / (1 + np.exp(-z))
relu = lambda z: np.maximum(0, z)
print(f"  Input x={x_in}, Weights w={w}, Bias b={b}")
print(f"  z = w·x + b = {z:.4f}")
print(f"  Sigmoid(z) = {sigmoid(z):.4f}")
print(f"  ReLU(z)    = {relu(z):.4f}")
print(f"  Tanh(z)    = {np.tanh(z):.4f}")

# ── 7. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def plot_decision_boundary(ax, model, X, y, title):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdYlBu', edgecolors='k', s=20)
    ax.set_title(title); ax.grid(True)

# 7a. AND gate boundary
plot_decision_boundary(axes[0,0], p_and, X_and, y_and, 'AND Gate Decision Boundary')

# 7b. 2D data boundary
plot_decision_boundary(axes[0,1], p2d, X2d, y2d, '2D Perceptron Decision Boundary')

# 7c. Training errors
ax = axes[1, 0]
ax.plot(p_and.errors_, 'b-o', label='AND')
ax.plot(p_or.errors_,  'r-o', label='OR')
ax.plot(p_xor.errors_[:30], 'g-o', label='XOR (fails)')
ax.set_title('Training Errors per Epoch'); ax.set_xlabel('Epoch')
ax.set_ylabel('Errors'); ax.legend(); ax.grid(True)

# 7d. Activation functions
ax = axes[1, 1]
z_vals = np.linspace(-5, 5, 300)
ax.plot(z_vals, sigmoid(z_vals), 'b-', linewidth=2, label='Sigmoid')
ax.plot(z_vals, relu(z_vals), 'r-', linewidth=2, label='ReLU')
ax.plot(z_vals, np.tanh(z_vals), 'g-', linewidth=2, label='Tanh')
ax.plot(z_vals, np.where(z_vals>=0, 1, 0), 'k--', linewidth=1.5, label='Step')
ax.set_title('Activation Functions'); ax.legend(); ax.grid(True)
ax.set_ylim(-1.5, 2)

plt.suptitle('Experiment 07: Perceptron and Forward Propagation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp07.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp07.png")
