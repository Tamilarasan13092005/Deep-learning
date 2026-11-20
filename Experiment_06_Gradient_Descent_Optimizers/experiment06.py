"""
Experiment 06: Gradient Descent, SGD, Momentum and Adam Comparison
CO1 | Packages: NumPy, TensorFlow, Keras, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 06: Gradient Descent Optimizers Comparison")
print("=" * 60)

np.random.seed(42)

# ── 1. Loss Surface ───────────────────────────────────────────
def loss_fn(w):
    return w**2 + 10 * np.sin(w)**2

def grad_fn(w):
    return 2*w + 20 * np.sin(w) * np.cos(w)

# ── 2. Optimizer Implementations ─────────────────────────────
def vanilla_gd(w_init, lr, n_steps):
    w = w_init; history = [w]
    for _ in range(n_steps):
        w -= lr * grad_fn(w)
        history.append(w)
    return np.array(history)

def sgd_momentum(w_init, lr, momentum, n_steps):
    w = w_init; v = 0; history = [w]
    for _ in range(n_steps):
        v = momentum * v - lr * grad_fn(w)
        w += v
        history.append(w)
    return np.array(history)

def adam(w_init, lr, beta1, beta2, eps, n_steps):
    w = w_init; m = 0; v = 0; history = [w]
    for t in range(1, n_steps+1):
        g = grad_fn(w)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w)
    return np.array(history)

def rmsprop(w_init, lr, decay, eps, n_steps):
    w = w_init; v = 0; history = [w]
    for _ in range(n_steps):
        g = grad_fn(w)
        v = decay*v + (1-decay)*g**2
        w -= lr * g / (np.sqrt(v) + eps)
        history.append(w)
    return np.array(history)

w0 = 4.0; n_steps = 100

gd_hist   = vanilla_gd(w0, lr=0.1, n_steps=n_steps)
mom_hist  = sgd_momentum(w0, lr=0.1, momentum=0.9, n_steps=n_steps)
adam_hist = adam(w0, lr=0.5, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=n_steps)
rms_hist  = rmsprop(w0, lr=0.3, decay=0.9, eps=1e-8, n_steps=n_steps)

print("\n[1] Final Values After Optimization")
for name, hist in [('GD', gd_hist), ('Momentum', mom_hist),
                   ('Adam', adam_hist), ('RMSProp', rms_hist)]:
    print(f"  {name:12s}: w={hist[-1]:.6f}, loss={loss_fn(hist[-1]):.6f}")

# ── 3. MLP Training with TF optimizers ───────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras

    # Synthetic regression dataset
    X = np.random.randn(500, 10).astype(np.float32)
    y = (X @ np.random.randn(10, 1) + np.random.randn(500, 1) * 0.1).astype(np.float32)

    histories = {}
    optimizer_configs = {
        'SGD':      keras.optimizers.SGD(learning_rate=0.01),
        'Momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'RMSProp':  keras.optimizers.RMSprop(learning_rate=0.01),
        'Adam':     keras.optimizers.Adam(learning_rate=0.01),
    }
    for name, opt in optimizer_configs.items():
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer=opt, loss='mse')
        h = model.fit(X, y, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
        histories[name] = h.history

    tf_available = True
    print("\n[2] TF Training complete — validation loss:")
    for name, h in histories.items():
        print(f"  {name:12s}: val_loss={h['val_loss'][-1]:.4f}")
except ImportError:
    tf_available = False
    print("\n[2] TensorFlow not installed — skipping NN training.")

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a. Loss surface + paths
ax = axes[0, 0]
w_vals = np.linspace(-5, 5, 300)
ax.plot(w_vals, loss_fn(w_vals), 'k-', linewidth=2, label='Loss')
ax.plot(gd_hist[:20], loss_fn(gd_hist[:20]), 'b.-', label='GD', alpha=0.8)
ax.plot(mom_hist[:20], loss_fn(mom_hist[:20]), 'r.-', label='Momentum', alpha=0.8)
ax.plot(adam_hist[:20], loss_fn(adam_hist[:20]), 'g.-', label='Adam', alpha=0.8)
ax.set_title('Optimizer Paths on Loss Surface'); ax.legend(); ax.grid(True)

# 4b. Loss convergence (1D)
ax = axes[0, 1]
ax.plot(loss_fn(gd_hist), 'b-', label='GD', linewidth=2)
ax.plot(loss_fn(mom_hist), 'r-', label='Momentum', linewidth=2)
ax.plot(loss_fn(adam_hist), 'g-', label='Adam', linewidth=2)
ax.plot(loss_fn(rms_hist), 'm-', label='RMSProp', linewidth=2)
ax.set_title('Loss Convergence (1D Problem)'); ax.set_xlabel('Step')
ax.set_ylabel('Loss'); ax.legend(); ax.grid(True); ax.set_yscale('log')

# 4c. & 4d. TF training curves
if tf_available:
    ax = axes[1, 0]
    for name, h in histories.items():
        ax.plot(h['loss'], label=name, linewidth=1.5)
    ax.set_title('Training Loss (NN)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    for name, h in histories.items():
        ax.plot(h['val_loss'], label=name, linewidth=1.5)
    ax.set_title('Validation Loss (NN)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)
else:
    # Fallback: w-history plot
    ax = axes[1, 0]
    for name, hist in [('GD', gd_hist), ('Momentum', mom_hist),
                       ('Adam', adam_hist), ('RMSProp', rms_hist)]:
        ax.plot(hist, label=name)
    ax.set_title('Weight History'); ax.legend(); ax.grid(True)
    axes[1, 1].axis('off')

plt.suptitle('Experiment 06: Gradient Descent Optimizers Comparison',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp06.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp06.png")
