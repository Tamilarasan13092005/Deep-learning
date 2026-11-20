"""
Experiment 17: RNN Gradient Analysis and Gradient Clipping
CO3 | Packages: TensorFlow, Keras, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 17: RNN Gradient Analysis and Gradient Clipping")
print("=" * 60)

np.random.seed(42)

# ── 1. Gradient Explosion Demo ────────────────────────────────
def simulate_gradient_explosion(T=50, Wh_scale=1.5):
    Wh = np.random.randn(4, 4) * Wh_scale
    h = np.ones(4) * 0.5
    grad = np.ones(4)
    norms = []
    for _ in range(T):
        h = np.tanh(Wh @ h)
        grad = Wh.T @ (grad * (1 - h**2))
        norms.append(np.linalg.norm(grad))
    return norms

def clip_gradient(grad, threshold):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * threshold / norm
    return grad

print("\n[1] Gradient Norm Without Clipping")
norms_explode = simulate_gradient_explosion(T=50, Wh_scale=1.5)
norms_vanish  = simulate_gradient_explosion(T=50, Wh_scale=0.5)
print(f"  Exploding (Wh~1.5): max norm = {max(norms_explode):.2f}")
print(f"  Vanishing (Wh~0.5): min norm = {min(norms_vanish):.6f}")

# ── 2. Clipping Comparison ────────────────────────────────────
print("\n[2] Gradient Clipping Thresholds")
def simulate_with_clipping(T=50, threshold=1.0):
    Wh = np.random.randn(4, 4) * 1.5
    h = np.ones(4) * 0.5
    grad = np.ones(4)
    norms_no_clip, norms_clip = [], []
    for _ in range(T):
        h = np.tanh(Wh @ h)
        grad_step = Wh.T @ (grad * (1 - h**2))
        norms_no_clip.append(np.linalg.norm(grad_step))
        grad_clipped = clip_gradient(grad_step.copy(), threshold)
        norms_clip.append(np.linalg.norm(grad_clipped))
        grad = grad_step
    return norms_no_clip, norms_clip

norms_nc, norms_c1 = simulate_with_clipping(threshold=1.0)
_, norms_c5 = simulate_with_clipping(threshold=5.0)

for thresh, norms in [(1.0, norms_c1), (5.0, norms_c5)]:
    print(f"  Threshold={thresh}: max clipped norm = {max(norms):.4f}")

# ── 3. TF Training with/without Clipping ─────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)

    n = 1000
    X_data = np.random.randn(n, 20, 1).astype(np.float32)
    y_data = np.random.randn(n, 1).astype(np.float32)

    histories_tf = {}
    for clip_val in [None, 1.0, 5.0]:
        name = f'clip={clip_val}' if clip_val else 'no_clip'
        opt = keras.optimizers.Adam(1e-3, clipnorm=clip_val) if clip_val else keras.optimizers.Adam(1e-3)
        model = keras.Sequential([
            layers.Input(shape=(20, 1)),
            layers.SimpleRNN(32),
            layers.Dense(1)
        ])
        model.compile(optimizer=opt, loss='mse')
        h = model.fit(X_data, y_data, epochs=50, batch_size=32,
                      validation_split=0.2, verbose=0)
        histories_tf[name] = h.history
        print(f"  {name:15s}: final val_loss={h.history['val_loss'][-1]:.4f}")

    tf_ok = True

except ImportError:
    tf_ok = False
    print("  TensorFlow not installed — gradient clipping demo only.")

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(norms_explode, 'r-', linewidth=2, label='Exploding (Wh~1.5)')
ax.plot(norms_vanish, 'b-', linewidth=2, label='Vanishing (Wh~0.5)')
ax.set_title('Gradient Explosion vs Vanishing'); ax.set_xlabel('Timestep')
ax.set_ylabel('Gradient Norm'); ax.legend(); ax.grid(True)
ax.set_yscale('symlog')

ax = axes[0, 1]
ax.plot(norms_nc, 'r-', linewidth=2, label='No Clipping')
ax.plot(norms_c1, 'g-', linewidth=2, label='Clip (threshold=1)')
ax.plot(norms_c5, 'b-', linewidth=2, label='Clip (threshold=5)')
ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
ax.axhline(5.0, color='blue',  linestyle='--', alpha=0.5)
ax.set_title('Effect of Gradient Clipping'); ax.set_xlabel('Timestep')
ax.set_ylabel('Gradient Norm'); ax.legend(); ax.grid(True)
ax.set_yscale('symlog')

if tf_ok:
    ax = axes[1, 0]
    for name, h in histories_tf.items():
        ax.plot(h['loss'], linewidth=2, label=name)
    ax.set_title('Training Loss (TF RNN)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    for name, h in histories_tf.items():
        ax.plot(h['val_loss'], linewidth=2, label=name)
    ax.set_title('Validation Loss (TF RNN)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)
else:
    ax = axes[1, 0]
    # Norm distribution with/without clipping
    ax.hist(norms_nc, bins=20, alpha=0.6, color='red', label='No clip', density=True)
    ax.hist(norms_c1, bins=20, alpha=0.6, color='green', label='Clip=1', density=True)
    ax.set_title('Gradient Norm Distribution'); ax.legend(); ax.grid(True)
    axes[1, 1].axis('off')

plt.suptitle('Experiment 17: RNN Gradient Analysis & Gradient Clipping',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp17.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp17.png")
