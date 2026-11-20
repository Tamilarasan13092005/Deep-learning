"""
Experiment 12: CNN Architecture and Hyperparameter Comparison
CO2 | Packages: TensorFlow, Keras, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 12: CNN Architecture & Hyperparameter Comparison")
print("=" * 60)

np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import mnist

    tf.random.set_seed(42)

    # ── 1. Load MNIST ─────────────────────────────────────────
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis].astype('float32') / 255.0
    X_test  = X_test[..., np.newaxis].astype('float32') / 255.0

    def build_cnn(filters1=32, filters2=64, kernel=3,
                  dropout=0.5, optimizer='adam', name='CNN'):
        m = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(filters1, kernel, activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters2, kernel, activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(10, activation='softmax')
        ], name=name)
        m.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    # ── 2. Architecture Comparison ────────────────────────────
    print("\n[2] Architecture Comparison")
    arch_configs = {
        'Small (16-32)': dict(filters1=16, filters2=32, name='Small_CNN'),
        'Medium (32-64)': dict(filters1=32, filters2=64, name='Medium_CNN'),
        'Large (64-128)': dict(filters1=64, filters2=128, name='Large_CNN'),
    }
    arch_histories = {}
    arch_results = {}
    for name, cfg in arch_configs.items():
        m = build_cnn(**cfg)
        h = m.fit(X_train, y_train, epochs=10, batch_size=256,
                  validation_split=0.1, verbose=0)
        loss, acc = m.evaluate(X_test, y_test, verbose=0)
        arch_histories[name] = h.history
        arch_results[name] = {'acc': acc, 'params': m.count_params()}
        print(f"  {name:20s}: acc={acc*100:.2f}%, params={m.count_params():,}")

    # ── 3. Optimizer Comparison ───────────────────────────────
    print("\n[3] Optimizer Comparison")
    opt_histories = {}
    for opt_name in ['adam', 'sgd', 'rmsprop']:
        m = build_cnn(optimizer=opt_name, name=f'CNN_{opt_name}')
        h = m.fit(X_train, y_train, epochs=10, batch_size=256,
                  validation_split=0.1, verbose=0)
        loss, acc = m.evaluate(X_test, y_test, verbose=0)
        opt_histories[opt_name] = h.history
        print(f"  {opt_name:12s}: acc={acc*100:.2f}%")

    # ── 4. Kernel Size Comparison ─────────────────────────────
    print("\n[4] Kernel Size Comparison")
    kernel_histories = {}
    for k in [3, 5, 7]:
        m = build_cnn(kernel=k, name=f'CNN_k{k}')
        h = m.fit(X_train, y_train, epochs=10, batch_size=256,
                  validation_split=0.1, verbose=0)
        loss, acc = m.evaluate(X_test, y_test, verbose=0)
        kernel_histories[f'k={k}'] = h.history
        print(f"  Kernel {k}x{k}: acc={acc*100:.2f}%")

    tf_ok = True

except ImportError:
    print("  TensorFlow not installed — demo with filter visualization.")
    tf_ok = False

# ── 5. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if tf_ok:
    # 5a. Architecture val_accuracy
    ax = axes[0, 0]
    for name, h in arch_histories.items():
        ax.plot(h['val_accuracy'], linewidth=2, label=name)
    ax.set_title('Architecture Comparison (Val Acc)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(True)

    # 5b. Architecture params vs acc
    ax = axes[0, 1]
    names = list(arch_results.keys())
    params = [arch_results[n]['params'] for n in names]
    accs   = [arch_results[n]['acc'] for n in names]
    ax.scatter(params, accs, s=200, c='steelblue', zorder=5)
    for n, p, a in zip(names, params, accs):
        ax.annotate(n.split(' ')[0], (p, a), textcoords='offset points',
                    xytext=(5, 5), fontsize=9)
    ax.set_title('Parameters vs Accuracy'); ax.set_xlabel('# Parameters')
    ax.set_ylabel('Test Accuracy'); ax.grid(True)

    # 5c. Optimizer comparison
    ax = axes[1, 0]
    for name, h in opt_histories.items():
        ax.plot(h['val_accuracy'], linewidth=2, label=name.upper())
    ax.set_title('Optimizer Comparison (Val Acc)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(True)

    # 5d. Kernel comparison
    ax = axes[1, 1]
    for name, h in kernel_histories.items():
        ax.plot(h['val_accuracy'], linewidth=2, label=name)
    ax.set_title('Kernel Size Comparison'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy'); ax.legend(); ax.grid(True)

else:
    # Convolution kernel visualization
    kernels_demo = {
        'Identity':    np.array([[0,0,0],[0,1,0],[0,0,0]]),
        'Edge Detect': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
        'Sharpen':     np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        'Emboss':      np.array([[-2,-1,0],[-1,1,1],[0,1,2]]),
    }
    for i, (kname, kern) in enumerate(kernels_demo.items()):
        ax = axes[i//2, i%2]
        im = ax.imshow(kern, cmap='RdBu', vmin=-3, vmax=3)
        ax.set_title(f'{kname} Kernel')
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(kern[r,c]), ha='center', va='center', fontsize=14)
        plt.colorbar(im, ax=ax)

plt.suptitle('Experiment 12: CNN Architecture & Hyperparameter Comparison',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp12.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp12.png")
