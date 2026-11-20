"""
Experiment 09: Multilayer Perceptron Implementation
CO2 | Packages: TensorFlow, Keras, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 09: Multilayer Perceptron (MLP) Implementation")
print("=" * 60)

np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import mnist

    print(f"\n  TensorFlow version: {tf.__version__}")

    # ── 1. Load MNIST ─────────────────────────────────────────
    print("\n[1] Loading MNIST Dataset")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test  = X_test.reshape(-1, 784) / 255.0
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── 2. Build MLP ──────────────────────────────────────────
    print("\n[2] Building MLP")
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64,  activation='relu'),
        layers.Dense(10,  activation='softmax')
    ], name='MLP')

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # ── 3. Train ──────────────────────────────────────────────
    print("\n[3] Training MLP")
    history = model.fit(X_train, y_train, epochs=20,
                        batch_size=128, validation_split=0.1,
                        verbose=1)

    # ── 4. Evaluate ───────────────────────────────────────────
    print("\n[4] Evaluation on Test Set")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")

    # ── 5. Predictions ────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    n_correct = (y_pred == y_test).sum()
    print(f"  Correct Predictions: {n_correct}/{len(y_test)}")

    tf_ok = True

except ImportError:
    print("  TensorFlow not available. Using NumPy MLP.")
    tf_ok = False

    # NumPy MLP fallback
    class NumpyMLP:
        def __init__(self, layers_dims, lr=0.01):
            self.lr = lr
            self.params = {}
            self.losses = []
            for l in range(1, len(layers_dims)):
                self.params[f'W{l}'] = np.random.randn(layers_dims[l-1], layers_dims[l]) * 0.01
                self.params[f'b{l}'] = np.zeros((1, layers_dims[l]))
            self.L = len(layers_dims) - 1

        def relu(self, z): return np.maximum(0, z)
        def softmax(self, z):
            ez = np.exp(z - z.max(axis=1, keepdims=True))
            return ez / ez.sum(axis=1, keepdims=True)

        def forward(self, X):
            self.cache = {'A0': X}
            A = X
            for l in range(1, self.L):
                Z = A @ self.params[f'W{l}'] + self.params[f'b{l}']
                A = self.relu(Z)
                self.cache[f'Z{l}'] = Z; self.cache[f'A{l}'] = A
            Z = A @ self.params[f'W{self.L}'] + self.params[f'b{self.L}']
            A = self.softmax(Z)
            self.cache[f'Z{self.L}'] = Z; self.cache[f'A{self.L}'] = A
            return A

        def compute_loss(self, y_hat, y):
            m = y.shape[0]
            log_probs = -np.log(y_hat[np.arange(m), y] + 1e-10)
            return log_probs.mean()

        def fit(self, X, y, epochs=20, batch_size=256):
            m = X.shape[0]
            for epoch in range(epochs):
                idx = np.random.permutation(m)
                total_loss = 0
                for i in range(0, m, batch_size):
                    Xb = X[idx[i:i+batch_size]]
                    yb = y[idx[i:i+batch_size]]
                    y_hat = self.forward(Xb)
                    total_loss += self.compute_loss(y_hat, yb)
                    # simple backprop
                    mb = Xb.shape[0]
                    dZ = y_hat; dZ[np.arange(mb), yb] -= 1; dZ /= mb
                    for l in range(self.L, 0, -1):
                        dW = self.cache[f'A{l-1}'].T @ dZ
                        db = dZ.sum(0, keepdims=True)
                        if l > 1:
                            dA = dZ @ self.params[f'W{l}'].T
                            dZ = dA * (self.cache[f'Z{l-1}'] > 0)
                        self.params[f'W{l}'] -= self.lr * dW
                        self.params[f'b{l}'] -= self.lr * db
                self.losses.append(total_loss / (m//batch_size))
                if (epoch+1) % 5 == 0:
                    print(f"  Epoch {epoch+1}: loss={self.losses[-1]:.4f}")

    from sklearn.datasets import load_digits
    digits = load_digits()
    X_data = digits.data / 16.0
    y_data = digits.target
    perm = np.random.permutation(len(y_data))
    split = int(0.8 * len(y_data))
    X_train, X_test = X_data[perm[:split]], X_data[perm[split:]]
    y_train, y_test = y_data[perm[:split]], y_data[perm[split:]]

    mlp = NumpyMLP([64, 64, 32, 10], lr=0.1)
    mlp.fit(X_train, y_train, epochs=50)
    pred = np.argmax(mlp.forward(X_test), axis=1)
    print(f"\n  Test Accuracy: {(pred==y_test).mean()*100:.1f}%")

# ── Visualisation ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if tf_ok:
    ax = axes[0, 0]
    ax.plot(history.history['accuracy'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(history.history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    ax.set_title('MLP Accuracy Curves'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_title('MLP Loss Curves'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss'); ax.legend(); ax.grid(True)

    # Sample predictions
    ax = axes[1, 0]
    X_test_imgs = X_test[:16].reshape(-1, 28, 28)
    preds_sample = y_pred[:16]
    for i in range(16):
        ax2 = plt.subplot2grid((2,2), (1,0), fig=fig) if i==0 else None
    ax.axis('off')
    ax.set_title('Sample Predictions')
    for idx in range(10):
        sub = fig.add_subplot(2, 10, 20+idx+1) if False else None

    # confusion-like accuracy by class
    ax = axes[1, 0]
    class_acc = [(y_pred[y_test==c] == c).mean() for c in range(10)]
    ax.bar(range(10), class_acc, color='steelblue')
    ax.set_title('Per-Class Accuracy'); ax.set_xlabel('Digit Class')
    ax.set_ylabel('Accuracy'); ax.set_xticks(range(10))
    for i, v in enumerate(class_acc):
        ax.text(i, v+0.005, f'{v:.2f}', ha='center', fontsize=8)
    ax.grid(True, axis='y')

    # Sample images with predictions
    ax = axes[1, 1]
    ax.axis('off')
    fig2, ax2_arr = plt.subplots(2, 5, figsize=(10, 4))
    for i, axx in enumerate(ax2_arr.flat):
        img = X_test[i].reshape(28, 28)
        axx.imshow(img, cmap='gray')
        axx.set_title(f'Pred:{y_pred[i]}\nTrue:{y_test[i]}',
                      color='green' if y_pred[i]==y_test[i] else 'red', fontsize=8)
        axx.axis('off')
    fig2.suptitle('Sample MNIST Predictions', fontsize=12)
    fig2.tight_layout()
    fig2.savefig('output_exp09_samples.png', dpi=100, bbox_inches='tight')
    plt.close(fig2)

else:
    ax = axes[0, 0]
    ax.plot(mlp.losses, 'b-', linewidth=2)
    ax.set_title('Training Loss'); ax.grid(True)

    for axi in [axes[0,1], axes[1,0], axes[1,1]]:
        axi.axis('off')

plt.suptitle('Experiment 09: Multilayer Perceptron (MLP)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp09.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
