"""
Experiment 13: Sequential Dataset Preparation and RNN Prediction
CO3 | Packages: TensorFlow, Keras, NumPy, Pandas, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 13: Sequential Dataset Preparation & RNN Prediction")
print("=" * 60)

np.random.seed(42)

# ── 1. Generate Synthetic Time Series ─────────────────────────
n = 1000
t = np.linspace(0, 10*np.pi, n)
signal = (np.sin(t) + 0.5*np.sin(3*t) + 0.2*np.random.randn(n)).astype(np.float32)

df = pd.DataFrame({'t': t, 'value': signal})
print("\n[1] Time Series Statistics")
print(df['value'].describe().round(4))

# ── 2. Sliding Window Dataset ─────────────────────────────────
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

SEQ_LEN = 50
X_all, y_all = create_sequences(signal, SEQ_LEN)
X_all = X_all[..., np.newaxis]  # (samples, timesteps, features)
print(f"\n[2] Dataset Shapes: X={X_all.shape}, y={y_all.shape}")

split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)

    # ── 3. Build Simple RNN ───────────────────────────────────
    print("\n[3] Building Simple RNN")
    rnn_model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, 1)),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ], name='SimpleRNN')
    rnn_model.summary()
    rnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # ── 4. Train ──────────────────────────────────────────────
    print("\n[4] Training RNN")
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = rnn_model.fit(X_train, y_train, epochs=50, batch_size=64,
                             validation_split=0.1, callbacks=[es], verbose=1)

    # ── 5. Predict ────────────────────────────────────────────
    y_pred = rnn_model.predict(X_test, verbose=0).flatten()
    mse = np.mean((y_pred - y_test)**2)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"\n[5] Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    tf_ok = True

except ImportError:
    print("  TensorFlow not installed — using NumPy AR model.")
    tf_ok = False

    def ar_predict(X_data, order=5):
        X2 = X_data[:, -order:, 0]
        w = np.linalg.lstsq(X_train[:, -order:, 0], y_train, rcond=None)[0]
        return X2 @ w

    y_pred = ar_predict(X_test)
    mse = np.mean((y_pred - y_test)**2)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"  AR Model — Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    history = None

# ── 6. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(t, signal, 'b-', linewidth=1, alpha=0.7, label='Full Signal')
ax.set_title('Synthetic Time Series'); ax.set_xlabel('t')
ax.set_ylabel('Value'); ax.legend(); ax.grid(True)

ax = axes[0, 1]
n_show = 200
ax.plot(range(n_show), y_test[:n_show], 'b-', linewidth=1.5, label='True', alpha=0.8)
ax.plot(range(n_show), y_pred[:n_show], 'r--', linewidth=1.5, label='Predicted', alpha=0.8)
ax.set_title('RNN Predictions vs True Values'); ax.set_xlabel('Sample')
ax.set_ylabel('Value'); ax.legend(); ax.grid(True)

ax = axes[1, 0]
if tf_ok and history:
    ax.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_title('Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.legend(); ax.grid(True)
else:
    ax.axis('off')

ax = axes[1, 1]
errors = y_pred - y_test
ax.hist(errors, bins=40, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(0, color='black', linestyle='--')
ax.set_title('Prediction Error Distribution'); ax.set_xlabel('Error')
ax.set_ylabel('Frequency'); ax.grid(True)

plt.suptitle('Experiment 13: Sequential Dataset & RNN Prediction',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp13.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp13.png")
