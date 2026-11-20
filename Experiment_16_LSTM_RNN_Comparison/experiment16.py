"""
Experiment 16: LSTM and RNN Performance Comparison
CO3 | Packages: TensorFlow, Keras, NumPy, Pandas, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 16: LSTM and RNN Performance Comparison")
print("=" * 60)

np.random.seed(42)

# ── 1. Time Series Data ───────────────────────────────────────
n = 2000
t = np.linspace(0, 20*np.pi, n)
signal = (np.sin(t) + 0.3*np.sin(5*t) + 0.1*np.cumsum(np.random.randn(n)*0.1)).astype(np.float32)
signal = (signal - signal.mean()) / signal.std()

SEQ_LEN = 60

def make_sequences(data, seq_len):
    X = np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)], dtype=np.float32)
    y = np.array([data[i+seq_len] for i in range(len(data)-seq_len)], dtype=np.float32)
    return X[..., np.newaxis], y

X_all, y_all = make_sequences(signal, SEQ_LEN)
split = int(0.8*len(X_all))
X_tr, X_te = X_all[:split], X_all[split:]
y_tr, y_te = y_all[:split], y_all[split:]

print(f"\n[1] Dataset: X_train={X_tr.shape}, X_test={X_te.shape}")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)

    models_configs = {
        'SimpleRNN': keras.Sequential([
            layers.Input(shape=(SEQ_LEN, 1)),
            layers.SimpleRNN(64),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ]),
        'LSTM': keras.Sequential([
            layers.Input(shape=(SEQ_LEN, 1)),
            layers.LSTM(64),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ]),
        'Stacked LSTM': keras.Sequential([
            layers.Input(shape=(SEQ_LEN, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ]),
        'Bidirectional LSTM': keras.Sequential([
            layers.Input(shape=(SEQ_LEN, 1)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ]),
    }

    histories = {}
    results = {}
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    print("\n[2] Training Models")
    for name, model in models_configs.items():
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        h = model.fit(X_tr, y_tr, epochs=50, batch_size=64,
                      validation_split=0.1, callbacks=[es], verbose=0)
        y_pred = model.predict(X_te, verbose=0).flatten()
        mse = np.mean((y_pred - y_te)**2)
        mae = np.mean(np.abs(y_pred - y_te))
        histories[name] = h.history
        results[name] = {'mse': mse, 'mae': mae, 'pred': y_pred,
                         'params': model.count_params(), 'epochs': len(h.history['loss'])}
        print(f"  {name:25s}: MSE={mse:.4f}, MAE={mae:.4f}, params={model.count_params():,}")

    tf_ok = True

except ImportError:
    print("  TensorFlow not installed — using NumPy LSTM demo.")
    tf_ok = False

    class NumpyLSTMCell:
        def __init__(self, input_size, hidden_size):
            s = np.sqrt(1.0/hidden_size)
            self.Wf = np.random.randn(input_size+hidden_size, hidden_size)*s
            self.Wi = np.random.randn(input_size+hidden_size, hidden_size)*s
            self.Wc = np.random.randn(input_size+hidden_size, hidden_size)*s
            self.Wo = np.random.randn(input_size+hidden_size, hidden_size)*s
            self.bf = np.zeros(hidden_size)
            self.bi = np.zeros(hidden_size)
            self.bc = np.zeros(hidden_size)
            self.bo = np.zeros(hidden_size)

        def step(self, x, h, c):
            xh = np.concatenate([x, h])
            sig = lambda z: 1/(1+np.exp(-z))
            f = sig(xh @ self.Wf + self.bf)
            i = sig(xh @ self.Wi + self.bi)
            g = np.tanh(xh @ self.Wc + self.bc)
            o = sig(xh @ self.Wo + self.bo)
            c_new = f*c + i*g
            h_new = o * np.tanh(c_new)
            return h_new, c_new

    cell = NumpyLSTMCell(1, 16)
    h, c = np.zeros(16), np.zeros(16)
    for t_ in range(SEQ_LEN):
        h, c = cell.step(X_tr[0, t_, :], h, c)
    print(f"  LSTM cell demo: final h norm={np.linalg.norm(h):.4f}")
    histories = None; results = None

# ── 3. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if tf_ok:
    ax = axes[0, 0]
    for name, h in histories.items():
        ax.plot(h['val_loss'], linewidth=2, label=name)
    ax.set_title('Validation Loss Comparison'); ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE'); ax.legend(fontsize=8); ax.grid(True)

    ax = axes[0, 1]
    n_show = 150
    colors = ['blue', 'red', 'green', 'purple']
    ax.plot(y_te[:n_show], 'k-', linewidth=2, label='True', alpha=0.9)
    for (name, res), color in zip(results.items(), colors):
        ax.plot(res['pred'][:n_show], '--', linewidth=1.5, label=name, color=color, alpha=0.7)
    ax.set_title('Predictions vs True'); ax.set_xlabel('Sample')
    ax.set_ylabel('Value'); ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1, 0]
    names = list(results.keys())
    mses = [results[n]['mse'] for n in names]
    bars = ax.bar(names, mses, color=['steelblue','coral','green','purple'])
    ax.set_title('Test MSE Comparison'); ax.set_ylabel('MSE')
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.0005, f'{val:.4f}', ha='center', fontsize=8)
    ax.grid(True, axis='y')

    ax = axes[1, 1]
    params = [results[n]['params'] for n in names]
    ax.barh(names, params, color='lightblue', edgecolor='navy')
    ax.set_title('Model Parameter Count'); ax.set_xlabel('# Parameters')
    for i, v in enumerate(params):
        ax.text(v+10, i, f'{v:,}', va='center', fontsize=9)
    ax.grid(True, axis='x')

else:
    ax = axes[0, 0]
    ax.plot(signal[:200], 'b-', linewidth=1.5, label='Signal')
    ax.set_title('Time Series'); ax.legend(); ax.grid(True)
    for axi in [axes[0,1], axes[1,0], axes[1,1]]:
        axi.axis('off')

plt.suptitle('Experiment 16: LSTM and RNN Performance Comparison',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp16.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp16.png")
