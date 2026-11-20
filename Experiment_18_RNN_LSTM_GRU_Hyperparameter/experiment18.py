"""
Experiment 18: RNN, LSTM and GRU Hyperparameter Experimentation
CO3 | Packages: TensorFlow, Keras, NumPy, KerasTuner, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 18: RNN, LSTM and GRU Hyperparameter Experimentation")
print("=" * 60)

np.random.seed(42)

n = 1500
t = np.linspace(0, 15*np.pi, n)
signal = (np.sin(t) + 0.4*np.cos(3*t)).astype(np.float32)
signal = (signal - signal.mean()) / signal.std()
SEQ_LEN = 40

def make_sequences(data, seq_len):
    X = np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)], dtype=np.float32)[..., np.newaxis]
    y = np.array([data[i+seq_len] for i in range(len(data)-seq_len)], dtype=np.float32)
    return X, y

X_all, y_all = make_sequences(signal, SEQ_LEN)
split = int(0.8*len(X_all))
X_tr, X_te = X_all[:split], X_all[split:]
y_tr, y_te = y_all[:split], y_all[split:]

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)

    # ── 1. Manual Grid Search ─────────────────────────────────
    print("\n[1] Grid Search: RNN Type × Units × Learning Rate")
    grid_results = []
    rnn_types = {'SimpleRNN': layers.SimpleRNN, 'LSTM': layers.LSTM, 'GRU': layers.GRU}
    units_list = [32, 64]
    lr_list = [1e-3, 1e-2]

    for rnn_name, rnn_layer in rnn_types.items():
        for units in units_list:
            for lr in lr_list:
                model = keras.Sequential([
                    layers.Input(shape=(SEQ_LEN, 1)),
                    rnn_layer(units),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1)
                ])
                model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
                h = model.fit(X_tr, y_tr, epochs=20, batch_size=64,
                              validation_split=0.1, verbose=0)
                y_pred = model.predict(X_te, verbose=0).flatten()
                mse = np.mean((y_pred - y_te)**2)
                grid_results.append({
                    'rnn': rnn_name, 'units': units, 'lr': lr,
                    'val_mse': h.history['val_loss'][-1], 'test_mse': mse
                })
                print(f"  {rnn_name:12s} units={units} lr={lr:.0e}: test_mse={mse:.4f}")

    # Sort by test MSE
    grid_results.sort(key=lambda x: x['test_mse'])
    print(f"\n  Best config: {grid_results[0]}")

    # ── 2. Try KerasTuner ─────────────────────────────────────
    try:
        import keras_tuner as kt

        def build_model(hp):
            rnn_type = hp.Choice('rnn_type', ['SimpleRNN', 'LSTM', 'GRU'])
            units    = hp.Int('units', 16, 128, step=16)
            lr       = hp.Float('lr', 1e-4, 1e-2, sampling='log')
            dropout  = hp.Float('dropout', 0.0, 0.5)
            rnn_map  = {'SimpleRNN': layers.SimpleRNN, 'LSTM': layers.LSTM, 'GRU': layers.GRU}
            model = keras.Sequential([
                layers.Input(shape=(SEQ_LEN, 1)),
                rnn_map[rnn_type](units),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
            model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
            return model

        tuner = kt.RandomSearch(build_model, objective='val_loss',
                                max_trials=5, directory='kt_exp18',
                                project_name='rnn_tuning', overwrite=True)
        tuner.search(X_tr, y_tr, epochs=10, validation_split=0.1, verbose=0)
        best_hp = tuner.get_best_hyperparameters(1)[0]
        print(f"\n  KerasTuner best: {best_hp.values}")
        kt_ok = True
    except ImportError:
        kt_ok = False
        print("\n  KerasTuner not installed — skipping AutoML search.")

    tf_ok = True

except ImportError:
    print("  TensorFlow not installed — demo comparison only.")
    tf_ok = False
    grid_results = [
        {'rnn': 'SimpleRNN', 'units': 32, 'lr': 1e-3, 'test_mse': 0.12},
        {'rnn': 'LSTM',      'units': 64, 'lr': 1e-3, 'test_mse': 0.08},
        {'rnn': 'GRU',       'units': 64, 'lr': 1e-3, 'test_mse': 0.07},
    ]

# ── 3. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a. Test MSE heatmap style: RNN type vs units
ax = axes[0, 0]
rnn_names = ['SimpleRNN', 'LSTM', 'GRU']
units_vals = [32, 64]
if tf_ok:
    matrix = np.zeros((len(rnn_names), len(units_vals)))
    for r in grid_results:
        if r['lr'] == 1e-3:  # filter one LR for clarity
            ri = rnn_names.index(r['rnn'])
            ui = units_vals.index(r['units'])
            matrix[ri, ui] = r['test_mse']
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(units_vals))); ax.set_xticklabels(units_vals)
    ax.set_yticks(range(len(rnn_names)));  ax.set_yticklabels(rnn_names)
    ax.set_title('Test MSE: RNN Type vs Units (lr=1e-3)'); ax.set_xlabel('Units')
    for i in range(len(rnn_names)):
        for j in range(len(units_vals)):
            ax.text(j, i, f'{matrix[i,j]:.4f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)

# 3b. Bar chart of all configs
ax = axes[0, 1]
labels = [f"{r['rnn']}\nu={r['units']}\nlr={r['lr']:.0e}" for r in grid_results[:6]]
mses   = [r['test_mse'] for r in grid_results[:6]]
colors = ['steelblue' if r['rnn']=='SimpleRNN' else 'coral' if r['rnn']=='LSTM' else 'green'
          for r in grid_results[:6]]
ax.bar(range(len(labels)), mses, color=colors)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
ax.set_title('Hyperparameter Grid Search Results')
ax.set_ylabel('Test MSE'); ax.grid(True, axis='y')

# 3c. LSTM gate concept
ax = axes[1, 0]
ax.axis('off')
gate_text = """
LSTM Gates (at each timestep t):

Forget gate:   f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
Input gate:    i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
Candidate:     g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)
Output gate:   o_t = σ(W_o·[h_{t-1}, x_t] + b_o)

Cell state:    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden state:  h_t = o_t ⊙ tanh(c_t)

GRU (simplified):
Reset gate:    r_t = σ(W_r·[h_{t-1}, x_t])
Update gate:   z_t = σ(W_z·[h_{t-1}, x_t])
New memory:    n_t = tanh(W_n·[r_t⊙h_{t-1}, x_t])
Output:        h_t = (1-z_t)⊙h_{t-1} + z_t⊙n_t
"""
ax.text(0.02, 0.98, gate_text, transform=ax.transAxes, va='top', fontsize=9,
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_title('LSTM & GRU Gate Equations')

# 3d. Learning rate effect
ax = axes[1, 1]
if tf_ok:
    for lr_val in lr_list:
        mse_per_rnn = {r['rnn']: r['test_mse'] for r in grid_results
                       if r['lr']==lr_val and r['units']==64}
        ax.plot(list(mse_per_rnn.keys()), list(mse_per_rnn.values()), 'o-',
                linewidth=2, markersize=10, label=f'lr={lr_val:.0e}')
    ax.set_title('RNN Type vs Test MSE by Learning Rate')
    ax.set_ylabel('Test MSE'); ax.legend(); ax.grid(True)
else:
    ax.axis('off')

plt.suptitle('Experiment 18: RNN, LSTM & GRU Hyperparameter Experimentation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp18.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp18.png")
