"""
Experiment 10: L2, Dropout and Early Stopping Comparison
CO2 | Packages: TensorFlow, Keras, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 10: L2, Dropout and Early Stopping Comparison")
print("=" * 60)

np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    tf.random.set_seed(42)

    # ── 1. Dataset ────────────────────────────────────────────
    print("\n[1] Generating Dataset")
    n = 1000
    X = np.random.randn(n, 20).astype(np.float32)
    true_w = np.random.randn(20, 1).astype(np.float32)
    y = (X @ true_w + 0.5*np.random.randn(n,1)).astype(np.float32)
    y_cls = (y > y.mean()).astype(np.float32)

    split = int(0.6*n); vsplit = int(0.8*n)
    X_tr, X_val, X_te = X[:split], X[split:vsplit], X[vsplit:]
    y_tr, y_val, y_te = y_cls[:split], y_cls[split:vsplit], y_cls[vsplit:]

    def build_model(name, l2_lam=0.0, dropout_rate=0.0):
        model = keras.Sequential(name=name)
        model.add(layers.Input(shape=(20,)))
        for units in [128, 64, 32]:
            model.add(layers.Dense(units, activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_lam) if l2_lam else None))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    histories = {}
    configs = {
        'No Regularization': dict(l2_lam=0.0, dropout_rate=0.0),
        'L2 (λ=0.01)':       dict(l2_lam=0.01, dropout_rate=0.0),
        'Dropout (0.4)':     dict(l2_lam=0.0,  dropout_rate=0.4),
        'L2 + Dropout':      dict(l2_lam=0.001, dropout_rate=0.3),
    }

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                 restore_best_weights=True, verbose=0)

    print("\n[2] Training Models")
    results = {}
    for name, cfg in configs.items():
        m = build_model(name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', ''),
                        **cfg)
        early = name == 'No Regularization'
        callbacks = [es_callback] if early else []
        h = m.fit(X_tr, y_tr, epochs=200, batch_size=32,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks, verbose=0)
        histories[name] = h.history
        loss, acc = m.evaluate(X_te, y_te, verbose=0)
        results[name] = {'test_acc': acc, 'test_loss': loss,
                         'epochs_trained': len(h.history['loss'])}
        print(f"  {name:28s}: test_acc={acc*100:.1f}%, epochs={results[name]['epochs_trained']}")

    # ── 3. Early Stopping Standalone ──────────────────────────
    print("\n[3] Early Stopping Model")
    m_es = build_model('EarlyStopping_model')
    es2 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                         restore_best_weights=True, verbose=1)
    h_es = m_es.fit(X_tr, y_tr, epochs=500, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=[es2], verbose=0)
    loss_es, acc_es = m_es.evaluate(X_te, y_te, verbose=0)
    print(f"  Stopped at epoch: {len(h_es.history['loss'])}")
    print(f"  Test Accuracy: {acc_es*100:.1f}%")

    tf_ok = True

except ImportError:
    print("  TensorFlow not installed — demo with NumPy L2.")
    tf_ok = False

    # Simple L2 demo with numpy linear regression
    n = 300
    X = np.random.randn(n, 5)
    y = X @ np.array([1.5, -0.5, 2.0, 0.3, -1.0]) + 0.5*np.random.randn(n)
    split = int(0.7*n)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    def ridge(X, y, lam):
        I = np.eye(X.shape[1])
        return np.linalg.solve(X.T@X + lam*I, X.T@y)

    lambdas = [0, 0.01, 0.1, 1, 10, 100]
    train_errs, test_errs = [], []
    for lam in lambdas:
        w = ridge(X_tr, y_tr, lam)
        train_errs.append(np.mean((X_tr@w - y_tr)**2))
        test_errs.append(np.mean((X_te@w - y_te)**2))
    histories = None
    results = dict(zip([str(l) for l in lambdas],
                       [{'test_acc': 1-e/max(test_errs)} for e in test_errs]))

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if tf_ok:
    colors = ['red', 'blue', 'green', 'purple']

    # 4a. Training loss
    ax = axes[0, 0]
    for (name, h), color in zip(histories.items(), colors):
        ax.plot(h['loss'], color=color, linewidth=1.5, label=name, alpha=0.8)
    ax.set_title('Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True)

    # 4b. Validation loss
    ax = axes[0, 1]
    for (name, h), color in zip(histories.items(), colors):
        ax.plot(h['val_loss'], color=color, linewidth=1.5, label=name, alpha=0.8)
    ax.set_title('Validation Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True)

    # 4c. Train vs Val accuracy comparison
    ax = axes[1, 0]
    names = list(configs.keys())
    x_pos = np.arange(len(names))
    train_accs = [max(histories[n]['accuracy']) for n in names]
    val_accs   = [max(histories[n]['val_accuracy']) for n in names]
    ax.bar(x_pos - 0.2, train_accs, 0.4, label='Train Acc', color='steelblue')
    ax.bar(x_pos + 0.2, val_accs, 0.4, label='Val Acc', color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=15, fontsize=8)
    ax.set_title('Train vs Val Accuracy'); ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(True, axis='y')

    # 4d. Early stopping visualization
    ax = axes[1, 1]
    ax.plot(h_es.history['loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(h_es.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.axvline(len(h_es.history['loss'])-1, color='green',
               linestyle='--', label=f"Stop @ {len(h_es.history['loss'])}")
    ax.set_title('Early Stopping'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss'); ax.legend(); ax.grid(True)

else:
    ax = axes[0, 0]
    ax.plot(lambdas, train_errs, 'b-o', label='Train MSE')
    ax.plot(lambdas, test_errs, 'r-o', label='Test MSE')
    ax.set_xscale('symlog'); ax.set_title('L2 Regularization')
    ax.set_xlabel('Lambda'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)
    for axi in [axes[0,1], axes[1,0], axes[1,1]]:
        axi.axis('off')

plt.suptitle('Experiment 10: L2, Dropout & Early Stopping Comparison',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp10.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp10.png")
