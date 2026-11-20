"""
Experiment 11: CNN for Image Classification
CO2 | Packages: TensorFlow, Keras, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 11: CNN for Image Classification")
print("=" * 60)

np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import cifar10

    tf.random.set_seed(42)
    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    # ── 1. Load CIFAR-10 ──────────────────────────────────────
    print("\n[1] Loading CIFAR-10")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── 2. Build CNN ──────────────────────────────────────────
    print("\n[2] Building CNN")
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name='CNN_CIFAR10')
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # ── 3. Train ──────────────────────────────────────────────
    print("\n[3] Training CNN")
    history = model.fit(X_train, y_train, epochs=25,
                        batch_size=128, validation_split=0.1,
                        verbose=1)

    # ── 4. Evaluate ───────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[4] Test Accuracy: {test_acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    class_acc = [(y_pred[y_test.flatten()==c]==c).mean() for c in range(10)]

    # ── 5. Feature Map Visualization ──────────────────────────
    feature_model = keras.Model(inputs=model.input,
                                outputs=model.layers[0].output)
    sample_img = X_test[0:1]
    feature_maps = feature_model.predict(sample_img, verbose=0)

    tf_ok = True

except ImportError:
    print("  TensorFlow not installed — using simple convolution demo.")
    tf_ok = False

    # Manual 2D convolution demo
    def convolve2d(image, kernel):
        kh, kw = kernel.shape
        ih, iw = image.shape
        out_h = ih - kh + 1
        out_w = iw - kw + 1
        out = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                out[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
        return out

    img = np.random.randn(28, 28)
    kernels = {
        'Edge Detect': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
        'Sharpen':     np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        'Blur':        np.ones((3,3))/9,
    }

# ── 6. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if tf_ok:
    ax = axes[0, 0]
    ax.plot(history.history['accuracy'], 'b-', label='Train', linewidth=2)
    ax.plot(history.history['val_accuracy'], 'r-', label='Val', linewidth=2)
    ax.set_title('CNN Accuracy'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_title('CNN Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.bar(class_names, class_acc, color='steelblue')
    ax.set_title('Per-Class Test Accuracy'); ax.set_xticklabels(class_names, rotation=45)
    ax.set_ylabel('Accuracy'); ax.grid(True, axis='y')

    ax = axes[1, 1]
    n_maps = min(16, feature_maps.shape[-1])
    for i in range(n_maps):
        sub = axes[1,1]
    ax.axis('off')
    ax.set_title('Sample Feature Maps')
    fig2, axes2 = plt.subplots(4, 4, figsize=(8, 8))
    for i, axx in enumerate(axes2.flat):
        if i < feature_maps.shape[-1] and i < 16:
            axx.imshow(feature_maps[0, :, :, i], cmap='viridis')
        axx.axis('off')
    fig2.suptitle('Conv Layer 1 Feature Maps', fontsize=12)
    fig2.tight_layout()
    fig2.savefig('output_exp11_features.png', dpi=100, bbox_inches='tight')
    plt.close(fig2)

else:
    img_gray = np.random.rand(64, 64)
    for i, (kname, kern) in enumerate(kernels.items()):
        ax = axes[i // 2, i % 2] if i < 4 else None
        if ax:
            filtered = convolve2d(img_gray[:30, :30], kern)
            ax.imshow(filtered, cmap='gray')
            ax.set_title(f'Filter: {kname}')
    for axi in axes.flat:
        axi.axis('off')

plt.suptitle('Experiment 11: CNN for Image Classification',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp11.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plots saved.")
