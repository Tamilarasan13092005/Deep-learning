"""
Experiment 05: Bias-Variance, Overfitting and Underfitting Analysis
CO1 | Packages: Scikit-learn, NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error

print("=" * 60)
print("Experiment 05: Bias-Variance, Overfitting & Underfitting")
print("=" * 60)

np.random.seed(42)

# ── 1. Generate Dataset ───────────────────────────────────────
def true_fn(x):
    return np.sin(2 * np.pi * x)

n = 100
x_data = np.sort(np.random.uniform(0, 1, n))
y_data = true_fn(x_data) + np.random.normal(0, 0.3, n)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42)

x_plot = np.linspace(0, 1, 300)

# ── 2. Fit Polynomials of Different Degrees ───────────────────
degrees = [1, 4, 15]
train_errors, test_errors = [], []

print("\n[2] Polynomial Degree Comparison")
print(f"{'Degree':>8} {'Train MSE':>12} {'Test MSE':>12} {'Diagnosis':>15}")
print("-" * 50)
for deg in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(deg)),
        ('lin', LinearRegression())
    ])
    model.fit(x_train.reshape(-1,1), y_train)
    tr_err = mean_squared_error(y_train, model.predict(x_train.reshape(-1,1)))
    te_err = mean_squared_error(y_test, model.predict(x_test.reshape(-1,1)))
    train_errors.append(tr_err)
    test_errors.append(te_err)
    if deg <= 2:
        diag = "Underfitting"
    elif deg >= 12:
        diag = "Overfitting"
    else:
        diag = "Good Fit"
    print(f"{deg:>8} {tr_err:>12.4f} {te_err:>12.4f} {diag:>15}")

# ── 3. Bias-Variance Decomposition ───────────────────────────
print("\n[3] Bias-Variance Tradeoff across Degrees")
all_degrees = range(1, 16)
all_train, all_test = [], []
for deg in all_degrees:
    model = Pipeline([('poly', PolynomialFeatures(deg)),
                      ('lin', LinearRegression())])
    model.fit(x_train.reshape(-1,1), y_train)
    all_train.append(mean_squared_error(y_train, model.predict(x_train.reshape(-1,1))))
    all_test.append(mean_squared_error(y_test, model.predict(x_test.reshape(-1,1))))
print(f"  Optimal degree (min test MSE): {list(all_degrees)[np.argmin(all_test)]}")

# ── 4. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['red', 'green', 'blue']
labels = ['Degree 1 (Underfit)', 'Degree 4 (Good Fit)', 'Degree 15 (Overfit)']

# 4a. Fits
ax = axes[0, 0]
ax.scatter(x_data, y_data, alpha=0.3, s=20, color='gray', label='Data')
ax.plot(x_plot, true_fn(x_plot), 'k--', linewidth=2, label='True Function')
for deg, color, label in zip(degrees, colors, labels):
    m = Pipeline([('poly', PolynomialFeatures(deg)), ('lin', LinearRegression())])
    m.fit(x_train.reshape(-1,1), y_train)
    ax.plot(x_plot, m.predict(x_plot.reshape(-1,1)), color=color,
            linewidth=1.5, label=label)
ax.set_title('Model Fits: Under/Good/Over'); ax.legend(fontsize=7); ax.grid(True)

# 4b. Train/Test error vs degree
ax = axes[0, 1]
ax.plot(list(all_degrees), all_train, 'bo-', label='Train MSE', linewidth=2)
ax.plot(list(all_degrees), all_test, 'rs-', label='Test MSE', linewidth=2)
ax.axvline(list(all_degrees)[np.argmin(all_test)], color='green',
           linestyle='--', label='Optimal')
ax.set_title('Bias-Variance Tradeoff'); ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)
ax.set_ylim(0, min(1.0, max(all_test) * 1.2))

# 4c. Learning Curve (degree 4)
ax = axes[1, 0]
model_lc = Pipeline([('poly', PolynomialFeatures(4)), ('lin', LinearRegression())])
train_sizes, train_scores, val_scores = learning_curve(
    model_lc, x_data.reshape(-1,1), y_data,
    train_sizes=np.linspace(0.1, 1.0, 20),
    cv=5, scoring='neg_mean_squared_error')
ax.plot(train_sizes, -train_scores.mean(1), 'b-o', label='Train MSE', linewidth=2)
ax.plot(train_sizes, -val_scores.mean(1), 'r-o', label='Val MSE', linewidth=2)
ax.fill_between(train_sizes, -train_scores.mean(1)-train_scores.std(1),
                -train_scores.mean(1)+train_scores.std(1), alpha=0.1, color='blue')
ax.fill_between(train_sizes, -val_scores.mean(1)-val_scores.std(1),
                -val_scores.mean(1)+val_scores.std(1), alpha=0.1, color='red')
ax.set_title('Learning Curve (Degree 4)'); ax.set_xlabel('Training Size')
ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)

# 4d. Residuals for each model
ax = axes[1, 1]
for deg, color, label in zip(degrees, colors, labels):
    m = Pipeline([('poly', PolynomialFeatures(deg)), ('lin', LinearRegression())])
    m.fit(x_train.reshape(-1,1), y_train)
    res = y_test - m.predict(x_test.reshape(-1,1))
    ax.scatter(x_test, res, alpha=0.5, s=20, color=color, label=label)
ax.axhline(0, color='black', linewidth=1)
ax.set_title('Residuals on Test Set'); ax.set_xlabel('x'); ax.set_ylabel('Residual')
ax.legend(fontsize=7); ax.grid(True)

plt.suptitle('Experiment 05: Bias-Variance, Overfitting & Underfitting',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp05.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp05.png")
