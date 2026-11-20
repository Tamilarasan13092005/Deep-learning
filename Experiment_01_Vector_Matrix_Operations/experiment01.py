"""
Experiment 01: Vector and Matrix Operations for Deep Learning
CO1 | Packages: NumPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment 01: Vector and Matrix Operations for Deep Learning")
print("=" * 60)

# ── 1. Vector Operations ──────────────────────────────────────
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("\n[1] Vector Operations")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a + b          = {a + b}")
print(f"  a - b          = {a - b}")
print(f"  Scalar * a (3) = {3 * a}")
print(f"  Dot product    = {np.dot(a, b)}")
print(f"  ||a||          = {np.linalg.norm(a):.4f}")

# ── 2. Matrix Operations ──────────────────────────────────────
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("\n[2] Matrix Operations")
print(f"  A =\n{A}")
print(f"  B =\n{B}")
print(f"  A + B =\n{A + B}")
print(f"  A @ B (matmul) =\n{A @ B}")
print(f"  A.T (transpose) =\n{A.T}")
print(f"  det(A) = {np.linalg.det(A):.4f}")

# ── 3. Inverse & Solve ────────────────────────────────────────
print("\n[3] Matrix Inverse and Linear System")
A_inv = np.linalg.inv(A)
print(f"  A_inv =\n{A_inv}")
print(f"  A @ A_inv ≈ I:\n{np.round(A @ A_inv, 6)}")

c = np.array([5, 11])
x = np.linalg.solve(A, c)
print(f"  Solve Ax=c where c={c}: x={x}")

# ── 4. Broadcasting ───────────────────────────────────────────
M = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
print("\n[4] Broadcasting")
print(f"  M =\n{M}")
print(f"  M + row =\n{M + row}")

# ── 5. Norms ──────────────────────────────────────────────────
v = np.array([3.0, 4.0])
print("\n[5] Norms")
print(f"  v = {v}")
print(f"  L1 norm = {np.linalg.norm(v, 1):.4f}")
print(f"  L2 norm = {np.linalg.norm(v, 2):.4f}")
print(f"  Linf    = {np.linalg.norm(v, np.inf):.4f}")

# ── 6. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 6a. 2D Vectors
ax = axes[0]
origin = [0, 0]
ax.quiver(*origin, 3, 1, angles='xy', scale_units='xy', scale=1, color='blue', label='v1=(3,1)')
ax.quiver(*origin, 1, 3, angles='xy', scale_units='xy', scale=1, color='red', label='v2=(1,3)')
ax.quiver(*origin, 4, 4, angles='xy', scale_units='xy', scale=1, color='green', label='v1+v2=(4,4)')
ax.set_xlim(-1, 6); ax.set_ylim(-1, 6)
ax.set_title('Vector Addition'); ax.legend(); ax.grid(True)

# 6b. Matrix heatmap
ax = axes[1]
mat = np.random.randint(1, 10, (4, 4))
im = ax.imshow(mat, cmap='Blues')
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=14)
ax.set_title('Matrix Heatmap'); plt.colorbar(im, ax=ax)

# 6c. Norm comparison
ax = axes[2]
norms = ['L1', 'L2', 'Linf']
vals = [np.linalg.norm(v, 1), np.linalg.norm(v, 2), np.linalg.norm(v, np.inf)]
ax.bar(norms, vals, color=['steelblue', 'coral', 'gold'])
ax.set_title('Norm Comparison'); ax.set_ylabel('Norm Value')
for i, val in enumerate(vals):
    ax.text(i, val + 0.05, f'{val:.2f}', ha='center')

plt.suptitle('Experiment 01: Vector & Matrix Operations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_exp01.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n[✓] Plot saved as output_exp01.png")
