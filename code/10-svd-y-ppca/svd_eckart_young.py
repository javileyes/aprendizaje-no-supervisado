"""
La descomposición en valores singulares y el teorema de Eckart-Young.

  1. Toda matriz admite X = U S V^T. Lo comprobamos reconstruyendo.
  2. Su relación con PCA: las direcciones principales son las columnas de V y
     los autovalores de la covarianza son sigma^2/(n-1). Sin formar nunca X^T X.
  3. Eckart-Young: truncar la SVD da la MEJOR aproximación de rango k en norma
     de Frobenius, y el error al cuadrado es la suma de los sigma^2 descartados.
     Lo contrastamos con 2000 aproximaciones de rango k construidas al azar.
  4. Compresión: una imagen sintética de 160x160 con rango bajo.

Ejecútalo con:  python code/10-svd-y-ppca/svd_eckart_young.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(6)

# ---------------- 1) La descomposición existe y reconstruye ----------------
n, d = 200, 40
X = rng.normal(size=(n, 5)) @ rng.normal(size=(5, d)) + 0.4 * rng.normal(size=(n, d))
Xc = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print(f"X centrada: {Xc.shape}.  U: {U.shape}, s: {s.shape}, V^T: {Vt.shape}")
print(f"1) ||X - U S V^T||_F = {np.linalg.norm(Xc - (U * s) @ Vt):.2e}")
print(f"   ||U^T U - I|| = {np.linalg.norm(U.T @ U - np.eye(len(s))):.2e}, "
      f"||V^T V - I|| = {np.linalg.norm(Vt @ Vt.T - np.eye(len(s))):.2e}")
print(f"   valores singulares (5 primeros): {np.round(s[:5], 4)}")

# ---------------- 2) SVD y PCA son lo mismo ----------------
S_cov = (Xc.T @ Xc) / (n - 1)
val, vec = np.linalg.eigh(S_cov)
val = val[::-1]
print(f"\n2) autovalores de la covarianza (5 primeros): {np.round(val[:5], 6)}")
print(f"   sigma^2/(n-1)               (5 primeros): {np.round(s[:5] ** 2 / (n - 1), 6)}")
print(f"   diferencia máxima: {np.abs(val[:5] - s[:5] ** 2 / (n - 1)).max():.2e}")
print(f"   ¿coinciden las direcciones? |v_1 . u_1| = "
      f"{abs(Vt[0] @ vec[:, -1]):.10f}  (1 = misma recta)")

# ---------------- 3) Eckart-Young contra el azar ----------------
print(f"\n3) Error de la mejor aproximación de rango k")
print(f"{'k':>4} {'||X-X_k||_F^2':>16} {'suma sigma^2':>15} {'dif':>10} "
      f"{'mejor de 2000 al azar':>22}")
for k in [1, 2, 3, 5, 10]:
    Xk = (U[:, :k] * s[:k]) @ Vt[:k]
    err = ((Xc - Xk) ** 2).sum()
    teo = (s[k:] ** 2).sum()
    mejor_azar = np.inf
    for _ in range(2000):
        B = rng.normal(size=(d, k))
        Q, _ = np.linalg.qr(B)                     # subespacio aleatorio de dim k
        R = (Xc @ Q) @ Q.T
        mejor_azar = min(mejor_azar, ((Xc - R) ** 2).sum())
    print(f"{k:4d} {err:16.6f} {teo:15.6f} {abs(err - teo):10.2e} {mejor_azar:21.6f}")
print("   Ningún subespacio aleatorio consigue bajar del valor de la SVD.")

# ---------------- 4) Compresión de una imagen sintética ----------------
t = np.linspace(-3, 3, 160)
gx, gy = np.meshgrid(t, t)
r = np.sqrt(gx ** 2 + gy ** 2)
img = (np.sin(3 * r)                                   # ondas radiales
       + 0.5 * np.sign(np.sin(4 * gx + 2 * gy))        # bordes en diagonal
       + 0.35 * np.cos(6 * np.arctan2(gy, gx)) * np.exp(-r / 3))   # textura angular
Ui, si, Vti = np.linalg.svd(img, full_matrices=False)
print(f"\n4) Imagen {img.shape[0]}x{img.shape[1]} = {img.size} números")
print(f"{'rango k':>8} {'números guardados':>18} {'compresión':>12} {'energía retenida':>18}")
for k in [1, 2, 5, 10, 20, 40]:
    guardados = k * (img.shape[0] + img.shape[1] + 1)
    energia = (si[:k] ** 2).sum() / (si ** 2).sum()
    print(f"{k:8d} {guardados:18d} {img.size / guardados:11.1f}x {energia:17.2%}")

fig, ax = plt.subplots(2, 3, figsize=(13, 8))
ax[0, 0].semilogy(s, "o-", ms=3, color="#5b8def")
ax[0, 0].set_title("valores singulares de X")
ax[0, 0].set_xlabel("índice")
ax[0, 1].semilogy(si, color="#43c59e")
ax[0, 1].set_title("valores singulares de la imagen")
ax[0, 1].set_xlabel("índice")
ax[0, 2].plot(np.cumsum(si ** 2) / (si ** 2).sum(), color="#e2b53d")
ax[0, 2].set_title("energía acumulada")
ax[0, 2].set_xlabel("rango k")
for a, k in zip(ax[1], [2, 10, 40]):
    rec = (Ui[:, :k] * si[:k]) @ Vti[:k]
    a.imshow(rec, cmap="magma")
    a.set_title(f"rango {k} · {(si[:k]**2).sum()/(si**2).sum():.1%} de energía")
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
