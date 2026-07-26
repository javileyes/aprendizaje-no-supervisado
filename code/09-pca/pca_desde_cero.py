"""
PCA desde cero por diagonalización de la covarianza, con tres comprobaciones.

  1. Las direcciones principales son autovectores de S: verificamos ||S u - lambda u||.
  2. El error de reconstrucción con k componentes es EXACTAMENTE la suma de los
     autovalores descartados. Esa igualdad es la doble lectura del método:
     máxima varianza retenida = mínimo error de reconstrucción.
  3. Reducir antes de agrupar mejora el clústering: los mismos datos, k-means
     antes y después de proyectar.

Los datos son 400 puntos en 60 dimensiones cuya estructura real vive en un
subespacio de dimensión 3, ahogada en ruido isótropo.

Ejecútalo con:  python code/09-pca/pca_desde_cero.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)
n, d, d_latente = 400, 200, 3

# --- Datos: tres grupos que viven en un subespacio de dimensión 3 ---
centros = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [3.5, 6.0, 0.0]])
Z = np.vstack([c + rng.normal(scale=1.0, size=(n // 3 + (i == 0), d_latente))
               for i, c in enumerate(centros)])[:n]
y = np.repeat([0, 1, 2], [n // 3 + 1, n // 3, n - 2 * (n // 3) - 1])
base = np.linalg.qr(rng.normal(size=(d, d_latente)))[0]     # 3 direcciones ortonormales
X = Z @ base.T + rng.normal(scale=1.8, size=(n, d))         # + ruido en las 200
print(f"Datos: {n} puntos en {d} dimensiones; la señal vive en {d_latente}.")


def pca(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    S = (Xc.T @ Xc) / (len(X) - 1)          # covarianza muestral, d x d
    val, vec = np.linalg.eigh(S)            # eigh: S es simétrica
    orden = val.argsort()[::-1]             # de mayor a menor varianza
    return val[orden], vec[:, orden], mu, S


val, vec, mu, S = pca(X)
Xc = X - mu

# ---------- 1) ¿Son de verdad autovectores? ----------
residuos = [np.linalg.norm(S @ vec[:, j] - val[j] * vec[:, j]) for j in range(5)]
print(f"\n1) ||S u_j - lambda_j u_j|| para las 5 primeras: "
      f"máximo {max(residuos):.2e}")
print(f"   ortonormalidad: ||U^T U - I|| = "
      f"{np.linalg.norm(vec.T @ vec - np.eye(d)):.2e}")
print(f"   varianza de la proyección sobre u_1: {np.var(Xc @ vec[:, 0], ddof=1):.4f} "
      f"| lambda_1 = {val[0]:.4f}")

# ---------- 2) Error de reconstrucción = suma de autovalores descartados ----------
print(f"\n2) Error de reconstrucción frente a la suma de autovalores descartados")
print(f"{'k':>4} {'error medido':>15} {'suma lambdas':>15} {'dif':>10} {'var. explicada':>15}")
errores, explicada = [], []
for k in [1, 2, 3, 5, 10, 50, 100, 200]:
    U = vec[:, :k]
    rec = (Xc @ U) @ U.T                                 # proyectar y volver
    err = ((Xc - rec) ** 2).sum() / (len(X) - 1)         # normalizado como la varianza
    teo = val[k:].sum()
    errores.append(err)
    explicada.append(val[:k].sum() / val.sum())
    print(f"{k:4d} {err:15.6f} {teo:15.6f} {abs(err - teo):10.2e} {explicada[-1]:14.1%}")

# ---------- 3) Agrupar antes y después de proyectar ----------
def kmeans(X, k, semilla=0, iters=200):
    r = np.random.default_rng(semilla)
    C = [X[r.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        C.append(X[r.choice(len(X), p=d2 / max(d2.sum(), 1e-12))])
    C = np.array(C)
    for _ in range(iters):
        z = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        Cn = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                       for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def acuerdo(a, b):
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


print("\n3) k-means sobre los datos crudos y sobre las primeras componentes")
print(f"   en las {d} dimensiones originales : "
      f"{np.mean([acuerdo(kmeans(X, 3, s), y) for s in range(10)]):.1%} "
      f"(media de 10 semillas)")
for k in [2, 3, 5, 10]:
    P = Xc @ vec[:, :k]
    a = np.mean([acuerdo(kmeans(P, 3, s), y) for s in range(10)])
    print(f"   sobre las {k:2d} primeras componentes : {a:.1%}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(val[:20], "o-", color="#5b8def")
ax[0].set_xlabel("componente")
ax[0].set_ylabel("autovalor (varianza)")
ax[0].set_title("espectro: 3 grandes y una meseta")
ax[1].plot(np.cumsum(val) / val.sum(), color="#43c59e")
ax[1].axhline(0.9, ls=":", c="k")
ax[1].set_xlabel("nº de componentes")
ax[1].set_title("varianza explicada acumulada")
P2 = Xc @ vec[:, :2]
ax[2].scatter(P2[:, 0], P2[:, 1], c=y, cmap="viridis", s=12, edgecolors="none")
ax[2].set_title("proyección sobre las 2 primeras")
ax[2].set_xticks([])
ax[2].set_yticks([])
plt.tight_layout()
plt.show()
