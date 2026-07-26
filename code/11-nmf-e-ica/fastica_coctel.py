"""
ICA: separar señales mezcladas (el "problema del cóctel") con FastICA.

Tres fuentes independientes y NO gaussianas -una senoide, una sierra y una
onda cuadrada con ruido- se mezclan linealmente en tres micrófonos. A partir
SOLO de las mezclas, ICA recupera las fuentes.

La clave: por el teorema central del límite, una mezcla de variables
independientes es MÁS gaussiana que sus componentes. Así que buscar las
direcciones MENOS gaussianas es buscar las fuentes. PCA no puede: solo
decorrelaciona, y decorrelación no es independencia.

Ejecútalo con:  python code/11-nmf-e-ica/fastica_coctel.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
n = 2000
t = np.linspace(0, 8, n)

# --- Tres fuentes independientes y no gaussianas ---
s1 = np.sin(3 * t)                                  # senoide
s2 = 2 * (t / 1.3 % 1.0) - 1.0                      # diente de sierra
s3 = np.sign(np.sin(5.5 * t)) + 0.15 * rng.normal(size=n)   # cuadrada con ruido
S = np.column_stack([s1, s2, s3])
S -= S.mean(axis=0)
S /= S.std(axis=0)

A = np.array([[1.0, 0.9, 0.6],                      # matriz de mezcla
              [0.7, 1.0, 0.8],
              [0.5, 0.6, 1.0]])
X = S @ A.T
print(f"{n} muestras, 3 fuentes, 3 mezclas.")


def curtosis(y):
    """Curtosis en exceso: 0 para una gaussiana, distinto de 0 si no lo es."""
    y = (y - y.mean()) / y.std()
    return (y ** 4).mean() - 3.0


print("\nCurtosis en exceso (0 = gaussiana):")
print(f"  fuentes : {np.round([curtosis(S[:, j]) for j in range(3)], 3)}")
print(f"  mezclas : {np.round([curtosis(X[:, j]) for j in range(3)], 3)}")
print("  Las mezclas están MÁS cerca de 0: mezclar gaussianiza. ICA deshace eso.")

# ---------------- Paso 1: centrar y blanquear ----------------
Xc = X - X.mean(axis=0)
C = np.cov(Xc, rowvar=False)
val, vec = np.linalg.eigh(C)
Blanqueo = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T
Z = Xc @ Blanqueo.T
print(f"\nTras blanquear, ||cov(Z) - I|| = "
      f"{np.linalg.norm(np.cov(Z, rowvar=False) - np.eye(3)):.2e}")
print("Con los datos blanqueados, la matriz que falta por encontrar es ORTOGONAL:")
print("el problema pasa de d^2 incógnitas a una rotación.")


def fastica(Z, k, iters=200, tol=1e-10, semilla=0):
    """FastICA con no linealidad g = tanh, por deflación (una fuente cada vez)."""
    r = np.random.default_rng(semilla)
    n, d = Z.shape
    Wm = np.zeros((k, d))
    for j in range(k):
        w = r.normal(size=d)
        w /= np.linalg.norm(w)
        for _ in range(iters):
            wx = Z @ w
            g = np.tanh(wx)                          # g = G'
            g_prima = 1.0 - g ** 2                   # g' = G''
            # punto fijo de Newton: w+ = E[x g(w.x)] - E[g'(w.x)] w
            w_nuevo = (Z * g[:, None]).mean(axis=0) - g_prima.mean() * w
            # ortogonalizar respecto a las direcciones ya encontradas
            w_nuevo -= Wm[:j].T @ (Wm[:j] @ w_nuevo)
            w_nuevo /= np.linalg.norm(w_nuevo)
            if abs(abs(w_nuevo @ w) - 1) < tol:
                w = w_nuevo
                break
            w = w_nuevo
        Wm[j] = w
    return Wm


Wm = fastica(Z, 3)
S_est = Z @ Wm.T
print(f"\n||W W^T - I|| = {np.linalg.norm(Wm @ Wm.T - np.eye(3)):.2e} (es ortogonal)")


def mejor_correlacion(S_est, S):
    """ICA no fija el orden ni el signo: emparejamos por |correlación|."""
    k = S.shape[1]
    C = np.abs(np.corrcoef(S_est.T, S.T)[:k, k:])
    usados, pares = set(), []
    for i in np.argsort(-C.max(axis=1)):
        j = int(np.argmax([C[i, x] if x not in usados else -1 for x in range(k)]))
        usados.add(j)
        pares.append((i, j, C[i, j]))
    return sorted(pares, key=lambda p: p[1])


print("\n|correlación| entre cada componente recuperada y su fuente:")
for i, j, c in mejor_correlacion(S_est, S):
    print(f"  ICA {i} <-> fuente {j}: {c:.4f}")
print(f"  media: {np.mean([c for _, _, c in mejor_correlacion(S_est, S)]):.4f}")

# --- PCA, para comparar: decorrelaciona pero no separa ---
Up, sp, Vtp = np.linalg.svd(Xc, full_matrices=False)
S_pca = Xc @ Vtp.T
print("\nLo mismo con PCA:")
for i, j, c in mejor_correlacion(S_pca, S):
    print(f"  PCA {i} <-> fuente {j}: {c:.4f}")
print(f"  media: {np.mean([c for _, _, c in mejor_correlacion(S_pca, S)]):.4f}")
print("\nCurtosis de lo recuperado por ICA: "
      f"{np.round([curtosis(S_est[:, j]) for j in range(3)], 3)}")

fig, ax = plt.subplots(4, 3, figsize=(13, 9))
for j in range(3):
    ax[0, j].plot(t[:500], S[:500, j], lw=0.8, color="#5b8def")
    ax[0, j].set_title(f"fuente {j}", fontsize=10)
    ax[1, j].plot(t[:500], X[:500, j], lw=0.8, color="#8b93a7")
    ax[1, j].set_title(f"mezcla {j}", fontsize=10)
for i, j, _ in mejor_correlacion(S_est, S):
    ax[2, j].plot(t[:500], S_est[:500, i], lw=0.8, color="#43c59e")
    ax[2, j].set_title(f"ICA -> fuente {j}", fontsize=10)
for i, j, _ in mejor_correlacion(S_pca, S):
    ax[3, j].plot(t[:500], S_pca[:500, i], lw=0.8, color="#ef476f")
    ax[3, j].set_title(f"PCA -> fuente {j}", fontsize=10)
for fila in ax:
    for a in fila:
        a.set_xticks([])
        a.set_yticks([])
plt.tight_layout()
plt.show()
