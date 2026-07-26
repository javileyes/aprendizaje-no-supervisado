"""
Clústering espectral completo, en veinte líneas de álgebra lineal.

  1. grafo de similitud (núcleo gaussiano o k vecinos más próximos);
  2. laplaciano normalizado L_sym = I - D^{-1/2} W D^{-1/2};
  3. los k autovectores de menor autovalor -> incrustación en R^k;
  4. normalizar las filas a norma 1 (Ng, Jordan y Weiss, 2001);
  5. k-means sobre esa incrustación.

Lo probamos sobre tres formas donde k-means no tiene ninguna posibilidad:
dos lunas, dos círculos concéntricos y tres anillos.

Ejecútalo con:  python code/07-clustering-espectral/espectral_desde_cero.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(8)


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


def grafo_knn(X, k_vecinos):
    """Grafo de k vecinos, simetrizado con OR: w_ij = 1 si i es vecino de j o al revés."""
    sq = (X ** 2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, axis=1)[:, :k_vecinos]
    W = np.zeros_like(D2)
    filas = np.repeat(np.arange(len(X)), k_vecinos)
    W[filas, idx.ravel()] = 1.0
    return np.maximum(W, W.T)


def espectral(X, k, k_vecinos=10, semilla=0):
    W = grafo_knn(X, k_vecinos)
    d = W.sum(axis=1)
    d_inv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L_sym = np.eye(len(X)) - (d_inv[:, None] * W) * d_inv[None, :]
    val, vec = np.linalg.eigh(L_sym)
    U = vec[:, :k]                                   # los k autovectores más bajos
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    return kmeans(U, k, semilla), val


# ---------------- Los tres conjuntos de prueba ----------------
t = rng.uniform(0, np.pi, 150)
lunas = np.vstack([np.column_stack([np.cos(t), np.sin(t)]),
                   np.column_stack([1 - np.cos(t), 0.4 - np.sin(t)])]) \
    + rng.normal(scale=0.07, size=(300, 2))
y_lunas = np.repeat([0, 1], 150)

ang = rng.uniform(0, 2 * np.pi, 150)
circulos = np.vstack([np.column_stack([np.cos(ang), np.sin(ang)]),
                      2.4 * np.column_stack([np.cos(ang), np.sin(ang)])]) \
    + rng.normal(scale=0.09, size=(300, 2))
y_circ = np.repeat([0, 1], 150)

ang3 = rng.uniform(0, 2 * np.pi, 110)
anillos = np.vstack([r * np.column_stack([np.cos(ang3), np.sin(ang3)])
                     for r in (1.0, 2.2, 3.4)]) + rng.normal(scale=0.08, size=(330, 2))
y_anillos = np.repeat([0, 1, 2], 110)

casos = [("dos lunas", lunas, y_lunas, 2),
         ("dos círculos", circulos, y_circ, 2),
         ("tres anillos", anillos, y_anillos, 3)]

# --- El grafo NO es un detalle: barremos el número de vecinos ---
vecinos_probados = [5, 6, 8, 10, 14, 20]
print("Acuerdo del clústering espectral según cuántos vecinos tenga el grafo:")
print(f"{'k vecinos':>10} " + " ".join(f"{n:>14}" for n, _, _, _ in casos))
tabla = {}
for kv in vecinos_probados:
    fila = []
    for nombre, X, y, k in casos:
        z, _ = espectral(X, k, k_vecinos=kv)
        fila.append(acuerdo(z, y))
        tabla[(nombre, kv)] = fila[-1]
    print(f"{kv:10d} " + " ".join(f"{a:13.1%} " for a in fila))

print("\nNingún valor sirve para los tres a la vez: con 10 vecinos las lunas salen")
print("perfectas pero los anillos se degradan, y con 5 pasa justo al revés.\n")

print(f"{'conjunto':<15} {'k-means':>9} {'espectral (mejor kv)':>22}   autovalores más bajos")
resultados = []
for nombre, X, y, k in casos:
    z_km = kmeans(X, k, semilla=0)
    kv_mejor = max(vecinos_probados, key=lambda kv: tabla[(nombre, kv)])
    z_es, val = espectral(X, k, k_vecinos=kv_mejor)
    resultados.append((nombre, X, y, z_km, z_es, val, k, kv_mejor))
    print(f"{nombre:<15} {acuerdo(z_km, y):8.1%} {acuerdo(z_es, y):15.1%} (kv={kv_mejor:2d})   "
          f"{np.round(val[:k + 2], 4)}")

print("\nSalto espectral: con un grafo de k vecinos bien elegido, los grupos quedan")
print("DESCONECTADOS, así que hay tantos autovalores nulos como grupos.")
for nombre, X, y, _, _, val, k, kv in resultados:
    nulos = int((val < 1e-9).sum())
    print(f"  {nombre:<15} autovalores por debajo de 1e-9: {nulos}  (grupos reales: {k})")

fig, ax = plt.subplots(3, 3, figsize=(13, 11))
for f, (nombre, X, y, z_km, z_es, val, k, kv) in enumerate(resultados):
    ax[f, 0].scatter(X[:, 0], X[:, 1], c=z_km, cmap="coolwarm", s=10, edgecolors="none")
    ax[f, 0].set_title(f"{nombre} · k-means {acuerdo(z_km, y):.0%}")
    ax[f, 1].scatter(X[:, 0], X[:, 1], c=z_es, cmap="coolwarm", s=10, edgecolors="none")
    ax[f, 1].set_title(f"{nombre} · espectral {acuerdo(z_es, y):.0%} (kv={kv})")
    for c in (0, 1):
        ax[f, c].set_xticks([])
        ax[f, c].set_yticks([])
        ax[f, c].set_aspect("equal")
    ax[f, 2].plot(val[:10], "o-", color="#43c59e")
    ax[f, 2].axvline(k - 0.5, ls="--", c="#ef476f")
    ax[f, 2].set_xlabel("índice del autovalor")
    ax[f, 2].set_title("espectro de L_sym")
plt.tight_layout()
plt.show()
