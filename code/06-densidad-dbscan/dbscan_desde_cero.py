"""
DBSCAN desde cero: clústeres de forma arbitraria y ruido declarado como tal.

Implementamos el algoritmo tal cual lo definieron Ester, Kriegel, Sander y Xu
(1996): puntos núcleo, alcanzabilidad por densidad y expansión por anchura.
Lo probamos sobre dos lunas con un 12 % de ruido uniforme -donde k-means daba
un 70,1 % en el capítulo 3- y luego barremos epsilon para ver la sensibilidad.

Ejecútalo con:  python code/06-densidad-dbscan/dbscan_desde_cero.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)

# --- Datos: dos lunas + ruido uniforme por encima ---
t = rng.uniform(0, np.pi, 150)
luna_a = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(scale=0.08, size=(150, 2))
luna_b = np.column_stack([1 - np.cos(t), 0.4 - np.sin(t)]) + rng.normal(scale=0.08, size=(150, 2))
ruido = rng.uniform([-1.6, -1.2], [2.6, 1.6], size=(40, 2))
X = np.vstack([luna_a, luna_b, ruido])
y = np.concatenate([np.zeros(150, int), np.ones(150, int), np.full(40, -1)])
n = len(X)
print(f"{n} puntos: 150 + 150 de las lunas y 40 de ruido ({40/n:.0%}).")

# Matriz de distancias (n es pequeño; con n grande se usaría un índice espacial)
sq = (X ** 2).sum(axis=1)
D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))


def dbscan(D, eps, min_pts):
    """Devuelve etiquetas; -1 significa ruido."""
    n = len(D)
    vecinos = [np.flatnonzero(D[i] <= eps) for i in range(n)]      # incluye a i
    nucleo = np.array([len(v) >= min_pts for v in vecinos])
    etiqueta = np.full(n, -1)
    c = 0
    for i in range(n):
        if etiqueta[i] != -1 or not nucleo[i]:
            continue
        # expansión en anchura desde el punto núcleo i
        etiqueta[i] = c
        cola = list(vecinos[i])
        while cola:
            j = cola.pop()
            if etiqueta[j] == -1:
                etiqueta[j] = c                     # punto frontera o núcleo nuevo
                if nucleo[j]:
                    cola.extend(vecinos[j])         # solo los núcleo propagan
        c += 1
    return etiqueta, nucleo


def acuerdo(a, b):
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


# --- El gráfico de k-distancias: la heurística para elegir epsilon ---
MIN_PTS = 8
k_dist = np.sort(np.sort(D, axis=1)[:, MIN_PTS - 1])
print(f"\nDistancia al {MIN_PTS}º vecino: mediana {np.median(k_dist):.3f}, "
      f"percentil 90 {np.percentile(k_dist, 90):.3f}")

# --- Barrido de epsilon ---
print(f"\n{'eps':>6} {'grupos':>7} {'ruido':>7} {'acuerdo lunas':>15} {'ruido acertado':>16}")
mejor, respaldo = None, None
for eps in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
    z, nucleo = dbscan(D, eps, MIN_PTS)
    n_grupos = len(set(z[z >= 0]))
    frac_ruido = (z == -1).mean()
    lunas = z[:300]
    acu = acuerdo(lunas, y[:300])
    ruido_ok = (z[300:] == -1).mean()
    print(f"{eps:6.2f} {n_grupos:7d} {frac_ruido:7.1%} {acu:14.1%} {ruido_ok:15.1%}")
    if n_grupos == 2 and (mejor is None or acu > mejor[1]):
        mejor = (eps, acu, z, nucleo)
    if respaldo is None or acu > respaldo[1]:
        respaldo = (eps, acu, z, nucleo)

if mejor is None:            # con otros parámetros puede no haber ningún eps con 2 grupos
    mejor = respaldo
    print("\n(ningún epsilon da exactamente 2 grupos; usamos el de mayor acuerdo)")
eps, acu, z, nucleo = mejor
print(f"\nEpsilon elegido: {eps:.2f}")
print(f"  acuerdo en las lunas       : {acu:.1%}")
print(f"  puntos de ruido detectados : {(z[300:] == -1).sum()}/40")
print(f"  puntos núcleo              : {nucleo.sum()}/{n}")
print(f"  puntos frontera            : {((z >= 0) & ~nucleo).sum()}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="none")
ax[0].set_title("verdad (gris = ruido)")
col = np.where(z < 0, -1, z)
ax[1].scatter(X[z < 0, 0], X[z < 0, 1], c="#aab", s=14, marker="x")
ax[1].scatter(X[z >= 0, 0], X[z >= 0, 1], c=col[z >= 0], cmap="coolwarm",
              s=12, edgecolors="none")
ax[1].set_title(f"DBSCAN eps={eps}: {acu:.0%}")
for a in ax[:2]:
    a.set_xticks([])
    a.set_yticks([])
ax[2].plot(k_dist, color="#5b8def")
ax[2].axhline(eps, ls="--", c="#ef476f")
ax[2].set_xlabel("puntos ordenados")
ax[2].set_ylabel(f"distancia al {MIN_PTS}º vecino")
ax[2].set_title("gráfico de k-distancias")
plt.tight_layout()
plt.show()
