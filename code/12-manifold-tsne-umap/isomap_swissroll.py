"""
El rollo suizo: por qué lo lineal no basta y cómo lo arregla Isomap.

Un rollo suizo es una lámina 2D enrollada en 3D. Dos puntos pueden estar muy
cerca en línea recta y muy lejos SOBRE la lámina (en capas distintas del rollo).
PCA y el MDS clásico solo saben de líneas rectas y fracasan. Isomap sustituye
la distancia euclídea por la GEODÉSICA -el camino más corto por el grafo de
vecinos- y con eso desenrolla la lámina.

Medimos el éxito con la correlación de Spearman entre la coordenada recuperada
y el parámetro real a lo largo del rollo.

Ejecútalo con:  python code/12-manifold-tsne-umap/isomap_swissroll.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 600

# --- El rollo suizo: t es la posición A LO LARGO de la lámina ---
t = 1.5 * np.pi * (1 + 2 * rng.random(n))
altura = 12 * rng.random(n)
X = np.column_stack([t * np.cos(t), altura, t * np.sin(t)])
X += 0.05 * rng.normal(size=X.shape)
print(f"Rollo suizo: {n} puntos en 3D, lámina intrínseca de dimensión 2.")


def dist_euclidea(X):
    sq = (X ** 2).sum(axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))


def mds_clasico(D, k=2):
    """MDS clásico: B = -1/2 J D^2 J y se diagonaliza."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    val, vec = np.linalg.eigh(B)
    val, vec = val[::-1][:k], vec[:, ::-1][:, :k]
    return vec * np.sqrt(np.maximum(val, 0))


def geodesicas(X, k_vecinos):
    """Distancias por el camino más corto en el grafo de k vecinos (Floyd-Warshall
    sería O(n^3); usamos Dijkstra vectorizado sobre matriz densa)."""
    D = dist_euclidea(X)
    idx = np.argsort(D, axis=1)[:, 1:k_vecinos + 1]
    G = np.full_like(D, np.inf)
    np.fill_diagonal(G, 0.0)
    filas = np.repeat(np.arange(len(X)), k_vecinos)
    G[filas, idx.ravel()] = D[filas, idx.ravel()]
    G = np.minimum(G, G.T)                       # grafo no dirigido
    # Dijkstra desde cada nodo (n veces), con matriz densa
    Dg = np.empty_like(G)
    for s in range(len(X)):
        dist = np.full(len(X), np.inf)
        dist[s] = 0.0
        visto = np.zeros(len(X), bool)
        for _ in range(len(X)):
            u = int(np.argmin(np.where(visto, np.inf, dist)))
            if not np.isfinite(dist[u]):
                break
            visto[u] = True
            nuevo = dist[u] + G[u]
            dist = np.minimum(dist, nuevo)
        Dg[s] = dist
    finito = np.isfinite(Dg)
    Dg[~finito] = Dg[finito].max() * 2           # componentes desconectadas
    return (Dg + Dg.T) / 2


def spearman(a, b):
    """Correlación de Spearman: correlación de los rangos."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return abs(np.corrcoef(ra, rb)[0, 1])


# --- 1) PCA (lineal) ---
Xc = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
Y_pca = Xc @ Vt[:2].T

# --- 2) MDS clásico con distancias euclídeas (equivale a PCA) ---
D_euc = dist_euclidea(X)
Y_mds = mds_clasico(D_euc, 2)
print(f"\nMDS clásico con distancias euclídeas = PCA: "
      f"máxima diferencia entre coordenadas (salvo signo) "
      f"{min(np.abs(np.abs(Y_mds) - np.abs(Y_pca)).max(), 9.99):.4f}")

# --- 3) Isomap: distancias geodésicas + MDS ---
print("\nCalculando geodésicas (esto tarda unos segundos)...")
resultados = []
for kv in [5, 8, 12, 20]:
    Dg = geodesicas(X, kv)
    Y_iso = mds_clasico(Dg, 2)
    resultados.append((kv, Y_iso, spearman(Y_iso[:, 0], t), Dg))

print(f"\n{'método':<26} {'|Spearman| con el parámetro t':>32}")
print(f"{'PCA (1ª componente)':<26} {spearman(Y_pca[:, 0], t):32.4f}")
print(f"{'MDS clásico (1ª coord.)':<26} {spearman(Y_mds[:, 0], t):32.4f}")
for kv, _, sp, _ in resultados:
    print(f"{'Isomap, ' + str(kv) + ' vecinos':<26} {sp:32.4f}")

mejor_kv, Y_iso, mejor_sp, Dg = max(resultados, key=lambda r: r[2])
print(f"\nMejor Isomap: {mejor_kv} vecinos, |Spearman| = {mejor_sp:.4f}")
print("Un valor cercano a 1 significa que la coordenada recuperada ordena los")
print("puntos igual que el recorrido REAL a lo largo de la lámina.")

# ¿Cuánto se equivoca la distancia euclídea? Comparamos con la geodésica.
tri = np.triu_indices(n, k=1)
razon = Dg[tri] / np.maximum(D_euc[tri], 1e-9)
print(f"\nCociente geodésica/euclídea: mediana {np.median(razon):.2f}, "
      f"percentil 99 {np.percentile(razon, 99):.2f}, máximo {razon.max():.2f}")
print(f"Hay pares de puntos que en línea recta están {razon.max():.0f} veces más cerca")
print("de lo que están sobre la lámina: eso es lo que engaña a PCA.")

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].scatter(X[:, 0], X[:, 2], c=t, cmap="Spectral", s=8)
ax[0].set_title("el rollo, visto de canto")
ax[1].scatter(Y_pca[:, 0], Y_pca[:, 1], c=t, cmap="Spectral", s=8)
ax[1].set_title(f"PCA · Spearman {spearman(Y_pca[:, 0], t):.2f}")
ax[2].scatter(Y_mds[:, 0], Y_mds[:, 1], c=t, cmap="Spectral", s=8)
ax[2].set_title(f"MDS clásico · {spearman(Y_mds[:, 0], t):.2f}")
ax[3].scatter(Y_iso[:, 0], Y_iso[:, 1], c=t, cmap="Spectral", s=8)
ax[3].set_title(f"Isomap ({mejor_kv} vec.) · {mejor_sp:.2f}")
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
