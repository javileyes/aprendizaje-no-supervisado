"""
Qué enlace elegir: el simple encadena, y eso es a la vez su virtud y su ruina.

Dos escenarios sobre los mismos algoritmos:
  A) Dos lunas entrelazadas -> el enlace simple, que sigue caminos de puntos
     próximos, las separa perfectamente; completo y Ward fracasan.
  B) Los mismos dos grupos, pero con un PUENTE de ruido entre ellos -> el
     enlace simple cruza el puente y se derrumba; completo y Ward aguantan.

Dibujamos además el dendrograma a mano, sin scipy, para ver de dónde sale.

Ejecútalo con:  python code/05-clustering-jerarquico/enlaces_y_dendrograma.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(12)

COEFS = {
    "simple":   lambda a, b, c: (0.5, 0.5, 0.0, -0.5),
    "completo": lambda a, b, c: (0.5, 0.5, 0.0, +0.5),
    "medio":    lambda a, b, c: (a / (a + b), b / (a + b), 0.0, 0.0),
    "ward":     lambda a, b, c: ((a + c) / (a + b + c), (b + c) / (a + b + c),
                                 -c / (a + b + c), 0.0),
}
CUADRATICOS = {"ward"}


def aglomerativo(X, enlace):
    n = len(X)
    sq = (X ** 2).sum(axis=1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    if enlace not in CUADRATICOS:
        D = np.sqrt(D)
    np.fill_diagonal(D, np.inf)
    tam = {i: 1 for i in range(n)}
    activos, Z, idx, siguiente = list(range(n)), [], {i: i for i in range(n)}, n
    for _ in range(n - 1):
        sub = np.array(activos)
        M = D[np.ix_(sub, sub)]
        p, q = np.unravel_index(np.argmin(M), M.shape)
        a, b = sub[p], sub[q]
        h = D[a, b]
        if enlace in CUADRATICOS:
            h = np.sqrt(max(h, 0.0))
        na, nb = tam[a], tam[b]
        Z.append((idx[a], idx[b], h, na + nb))
        for c in activos:
            if c in (a, b):
                continue
            al, be, bt, ga = COEFS[enlace](na, nb, tam[c])
            D[a, c] = D[c, a] = (al * D[a, c] + be * D[b, c] + bt * D[a, b]
                                 + ga * abs(D[a, c] - D[b, c]))
        tam[a] = na + nb
        idx[a] = siguiente
        siguiente += 1
        activos.remove(b)
        D[b, :] = D[:, b] = np.inf
    return np.array(Z, dtype=float)


def cortar(Z, n, k):
    padre = {}
    for t, (i, j, _, _) in enumerate(Z[:n - k]):
        padre[int(i)] = n + t
        padre[int(j)] = n + t
    def raiz(x):
        while x in padre:
            x = padre[x]
        return x
    raices = [raiz(i) for i in range(n)]
    unicas = {r: c for c, r in enumerate(sorted(set(raices)))}
    return np.array([unicas[r] for r in raices])


def acuerdo(a, b):
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


def cofenetica(Z, X):
    """Correlación entre la distancia original y la altura de fusión."""
    n = len(X)
    cof = np.zeros((n, n))
    miembros = {i: [i] for i in range(n)}
    for t, (i, j, h, _) in enumerate(Z):
        a, b = miembros[int(i)], miembros[int(j)]
        for u in a:
            for v in b:
                cof[u, v] = cof[v, u] = h
        miembros[n + t] = a + b
    sq = (X ** 2).sum(axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    t = np.triu_indices(n, k=1)
    return np.corrcoef(D[t], cof[t])[0, 1]


def dibuja_dendrograma(ax, Z, n, corte=None):
    """Dendrograma desde cero: posición x de cada nodo = media de sus hijos."""
    pos, orden = {}, []

    def hojas(nodo):
        if nodo < n:
            orden.append(int(nodo))
            return
        i, j = int(Z[int(nodo) - n, 0]), int(Z[int(nodo) - n, 1])
        hojas(i)
        hojas(j)

    hojas(2 * n - 2)
    for x, hoja in enumerate(orden):
        pos[hoja] = (x, 0.0)
    for t, (i, j, h, _) in enumerate(Z):
        xi, yi = pos[int(i)]
        xj, yj = pos[int(j)]
        ax.plot([xi, xi, xj, xj], [yi, h, h, yj], c="#5b8def", lw=0.9)
        pos[n + t] = ((xi + xj) / 2, h)
    if corte is not None:
        ax.axhline(corte, color="#ef476f", ls="--", lw=1.4)
    ax.set_xticks([])


# ---------------- Escenario A: dos lunas ----------------
t = rng.uniform(0, np.pi, 90)
XA = np.vstack([np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(scale=0.07, size=(90, 2)),
                np.column_stack([1 - np.cos(t), 0.4 - np.sin(t)]) + rng.normal(scale=0.07, size=(90, 2))])
yA = np.repeat([0, 1], 90)

# ------- Escenario B: dos grupos claros unidos por un puente de 16 puntos -------
n_puente = 16                       # <-- cambia este número y verás el efecto
B0 = rng.normal(scale=0.55, size=(80, 2))
B1 = rng.normal(scale=0.55, size=(80, 2)) + [6.0, 0.0]
puente = np.column_stack([np.linspace(1.2, 4.8, n_puente), np.zeros(n_puente)]) \
    + rng.normal(scale=0.06, size=(n_puente, 2))
XB = np.vstack([B0, B1, puente])
yB = np.concatenate([np.zeros(80, int), np.ones(80, int)])   # el puente no se evalúa

print(f"{'escenario':<28} {'simple':>9} {'completo':>10} {'medio':>8} {'ward':>8}")
fila = [acuerdo(cortar(aglomerativo(XA, e), len(XA), 2), yA)
        for e in ["simple", "completo", "medio", "ward"]]
print(f"{'A · dos lunas limpias':<28} {fila[0]:8.1%} {fila[1]:9.1%} {fila[2]:7.1%} {fila[3]:7.1%}")
fila, tam = [], []
for e in ["simple", "completo", "medio", "ward"]:
    z = cortar(aglomerativo(XB, e), len(XB), 2)
    fila.append(acuerdo(z[:160], yB))          # solo los 160 puntos de los grupos
    tam.append(np.bincount(z))
print(f"{'B · dos grupos con puente':<28} {fila[0]:8.1%} {fila[1]:9.1%} {fila[2]:7.1%} {fila[3]:7.1%}")
print(f"   tamaños de los 2 grupos: " +
      "  ".join(f"{e}={t.tolist()}" for e, t in zip(["simple", "completo", "medio", "ward"], tam)))

print("\nCorrelación cofenética (cuánto respeta el dendrograma las distancias reales):")
for enlace in ["simple", "completo", "medio", "ward"]:
    Z = aglomerativo(XA, enlace)
    print(f"  {enlace:<10} {cofenetica(Z, XA):.4f}")

fig, ax = plt.subplots(2, 3, figsize=(14, 7))
casos = [(XA, yA, np.arange(len(XA)), "A · lunas"),
         (XB, yB, np.arange(160), "B · con puente")]
for f, (X, y, ev, titulo) in enumerate(casos):
    Zs, Zw = aglomerativo(X, "simple"), aglomerativo(X, "ward")
    zs, zw = cortar(Zs, len(X), 2), cortar(Zw, len(X), 2)
    ax[f, 0].scatter(X[:, 0], X[:, 1], c=zs, cmap="coolwarm", s=12, edgecolors="none")
    ax[f, 0].set_title(f"{titulo} · simple: {acuerdo(zs[ev], y):.0%}")
    ax[f, 1].scatter(X[:, 0], X[:, 1], c=zw, cmap="coolwarm", s=12, edgecolors="none")
    ax[f, 1].set_title(f"{titulo} · ward: {acuerdo(zw[ev], y):.0%}")
    for c in (0, 1):
        ax[f, c].set_xticks([])
        ax[f, c].set_yticks([])
    dibuja_dendrograma(ax[f, 2], Zs, len(X), corte=Zs[-1, 2] * 0.6)
    ax[f, 2].set_title(f"{titulo} · dendrograma (simple)")
plt.tight_layout()
plt.show()
