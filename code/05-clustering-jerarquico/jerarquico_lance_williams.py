"""
Clústering jerárquico aglomerativo con los cuatro enlaces clásicos, todos
implementados con la MISMA recurrencia de Lance-Williams.

Comprobamos tres cosas:
  1. que la fórmula de Ward, Delta = nA*nB/(nA+nB) * ||muA - muB||^2, coincide
     con el aumento REAL de la suma de cuadrados intra-grupo al fusionar;
  2. que Lance-Williams reproduce ese mismo Delta sin recalcular medias;
  3. que el enlace por centroide puede producir INVERSIONES (una fusión más
     baja que la anterior), y que los otros cuatro no.

Ejecútalo con:  python code/05-clustering-jerarquico/jerarquico_lance_williams.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)

# --- Coeficientes de Lance-Williams: d(A∪B, C) en función de d(A,C), d(B,C), d(A,B) ---
# Devuelven (alpha_A, alpha_B, beta, gamma) a partir de los tamaños nA, nB, nC.
COEFS = {
    "simple":   lambda a, b, c: (0.5, 0.5, 0.0, -0.5),
    "completo": lambda a, b, c: (0.5, 0.5, 0.0, +0.5),
    "medio":    lambda a, b, c: (a / (a + b), b / (a + b), 0.0, 0.0),
    "centroide": lambda a, b, c: (a / (a + b), b / (a + b), -a * b / (a + b) ** 2, 0.0),
    "ward":     lambda a, b, c: ((a + c) / (a + b + c), (b + c) / (a + b + c),
                                 -c / (a + b + c), 0.0),
}
# "centroide" y "ward" trabajan sobre distancias AL CUADRADO; los otros, sobre
# distancias sin elevar. Es la única diferencia de tratamiento.
CUADRATICOS = {"centroide", "ward"}


def aglomerativo(X, enlace):
    """Devuelve la matriz de enlace en formato (i, j, altura, tamaño)."""
    n = len(X)
    sq = (X ** 2).sum(axis=1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)   # cuadrados
    if enlace not in CUADRATICOS:
        D = np.sqrt(D)
    np.fill_diagonal(D, np.inf)

    tam = {i: 1 for i in range(n)}
    activos = list(range(n))
    Z = []
    siguiente = n
    idx = {i: i for i in range(n)}          # posición en D -> identificador

    for _ in range(n - 1):
        sub = np.array(activos)
        M = D[np.ix_(sub, sub)]
        p, q = np.unravel_index(np.argmin(M), M.shape)
        a, b = sub[p], sub[q]
        altura = D[a, b]
        if enlace in CUADRATICOS:
            altura = np.sqrt(max(altura, 0.0))
        na, nb = tam[a], tam[b]
        Z.append((idx[a], idx[b], altura, na + nb))

        # --- actualización de Lance-Williams para el nuevo grupo (reusamos la fila a) ---
        for c in activos:
            if c in (a, b):
                continue
            al, be, bt, ga = COEFS[enlace](na, nb, tam[c])
            nueva = (al * D[a, c] + be * D[b, c] + bt * D[a, b]
                     + ga * abs(D[a, c] - D[b, c]))
            D[a, c] = D[c, a] = nueva
        tam[a] = na + nb
        idx[a] = siguiente
        siguiente += 1
        activos.remove(b)
        D[b, :] = D[:, b] = np.inf
    return np.array(Z, dtype=float)


def cortar(Z, n, k):
    """Etiquetas al cortar el dendrograma para quedarse con k grupos."""
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


# ---------------- 1) La fórmula de Ward, comprobada ----------------
A = rng.normal(size=(7, 3)) + 1.0
B = rng.normal(size=(11, 3)) - 1.5


def W(P):
    return ((P - P.mean(axis=0)) ** 2).sum()


real = W(np.vstack([A, B])) - W(A) - W(B)
formula = len(A) * len(B) / (len(A) + len(B)) * ((A.mean(0) - B.mean(0)) ** 2).sum()
print("1) Coste de Ward al fusionar dos grupos")
print(f"   aumento real de la suma de cuadrados : {real:.10f}")
print(f"   fórmula nA*nB/(nA+nB)*||muA-muB||^2  : {formula:.10f}")
print(f"   diferencia                           : {abs(real - formula):.2e}\n")

# ---------------- 2) Los cuatro enlaces sobre los mismos datos ----------------
centros = np.array([[0.0, 0.0], [4.0, 0.5], [2.0, 4.0]])
X = np.vstack([c + rng.normal(scale=0.7, size=(40, 2)) for c in centros])
y = np.repeat([0, 1, 2], 40)
n = len(X)


def acuerdo(a, b):
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


print("2) Acuerdo con la verdad al cortar en k=3, y monotonía del dendrograma")
print(f"   {'enlace':<10} {'acuerdo':>9} {'alturas monótonas':>19} {'inversiones':>12}")
Zs = {}
for enlace in ["simple", "completo", "medio", "centroide", "ward"]:
    Z = aglomerativo(X, enlace)
    Zs[enlace] = Z
    z = cortar(Z, n, 3)
    alturas = Z[:, 2]
    inversiones = int((np.diff(alturas) < -1e-12).sum())
    print(f"   {enlace:<10} {acuerdo(z, y):8.1%} {'sí' if inversiones == 0 else 'NO':>19} "
          f"{inversiones:12d}")

# ---------------- 3) Un caso donde el centroide SÍ se invierte ----------------
print("\n3) Tres puntos casi equiláteros: el caso clásico de inversión")
T = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2 * 1.02]])
for enlace in ["ward", "centroide"]:
    Z = aglomerativo(T, enlace)
    alturas = np.round(Z[:, 2], 4)
    inv = "INVERSIÓN" if alturas[1] < alturas[0] else "monótono"
    print(f"   {enlace:<10} alturas {alturas} -> {inv}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, enlace in zip(ax, ["simple", "completo", "ward"]):
    z = cortar(Zs[enlace], n, 3)
    a.scatter(X[:, 0], X[:, 1], c=z, cmap="viridis", s=16, edgecolors="none")
    a.set_title(f"{enlace}: {acuerdo(z, y):.0%}")
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
