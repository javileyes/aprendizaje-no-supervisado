"""
Los tres índices internos clásicos, implementados desde cero, y su punto ciego.

  - Silueta:            s(i) = (b(i) - a(i)) / max(a(i), b(i))
  - Calinski-Harabasz:  (B/(k-1)) / (W/(n-k))    [de la identidad ANOVA del cap. 3]
  - Davies-Bouldin:     media del peor cociente (dispersión / separación)

Los aplicamos a dos conjuntos: uno donde aciertan (tres grupos esféricos) y otro
donde fallan sistemáticamente (dos lunas). El punto del capítulo es que un índice
interno no mide "calidad": mide PARECIDO A SU PROPIA IDEA DE GRUPO.

Ejecútalo con:  python code/08-validacion-clusteres/indices_internos.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


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
    z = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return z, C


def matriz_dist(X):
    sq = (X ** 2).sum(axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))


def silueta(X, z):
    """a(i): distancia media a su grupo. b(i): la menor media a otro grupo."""
    D = matriz_dist(X)
    k = z.max() + 1
    s = np.zeros(len(X))
    for i in range(len(X)):
        propio = z == z[i]
        n_propio = propio.sum()
        if n_propio <= 1:
            s[i] = 0.0
            continue
        a = D[i, propio].sum() / (n_propio - 1)          # se excluye él mismo
        b = min(D[i, z == j].mean() for j in range(k) if j != z[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


def calinski_harabasz(X, z):
    n, k = len(X), z.max() + 1
    mu = X.mean(axis=0)
    W = sum(((X[z == j] - X[z == j].mean(axis=0)) ** 2).sum() for j in range(k))
    B = sum((z == j).sum() * ((X[z == j].mean(axis=0) - mu) ** 2).sum() for j in range(k))
    return (B / (k - 1)) / (W / (n - k))


def davies_bouldin(X, z):
    k = z.max() + 1
    C = np.array([X[z == j].mean(axis=0) for j in range(k)])
    S = np.array([np.sqrt(((X[z == j] - C[j]) ** 2).sum(axis=1).mean()) for j in range(k)])
    total = 0.0
    for i in range(k):
        peor = max((S[i] + S[j]) / np.linalg.norm(C[i] - C[j])
                   for j in range(k) if j != i)
        total += peor
    return total / k


def inercia(X, z, C):
    return ((X - C[z]) ** 2).sum()


# ---------------- Conjunto A: tres grupos esféricos ----------------
centros = np.array([[0.0, 0.0], [5.0, 0.5], [2.5, 5.0]])
XA = np.vstack([c + rng.normal(scale=0.8, size=(100, 2)) for c in centros])

# ---------------- Conjunto B: dos lunas ----------------
t = rng.uniform(0, np.pi, 150)
XB = np.vstack([np.column_stack([np.cos(t), np.sin(t)]),
                np.column_stack([1 - np.cos(t), 0.4 - np.sin(t)])]) \
    + rng.normal(scale=0.07, size=(300, 2))
yB = np.repeat([0, 1], 150)

for nombre, X, verdad in [("A · tres grupos esféricos", XA, 3),
                          ("B · dos lunas", XB, 2)]:
    print(f"\n{nombre}  (k verdadero = {verdad})")
    print(f"{'k':>3} {'inercia':>11} {'silueta':>9} {'Calinski-H':>12} {'Davies-B':>10}")
    sil, ch, db = [], [], []
    for k in range(2, 8):
        z, C = kmeans(X, k, semilla=0)
        s = silueta(X, z)
        c = calinski_harabasz(X, z)
        d = davies_bouldin(X, z)
        sil.append(s)
        ch.append(c)
        db.append(d)
        print(f"{k:3d} {inercia(X, z, C):11.2f} {s:9.4f} {c:12.2f} {d:10.4f}")
    ks = list(range(2, 8))
    print(f"    -> silueta elige k={ks[int(np.argmax(sil))]}, "
          f"Calinski-H elige k={ks[int(np.argmax(ch))]}, "
          f"Davies-B elige k={ks[int(np.argmin(db))]}")

# --- El punto ciego: la partición CORRECTA de las lunas puntúa PEOR ---
z_km, _ = kmeans(XB, 2, semilla=0)
print("\nEn las lunas, comparemos la partición de k-means con la verdadera:")
print(f"  silueta de k-means (k=2)      : {silueta(XB, z_km):.4f}")
print(f"  silueta de la partición REAL  : {silueta(XB, yB):.4f}")
print(f"  Calinski-H de k-means         : {calinski_harabasz(XB, z_km):.2f}")
print(f"  Calinski-H de la partición REAL: {calinski_harabasz(XB, yB):.2f}")
print("  Los índices prefieren la respuesta EQUIVOCADA: premian la compacidad,")
print("  y las lunas correctas no son compactas.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for nombre, X, col in [("A", XA, "#5b8def"), ("B", XB, "#ef476f")]:
    valores = [silueta(X, kmeans(X, k, semilla=0)[0]) for k in range(2, 8)]
    ax[0].plot(range(2, 8), valores, "o-", color=col, label=nombre)
ax[0].set_xlabel("k")
ax[0].set_ylabel("silueta media")
ax[0].legend()
ax[0].set_title("silueta frente a k")
ax[1].scatter(XB[:, 0], XB[:, 1], c=z_km, cmap="coolwarm", s=10, edgecolors="none")
ax[1].set_title(f"k-means: silueta {silueta(XB, z_km):.3f}")
ax[2].scatter(XB[:, 0], XB[:, 1], c=yB, cmap="coolwarm", s=10, edgecolors="none")
ax[2].set_title(f"verdad: silueta {silueta(XB, yB):.3f}")
for a in ax[1:]:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
