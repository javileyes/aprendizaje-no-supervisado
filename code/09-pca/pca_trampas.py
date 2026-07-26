"""
Las dos trampas de PCA, medidas.

TRAMPA 1 · Varianza no es información. PCA busca la dirección donde los datos
se estiran más, no la que separa los grupos. Si el ruido tiene más varianza que
la señal, la primera componente es ruido puro y quedarse con ella DESTRUYE la
estructura.

TRAMPA 2 · PCA no es invariante de escala. Cambiar las unidades de una columna
cambia las componentes. Por eso "PCA sobre la covarianza" y "PCA sobre la
correlación" son dos métodos distintos, no dos implementaciones del mismo.

Ejecútalo con:  python code/09-pca/pca_trampas.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(21)


def pca(X):
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / (len(X) - 1)
    val, vec = np.linalg.eigh(S)
    o = val.argsort()[::-1]
    return val[o], vec[:, o], Xc


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


# ================= TRAMPA 1: varianza no es información =================
n = 300
ruido = rng.normal(scale=6.0, size=n)                       # mucha varianza, sin grupos
senal = np.repeat([-1.5, 1.5], n // 2) + rng.normal(scale=0.35, size=n)
y = np.repeat([0, 1], n // 2)
X1 = np.column_stack([ruido, senal])

val, vec, Xc = pca(X1)
print("TRAMPA 1 · dos columnas: ruido (sd 6) y señal con dos grupos (sd 0,35)")
print(f"  varianza de cada componente: {np.round(val, 3)}")
print(f"  varianza explicada por la 1ª: {val[0] / val.sum():.1%}")
print(f"  la 1ª componente apunta a:   {np.round(np.abs(vec[:, 0]), 3)}  "
      f"(1,0 = ruido puro)")

for k, nombre in [(1, "solo la 1ª componente"), (2, "las dos componentes")]:
    P = Xc @ vec[:, :k]
    print(f"  k-means sobre {nombre:24s}: {acuerdo(kmeans(P, 2), y):.1%}")
P2 = (Xc @ vec[:, 1:2])
print(f"  k-means sobre la 2ª componente sola : {acuerdo(kmeans(P2, 2), y):.1%}")
print("  Quedarse con la componente de más varianza tira justo lo que importaba.\n")

# ================= TRAMPA 2: la escala cambia las componentes =================
# Dos variables correlacionadas medidas en unidades muy distintas
lat = rng.normal(size=n)
altura_cm = 170 + 8 * lat + rng.normal(scale=2.0, size=n)      # decenas
peso_kg = 70 + 9 * lat + rng.normal(scale=3.0, size=n)         # decenas
ruido_mm = rng.normal(scale=300.0, size=n)                     # ¡cientos!
X2 = np.column_stack([altura_cm, peso_kg, ruido_mm])
nombres = ["altura(cm)", "peso(kg)", "sensor(mm)"]

val_cov, vec_cov, _ = pca(X2)
Z = (X2 - X2.mean(axis=0)) / X2.std(axis=0, ddof=1)            # estandarizar
val_cor, vec_cor, _ = pca(Z)

print("TRAMPA 2 · las mismas tres variables, con y sin estandarizar")
print(f"  {'':22s} {nombres[0]:>12s} {nombres[1]:>12s} {nombres[2]:>12s}")
print(f"  {'desviación típica':22s} " +
      " ".join(f"{s:12.2f}" for s in X2.std(axis=0, ddof=1)))
print(f"  {'1ª comp. (covarianza)':22s} " +
      " ".join(f"{c:12.3f}" for c in vec_cov[:, 0]))
print(f"  {'1ª comp. (correlación)':22s} " +
      " ".join(f"{c:12.3f}" for c in vec_cor[:, 0]))
print(f"  varianza explicada por la 1ª: covarianza {val_cov[0]/val_cov.sum():.1%}, "
      f"correlación {val_cor[0]/val_cor.sum():.1%}")
print("  Sin estandarizar, la 1ª componente ES el sensor con números grandes.")
print("  Estandarizando, recoge la relación real entre altura y peso.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X1[:, 0], X1[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="none")
flecha = vec[:, 0] * 3 * np.sqrt(val[0])
ax[0].annotate("", xy=flecha, xytext=(-flecha[0], -flecha[1]),
               arrowprops=dict(arrowstyle="<->", lw=2, color="k"))
ax[0].set_title("trampa 1: la 1ª componente es el ruido")
ax[0].set_aspect("equal")
ax[1].hist([(Xc @ vec[:, 0])[y == 0], (Xc @ vec[:, 0])[y == 1]], bins=25, stacked=True)
ax[1].set_title("proyección sobre la 1ª: grupos mezclados")
ax[2].hist([(Xc @ vec[:, 1])[y == 0], (Xc @ vec[:, 1])[y == 1]], bins=25, stacked=True)
ax[2].set_title("proyección sobre la 2ª: grupos separados")
plt.tight_layout()
plt.show()
