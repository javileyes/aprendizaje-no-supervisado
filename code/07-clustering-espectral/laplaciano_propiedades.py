"""
El laplaciano del grafo, comprobado propiedad por propiedad.

  1. Identidad clave: f^T L f = (1/2) * suma_ij w_ij (f_i - f_j)^2.
     De ella salen todas las demás: L es semidefinida positiva y su forma
     cuadrática mide "cuánto varía f a lo largo de las aristas".
  2. La multiplicidad del autovalor 0 es EXACTAMENTE el número de componentes
     conexas, y su núcleo está GENERADO por los indicadores de esas componentes
     (generado, no formado: la base concreta que devuelve eigh es arbitraria).
  3. El segundo autovector (vector de Fiedler) separa el grafo por su punto
     más débil, y su corte es mejor que el de miles de particiones al azar.

Ejecútalo con:  python code/07-clustering-espectral/laplaciano_propiedades.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def grafo_gaussiano(X, sigma):
    sq = (X ** 2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    W = np.exp(-D2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return W


# ---------------- 1) La identidad de la forma cuadrática ----------------
W = rng.random((60, 60))
W = (W + W.T) / 2
np.fill_diagonal(W, 0.0)
L = np.diag(W.sum(axis=1)) - W
f = rng.normal(size=60)

forma = f @ L @ f
suma = 0.5 * (W * (f[:, None] - f[None, :]) ** 2).sum()
print("1) Identidad f^T L f = (1/2) sum_ij w_ij (f_i - f_j)^2")
print(f"   f^T L f             = {forma:.10f}")
print(f"   (1/2)sum w(fi-fj)^2 = {suma:.10f}")
print(f"   diferencia          = {abs(forma - suma):.2e}")
print(f"   autovalor mínimo de L = {np.linalg.eigvalsh(L).min():.2e}  (debe ser ~0, "
      f"nunca negativo)\n")

# ---------------- 2) Multiplicidad del 0 = componentes conexas ----------------
print("2) Multiplicidad del autovalor 0 frente al número de componentes")
print(f"   {'componentes':>12} {'autovalores casi nulos':>24} {'salto espectral':>16}")
for n_comp in [1, 2, 3, 4]:
    centros = np.array([[6.0 * i, 0.0] for i in range(n_comp)])
    X = np.vstack([c + rng.normal(scale=0.35, size=(30, 2)) for c in centros])
    Wg = grafo_gaussiano(X, sigma=0.5)
    Wg[Wg < 1e-8] = 0.0                       # cortamos: grafo realmente desconectado
    Lg = np.diag(Wg.sum(axis=1)) - Wg
    val = np.linalg.eigvalsh(Lg)
    casi_cero = int((val < 1e-8).sum())
    print(f"   {n_comp:12d} {casi_cero:24d} {val[n_comp]:16.4f}")
print("   El autovalor que sigue al último cero (lambda_k+1) es el 'salto espectral'.\n")

# ---------------- 3) El vector de Fiedler contra 20 000 particiones al azar ----
print("3) ¿Corta bien el vector de Fiedler? (RatioCut: más bajo, mejor)")
X = np.vstack([rng.normal(scale=0.6, size=(45, 2)),
               rng.normal(scale=0.6, size=(45, 2)) + [4.0, 0.0]])
W = grafo_gaussiano(X, sigma=0.9)
L = np.diag(W.sum(axis=1)) - W
val, vec = np.linalg.eigh(L)
fiedler = vec[:, 1]                            # el segundo autovector
z_esp = (fiedler > 0).astype(int)


def ratio_cut(W, z):
    corte = W[z == 0][:, z == 1].sum()
    n0, n1 = (z == 0).sum(), (z == 1).sum()
    if n0 == 0 or n1 == 0:
        return np.inf
    return corte / n0 + corte / n1


rc_esp = ratio_cut(W, z_esp)
azar = [ratio_cut(W, rng.integers(0, 2, len(X))) for _ in range(20000)]
azar = np.array(azar)
print(f"   RatioCut del vector de Fiedler : {rc_esp:.5f}")
print(f"   RatioCut al azar (20 000 veces): mediana {np.median(azar):.5f}, "
      f"mínimo {azar.min():.5f}")
print(f"   veces que el azar lo mejora    : {(azar < rc_esp).sum()} de 20000")
print(f"   autovalores 1 a 4: {np.round(val[:4], 5)}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=fiedler, cmap="coolwarm", s=22, edgecolors="none")
ax[0].set_title("valor del vector de Fiedler")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].plot(np.sort(fiedler), "o", ms=3, color="#5b8def")
ax[1].axhline(0, ls="--", c="#ef476f")
ax[1].set_title("Fiedler ordenado: dos mesetas")
ax[2].hist(azar, bins=60, color="#8b93a7")
ax[2].axvline(rc_esp, color="#ef476f", lw=2)
ax[2].set_xlabel("RatioCut")
ax[2].set_title("20 000 particiones al azar vs. Fiedler")
plt.tight_layout()
plt.show()
