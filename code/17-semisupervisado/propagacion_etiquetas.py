"""
Propagación de etiquetas en un grafo: la solución armónica, en forma cerrada.

Con L puntos etiquetados y U sin etiquetar, se construye el grafo de similitud
y se busca la función f más SUAVE sobre el grafo que respete las etiquetas:

    minimizar  (1/2) f^T Laplaciano f     sujeto a   f_L = y_L

La condición de optimalidad da un sistema lineal cuya solución es
    f_U = (D_UU - W_UU)^{-1} W_UL y_L
y se llama solución ARMÓNICA porque cada valor no etiquetado resulta ser la
media ponderada de sus vecinos. Reaparece aquí el laplaciano del capítulo 7.

Segunda parte: un caso donde la hipótesis del clúster es FALSA y los datos sin
etiquetar EMPEORAN el resultado. No es un fallo de implementación: es lo que
ocurre cuando el supuesto no se cumple.

Ejecútalo con:  python code/17-semisupervisado/propagacion_etiquetas.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)


def grafo_gaussiano(X, sigma, k_vecinos=12):
    """Similitud gaussiana, restringida a los k vecinos más próximos."""
    sq = (X ** 2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    W = np.exp(-D2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    idx = np.argsort(-W, axis=1)[:, k_vecinos:]
    filas = np.repeat(np.arange(len(X)), idx.shape[1])
    W[filas, idx.ravel()] = 0.0
    return np.maximum(W, W.T)


def solucion_armonica(W, etiquetado, y_l):
    """f_U = (D_UU - W_UU)^{-1} W_UL y_L, con y_L en formato one-hot."""
    d = W.sum(axis=1)
    L_idx = np.flatnonzero(etiquetado)
    U_idx = np.flatnonzero(~etiquetado)
    Luu = np.diag(d[U_idx]) - W[np.ix_(U_idx, U_idx)]
    Wul = W[np.ix_(U_idx, L_idx)]
    F = np.zeros((len(W), y_l.shape[1]))
    F[L_idx] = y_l
    F[U_idx] = np.linalg.solve(Luu + 1e-9 * np.eye(len(U_idx)), Wul @ y_l)
    return F


def knn_supervisado(Xtr, ytr, Xte, k=1):
    sq_tr = (Xtr ** 2).sum(axis=1)
    sq_te = (Xte ** 2).sum(axis=1)
    D = np.maximum(sq_te[:, None] + sq_tr[None, :] - 2 * Xte @ Xtr.T, 0.0)
    idx = np.argsort(D, axis=1)[:, :k]
    votos = ytr[idx]
    return np.array([np.bincount(v, minlength=ytr.max() + 1).argmax() for v in votos])


# ================= PARTE 1: dos lunas =================
n_por_luna = 200
t = rng.uniform(0, np.pi, n_por_luna)
X = np.vstack([np.column_stack([np.cos(t), np.sin(t)]),
               np.column_stack([1 - np.cos(t), 0.4 - np.sin(t)])]) \
    + rng.normal(scale=0.08, size=(2 * n_por_luna, 2))
y = np.repeat([0, 1], n_por_luna)
n = len(X)
W = grafo_gaussiano(X, sigma=0.22)

print("PARTE 1 · dos lunas (la hipótesis del clúster SE CUMPLE)")
print(f"{'etiquetas':>10} {'1-NN supervisado':>18} {'propagación':>13} {'mejora':>9} "
      f"{'rango de la propagación':>25}")
for n_etq in [2, 4, 10, 20, 50]:
    accs_sup, accs_prop = [], []
    for rep in range(20):                      # 20 sorteos de qué puntos se etiquetan
        r = np.random.default_rng(100 + rep)
        sel = np.concatenate([r.choice(np.flatnonzero(y == c), n_etq // 2, replace=False)
                              for c in (0, 1)])
        etiquetado = np.zeros(n, bool)
        etiquetado[sel] = True
        Y = np.eye(2)[y[sel]]
        F = solucion_armonica(W, etiquetado, Y)
        pred = F.argmax(axis=1)
        accs_prop.append((pred[~etiquetado] == y[~etiquetado]).mean())
        pred_sup = knn_supervisado(X[sel], y[sel], X[~etiquetado])
        accs_sup.append((pred_sup == y[~etiquetado]).mean())
    m_sup, m_prop = np.mean(accs_sup), np.mean(accs_prop)
    print(f"{n_etq:10d} {m_sup:17.1%} {m_prop:12.1%} {m_prop - m_sup:+8.1%} "
          f"{min(accs_prop):11.1%} a {max(accs_prop):8.1%}")

# --- Comprobación: la solución es ARMÓNICA (media ponderada de los vecinos) ---
r = np.random.default_rng(100)
sel_fig = np.concatenate([r.choice(np.flatnonzero(y == c), 2, replace=False) for c in (0, 1)])
etiquetado = np.zeros(n, bool)
etiquetado[sel_fig] = True
F = solucion_armonica(W, etiquetado, np.eye(2)[y[sel_fig]])
d = W.sum(axis=1)
media_vecinos = (W @ F) / d[:, None]
err = np.abs(F[~etiquetado] - media_vecinos[~etiquetado]).max()
print(f"\nPropiedad armónica: max |f(i) - media ponderada de sus vecinos| = {err:.2e}")
print("Cada valor no etiquetado ES exactamente la media de sus vecinos. De ahí")
print("el nombre: es la versión discreta de una función armónica.")

# ================= PARTE 2: cuando la hipótesis es FALSA =================
# Dos clases separadas por una frontera que atraviesa una zona DENSA.
n2 = 400
X2 = rng.normal(scale=1.0, size=(n2, 2))
y2 = (X2[:, 0] > 0).astype(int)          # la frontera pasa por el centro de la nube
W2 = grafo_gaussiano(X2, sigma=0.5)

print("\nPARTE 2 · una nube gaussiana partida por la mitad")
print("(la frontera atraviesa la ZONA MÁS DENSA: la hipótesis del clúster es falsa)")
print(f"{'etiquetas':>10} {'1-NN supervisado':>18} {'propagación':>13} {'mejora':>9} "
      f"{'rango de la propagación':>25}")
for n_etq in [4, 10, 20, 50]:
    accs_sup, accs_prop = [], []
    for rep in range(20):
        r = np.random.default_rng(200 + rep)
        sel = np.concatenate([r.choice(np.flatnonzero(y2 == c), n_etq // 2, replace=False)
                              for c in (0, 1)])
        etq = np.zeros(n2, bool)
        etq[sel] = True
        F2 = solucion_armonica(W2, etq, np.eye(2)[y2[sel]])
        accs_prop.append((F2.argmax(axis=1)[~etq] == y2[~etq]).mean())
        accs_sup.append((knn_supervisado(X2[sel], y2[sel], X2[~etq]) == y2[~etq]).mean())
    m_sup, m_prop = np.mean(accs_sup), np.mean(accs_prop)
    print(f"{n_etq:10d} {m_sup:17.1%} {m_prop:12.1%} {m_prop - m_sup:+8.1%} "
          f"{min(accs_prop):11.1%} a {max(accs_prop):8.1%}")
# ¿Qué está pasando? Comparamos la ESTABILIDAD entre repeticiones.
def dispersion(X, y, W, n_etq, semilla_base, reps=20):
    out = []
    for rep in range(reps):
        r = np.random.default_rng(semilla_base + rep)
        sel = np.concatenate([r.choice(np.flatnonzero(y == c), n_etq // 2, replace=False)
                              for c in (0, 1)])
        etq = np.zeros(len(X), bool)
        etq[sel] = True
        F = solucion_armonica(W, etq, np.eye(2)[y[sel]])
        out.append((F.argmax(axis=1)[~etq] == y[~etq]).mean())
    return np.array(out)


a1 = dispersion(X, y, W, 50, 100)
a2 = dispersion(X2, y2, W2, 50, 200)
print(f"\nCon 50 etiquetas y 20 sorteos distintos de CUÁLES etiquetar:")
print(f"  dos lunas          : media {a1.mean():.1%}, desviación {a1.std():.1%}, "
      f"rango [{a1.min():.1%}, {a1.max():.1%}]")
print(f"  nube partida       : media {a2.mean():.1%}, desviación {a2.std():.1%}, "
      f"rango [{a2.min():.1%}, {a2.max():.1%}]")
print("Ahí está el diagnóstico. En las lunas, el resultado no depende de qué")
print("puntos se etiqueten: siempre 100 %. En la nube partida oscila decenas de")
print("puntos porcentuales según el sorteo. Sin una zona de baja densidad donde")
print("apoyar la frontera, la propagación no tiene nada a lo que agarrarse y")
print("acaba determinada por el azar del muestreo: no es que estorbe siempre,")
print("es que su respuesta deja de tener que ver con los datos.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.35, edgecolors="none")
ax[0].scatter(X[sel_fig, 0], X[sel_fig, 1], c="k", s=90, marker="*")
ax[0].set_title("2 etiquetas por clase (estrellas)")
ax[1].scatter(X[:, 0], X[:, 1], c=F[:, 1], cmap="coolwarm", s=10, edgecolors="none")
ax[1].set_title("f propagada por el grafo")
ax[2].scatter(X2[:, 0], X2[:, 1], c=y2, cmap="coolwarm", s=8, edgecolors="none")
ax[2].set_title("parte 2: la frontera cruza lo denso")
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
