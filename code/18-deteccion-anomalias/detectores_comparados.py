"""
Cinco detectores de anomalías, implementados desde cero y comparados en tres
escenarios distintos. La conclusión es que NINGUNO gana siempre, y que cuál
gana depende del TIPO de anomalía.

Detectores:
  1. Mahalanobis   : ¿está lejos del centro, medido con la covarianza? (cap. 2)
  2. distancia al k-ésimo vecino
  3. LOF           : ¿es su densidad local peor que la de SUS vecinos?
  4. Isolation Forest : ¿cuántos cortes al azar hacen falta para aislarlo?
  5. reconstrucción por PCA : ¿cuánto se sale del subespacio principal? (cap. 9)

Escenarios:
  A. anomalías GLOBALES: puntos lejos de todo
  B. anomalías LOCALES: puntos en un hueco entre dos grupos de densidad distinta
  C. anomalías de SUBESPACIO: puntos que rompen la correlación, sin ser extremos

Se mide con el área bajo la curva ROC (1 = perfecto, 0,5 = azar).

Ejecútalo con:  python code/18-deteccion-anomalias/detectores_comparados.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def auc(puntuacion, es_anomalia):
    """Área bajo la ROC vía el estadístico de Mann-Whitney."""
    orden = np.argsort(puntuacion)
    rangos = np.empty(len(puntuacion), float)
    rangos[orden] = np.arange(1, len(puntuacion) + 1)
    n1 = es_anomalia.sum()
    n0 = len(puntuacion) - n1
    return (rangos[es_anomalia.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def dist2(X):
    sq = (X ** 2).sum(axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)


# ---------------- 1) Mahalanobis ----------------
def mahalanobis(X):
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False) + 1e-8 * np.eye(X.shape[1])
    Si = np.linalg.inv(S)
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, Si, d)


# ---------------- 2) Distancia al k-ésimo vecino ----------------
def knn_dist(X, k=10):
    D = np.sqrt(dist2(X))
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, k - 1]


# ---------------- 3) LOF (Breunig et al., 2000) ----------------
def lof(X, k=20):
    D = np.sqrt(dist2(X))
    np.fill_diagonal(D, np.inf)
    orden = np.argsort(D, axis=1)
    vecinos = orden[:, :k]
    d_k = D[np.arange(len(X)), orden[:, k - 1]]          # distancia k-ésima
    # distancia de alcance: max(d_k(o), d(p,o)) -- suaviza las fluctuaciones
    alcance = np.maximum(d_k[vecinos], D[np.arange(len(X))[:, None], vecinos])
    lrd = 1.0 / (alcance.mean(axis=1) + 1e-12)           # densidad local
    return lrd[vecinos].mean(axis=1) / (lrd + 1e-12)     # LOF = cociente


# ---------------- 4) Isolation Forest (Liu et al., 2008) ----------------
def c_factor(n):
    """Longitud media de camino en un árbol de búsqueda binario con n nodos."""
    if n <= 1:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def camino(X, idx, prof, prof_max, r):
    """Longitud del camino hasta aislar cada punto, con cortes al azar."""
    if prof >= prof_max or len(idx) <= 1:
        return {i: prof + c_factor(len(idx)) for i in idx}
    j = r.integers(X.shape[1])
    lo, hi = X[idx, j].min(), X[idx, j].max()
    if lo == hi:
        return {i: prof + c_factor(len(idx)) for i in idx}
    corte = r.uniform(lo, hi)
    izq = idx[X[idx, j] < corte]
    der = idx[X[idx, j] >= corte]
    out = camino(X, izq, prof + 1, prof_max, r)
    out.update(camino(X, der, prof + 1, prof_max, r))
    return out


def isolation_forest(X, n_arboles=100, muestra=256, semilla=0):
    r = np.random.default_rng(semilla)
    n = len(X)
    m = min(muestra, n)
    prof_max = int(np.ceil(np.log2(m)))
    total = np.zeros(n)
    cuenta = np.zeros(n)
    for _ in range(n_arboles):
        sub = r.choice(n, m, replace=False)
        largos = camino(X, sub, 0, prof_max, r)
        for i, v in largos.items():
            total[i] += v
            cuenta[i] += 1
    medio = total / np.maximum(cuenta, 1)
    medio[cuenta == 0] = c_factor(m)                     # puntos nunca muestreados
    return 2.0 ** (-medio / c_factor(m))                 # más alto = más anómalo


# ---------------- 5) Error de reconstrucción con PCA ----------------
def pca_recon(X, k=1):
    mu = X.mean(axis=0)
    Xc = X - mu
    S = (Xc.T @ Xc) / (len(X) - 1)
    val, vec = np.linalg.eigh(S)
    U = vec[:, ::-1][:, :k]
    return ((Xc - (Xc @ U) @ U.T) ** 2).sum(axis=1)


DETECTORES = [("Mahalanobis", mahalanobis),
              ("dist. al 10º vecino", lambda X: knn_dist(X, 10)),
              ("LOF (k=20)", lambda X: lof(X, 20)),
              ("Isolation Forest", isolation_forest),
              ("reconstrucción PCA", lambda X: pca_recon(X, 1))]

# ---------------- Escenario A: anomalías globales ----------------
XA = np.vstack([rng.normal(scale=1.0, size=(400, 2)),
                rng.uniform(-9, 9, size=(20, 2))])
yA = np.r_[np.zeros(400), np.ones(20)]

# ---------------- Escenario B: anomalías locales ----------------
denso = rng.normal(scale=0.35, size=(250, 2)) + [0.0, 0.0]
disperso = rng.normal(scale=2.2, size=(170, 2)) + [9.0, 0.0]
locales = np.column_stack([rng.normal(1.6, 0.12, 20), rng.normal(0.0, 0.12, 20)])
XB = np.vstack([denso, disperso, locales])
yB = np.r_[np.zeros(420), np.ones(20)]

# ---------------- Escenario C: anomalías de subespacio ----------------
lat = rng.normal(size=400)
XC_norm = np.column_stack([lat, 0.95 * lat]) + 0.12 * rng.normal(size=(400, 2))
XC_anom = np.column_stack([rng.normal(size=20), -0.95 * rng.normal(size=20)])
XC = np.vstack([XC_norm, XC_anom])
yC = np.r_[np.zeros(400), np.ones(20)]

escenarios = [("A · globales", XA, yA), ("B · locales", XB, yB),
              ("C · subespacio", XC, yC)]

print(f"{'detector':22} " + " ".join(f"{n:>16}" for n, _, _ in escenarios))
tabla = {}
for nombre, f in DETECTORES:
    fila = []
    for _, X, yv in escenarios:
        Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        a = auc(f(Xn), yv)
        fila.append(a)
        tabla[(nombre, _)] = a
    print(f"{nombre:22} " + " ".join(f"{a:15.3f} " for a in fila))

print("\n(AUC: 1,000 = perfecto, 0,500 = azar. Nadie gana en las tres columnas.)")

# --- ¿Cuánto se degrada todo esto en dimensión alta? ---
print("\nLa maldición del capítulo 2, aplicada a la detección de anomalías:")
print(f"{'dimensión':>10} " + " ".join(f"{n:>20}" for n, _ in DETECTORES[:3]))
for d in [2, 10, 50, 200]:
    r = np.random.default_rng(3)
    Xn_ = r.normal(size=(400, d))
    Xa_ = r.normal(size=(20, d)) * 1.0
    Xa_[:, 0] += 6.0                       # la anomalía solo afecta a UNA dimensión
    Xd = np.vstack([Xn_, Xa_])
    yd = np.r_[np.zeros(400), np.ones(20)]
    Xd = (Xd - Xd.mean(axis=0)) / (Xd.std(axis=0) + 1e-9)
    print(f"{d:10d} " + " ".join(f"{auc(f(Xd), yd):19.3f} " for _, f in DETECTORES[:3]))
print("La señal está siempre en la misma dimensión, pero se diluye entre las")
print("demás. Por eso en dimensión alta se detectan anomalías DESPUÉS de reducir.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, (nombre, X, yv) in zip(ax, escenarios):
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    s = lof(Xn, 20)
    a.scatter(X[yv == 0, 0], X[yv == 0, 1], s=10, c="#8b93a7", edgecolors="none")
    a.scatter(X[yv == 1, 0], X[yv == 1, 1], s=45, c="#ef476f", marker="X")
    a.set_title(f"{nombre} · LOF: AUC {auc(s, yv):.2f}")
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
