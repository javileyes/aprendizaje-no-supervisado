"""
Deep clustering: DEC (Xie, Girshick y Farhadi, 2016) e IDEC (Guo, 2017).

La idea: en vez de (a) reducir la dimensión y luego (b) agrupar, hacer las dos
cosas A LA VEZ, de forma que la representación se moldee para ser agrupable.

  - Asignación blanda con t de Student, igual que en t-SNE (cap. 12).
  - Distribución OBJETIVO p, construida elevando q al cuadrado y normalizando
    por la frecuencia de cada grupo: refuerza las asignaciones seguras.
  - Pérdida KL(P||Q), con P recalculada cada cierto número de iteraciones.
  - IDEC añade el término de reconstrucción, y ahí está toda la diferencia.

Los datos: cuatro grupos gaussianos en un espacio latente de dimensión 2, pasados
por una red aleatoria fija hasta 50 dimensiones. La estructura sigue ahí, pero
deformada: k-means en el espacio original lo pasa mal.

Ejecútalo con:  python code/15-deep-clustering/dec_idec.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
K, D_LAT, D_OBS, N = 4, 2, 50, 1200

# --- Datos: 4 grupos en 2D deformados no linealmente hasta 50D ---
centros = np.array([[0.0, 0.0], [3.2, 0.0], [1.6, 2.8], [1.6, -2.8]])
Z_real = np.vstack([c + rng.normal(scale=0.62, size=(N // K, D_LAT)) for c in centros])
y = np.repeat(np.arange(K), N // K)
A1 = rng.normal(size=(D_LAT, 32))
A2 = rng.normal(size=(32, D_OBS))
X = np.tanh(np.tanh(Z_real @ A1) @ A2) + 0.06 * rng.normal(size=(N, D_OBS))
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
print(f"{N} puntos: 4 grupos en 2D deformados hasta {D_OBS} dimensiones.")


def acuerdo(a, b):
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


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
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1), C


# ---------------- El autoencoder (mismo esqueleto que el capítulo 13) ----------------
CAPAS = [D_OBS, 64, 32, D_LAT, 32, 64, D_OBS]
MEDIO = len(CAPAS) // 2 - 1          # índice de la capa que produce el código


def init(semilla=0):
    r = np.random.default_rng(semilla)
    return [[r.normal(scale=np.sqrt(2.0 / (a + b)), size=(a, b)), np.zeros(b)]
            for a, b in zip(CAPAS[:-1], CAPAS[1:])]


def adelante(par, X):
    acts, A = [X], X
    for i, (W, b) in enumerate(par):
        Zl = A @ W + b
        A = Zl if (i == len(par) - 1 or i == MEDIO) else np.tanh(Zl)
        acts.append(A)
    return acts


def atras(par, acts, dA_final, desde=None):
    """Retropropaga dA_final desde la capa 'desde' (por defecto, la última)."""
    desde = len(par) - 1 if desde is None else desde
    grads = [[np.zeros_like(W), np.zeros_like(b)] for W, b in par]
    dA = dA_final
    for i in range(desde, -1, -1):
        if i != len(par) - 1 and i != MEDIO:
            dA = dA * (1.0 - acts[i + 1] ** 2)
        grads[i] = [acts[i].T @ dA, dA.sum(axis=0)]
        dA = dA @ par[i][0].T
    return grads


def adam(par, grads, estado, lr, paso):
    m, v = estado
    for i in range(len(par)):
        for j in range(2):
            m[i][j] = 0.9 * m[i][j] + 0.1 * grads[i][j]
            v[i][j] = 0.999 * v[i][j] + 0.001 * grads[i][j] ** 2
            par[i][j] -= lr * (m[i][j] / (1 - 0.9 ** paso)) / (
                np.sqrt(v[i][j] / (1 - 0.999 ** paso)) + 1e-8)


def nuevo_estado(par):
    return ([[np.zeros_like(W), np.zeros_like(b)] for W, b in par],
            [[np.zeros_like(W), np.zeros_like(b)] for W, b in par])


# ---------------- Fase 1: preentrenar el autoencoder ----------------
par = init(0)
est = nuevo_estado(par)
paso = 0
r = np.random.default_rng(1)
for ep in range(300):
    orden = r.permutation(N)
    for i in range(0, N, 128):
        xb = X[orden[i:i + 128]]
        acts = adelante(par, xb)
        g = atras(par, acts, 2.0 * (acts[-1] - xb) / xb.size)
        paso += 1
        adam(par, g, est, 1e-3, paso)
rec_pre = ((adelante(par, X)[-1] - X) ** 2).mean()
Z = adelante(par, X)[MEDIO + 1]
z_km_lat, C0 = kmeans(Z, K, semilla=0)

print(f"\nFase 1 · autoencoder preentrenado: error {rec_pre:.5f}")
print(f"{'k-means en las 50 dimensiones originales':45} "
      f"{acuerdo(kmeans(X, K, 0)[0], y):.1%}")
print(f"{'k-means sobre el código del autoencoder':45} {acuerdo(z_km_lat, y):.1%}")


# ---------------- Fases 2 y 3: DEC e IDEC ----------------
def q_suave(Z, C):
    """t de Student con 1 grado de libertad, igual que en t-SNE."""
    d2 = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    num = 1.0 / (1.0 + d2)
    return num / num.sum(axis=1, keepdims=True), num


def objetivo(q):
    """p_ij propto q_ij^2 / f_j : refuerza lo seguro y equilibra los tamaños."""
    f = q.sum(axis=0)
    w = q ** 2 / f
    return w / w.sum(axis=1, keepdims=True)


def entrena_dec(par0, C0, gamma_rec, iters=3000, lr=1e-3, cada=140):
    par = [[W.copy(), b.copy()] for W, b in par0]
    C = C0.copy()
    est = nuevo_estado(par)
    mC, vC, paso = np.zeros_like(C), np.zeros_like(C), 0
    hist = []
    P = None
    r = np.random.default_rng(2)
    for it in range(iters):
        if it % cada == 0:                       # se recalcula la distribución objetivo
            Zt = adelante(par, X)[MEDIO + 1]
            P = objetivo(q_suave(Zt, C)[0])
            zt = q_suave(Zt, C)[0].argmax(axis=1)
            hist.append((it, acuerdo(zt, y),
                         ((adelante(par, X)[-1] - X) ** 2).mean()))
        idx = r.choice(N, 128, replace=False)
        xb, Pb = X[idx], P[idx]
        acts = adelante(par, xb)
        Zb = acts[MEDIO + 1]
        q, num = q_suave(Zb, C)
        # gradiente de KL(P||Q) respecto a z y a los centros (alpha = 1)
        coef = 2.0 * num * (Pb - q) / len(xb)     # (n, K)
        dif = Zb[:, None, :] - C[None, :, :]      # (n, K, d)
        dZ = (coef[:, :, None] * dif).sum(axis=1)
        gC = -(coef[:, :, None] * dif).sum(axis=0)
        # el término de reconstrucción (IDEC) entra por la última capa
        g = atras(par, acts, gamma_rec * 2.0 * (acts[-1] - xb) / xb.size)
        g_clu = atras(par, acts, dZ, desde=MEDIO)
        for i in range(len(par)):
            for j in range(2):
                g[i][j] = g[i][j] + g_clu[i][j]
        paso += 1
        adam(par, g, est, lr, paso)
        mC = 0.9 * mC + 0.1 * gC
        vC = 0.999 * vC + 0.001 * gC ** 2
        C -= lr * (mC / (1 - 0.9 ** paso)) / (np.sqrt(vC / (1 - 0.999 ** paso)) + 1e-8)
    Zt = adelante(par, X)[MEDIO + 1]
    qf = q_suave(Zt, C)[0]
    zt = qf.argmax(axis=1)
    return zt, Zt, ((adelante(par, X)[-1] - X) ** 2).mean(), hist, qf


print("\nFases 2 y 3 · afinado conjunto de representación y clústeres")
resultados = {}
q_ini = q_suave(Z, C0)[0]
for nombre, gamma in [("DEC (solo KL)", 0.0), ("IDEC (KL + reconstrucción)", 1.0)]:
    z_f, Z_f, rec_f, hist, q_f = entrena_dec(par, C0, gamma)
    resultados[nombre] = (z_f, Z_f, rec_f, hist, q_f)
    print(f"\n{nombre}")
    print(f"{'iteración':>10} {'acuerdo':>9} {'error de reconstrucción':>25}")
    for it, ac, rc in hist[::4] + [hist[-1]]:
        print(f"{it:10d} {ac:8.1%} {rc:25.5f}")
    print(f"  final: acuerdo {acuerdo(z_f, y):.1%}, reconstrucción {rec_f:.5f} "
          f"(al empezar: {rec_pre:.5f})")

def confianza(q):
    return q.max(axis=1).mean(), (q.max(axis=1) > 0.9).mean()


print("\nResumen")
print(f"{'método':38} {'acuerdo':>9} {'reconstr.':>11} {'confianza':>11} {'q>0,9':>9}")
print(f"{'k-means en el espacio original':38} {acuerdo(kmeans(X, K, 0)[0], y):8.1%} "
      f"{'-':>11} {'-':>11} {'-':>9}")
c, f = confianza(q_ini)
print(f"{'autoencoder + k-means':38} {acuerdo(z_km_lat, y):8.1%} {rec_pre:11.5f} "
      f"{c:11.4f} {f:8.1%}")
for nombre, (z_f, _, rec_f, _, q_f) in resultados.items():
    c, f = confianza(q_f)
    print(f"{nombre:38} {acuerdo(z_f, y):8.1%} {rec_f:11.5f} {c:11.4f} {f:8.1%}")
# ---------------- El colapso: DEC SIN preentrenar ----------------
print("\n¿Y si nos saltamos el preentrenamiento? (DEC desde pesos al azar)")
par_azar = init(7)
Z_azar = adelante(par_azar, X)[MEDIO + 1]
C_azar = kmeans(Z_azar, K, semilla=0)[1]
z_c, Z_c, rec_c, _, q_c = entrena_dec(par_azar, C_azar, 0.0)
ocupados = int((np.bincount(z_c, minlength=K) > 0).sum())
print(f"  acuerdo con la verdad     : {acuerdo(z_c, y):.1%}")
print(f"  grupos no vacíos          : {ocupados} de {K}")
print(f"  tamaños                   : {np.bincount(z_c, minlength=K).tolist()}")
print(f"  dispersión del latente    : {Z_c.std():.4f}  "
      f"(con preentrenamiento: {resultados['DEC (solo KL)'][1].std():.4f})")
print("  No colapsa a un solo grupo, pero se queda al nivel de k-means en el")
print("  espacio original: el objetivo KL(P||Q) solo REFUERZA la partición con la")
print("  que arranca, no la descubre. Autoentrenarse sobre las propias creencias")
print("  amplifica lo que ya había, aciertos y errores. Por eso DEC se preentrena.")

print("\nLo que DEC cambia de verdad no es el ACUERDO (sube poco) sino la")
print("CONFIANZA: la asignación blanda pasa de dudosa a casi binaria. Es lo que")
print("busca la distribución objetivo p, que eleva q al cuadrado.")
print("\nDEC agrupa muy bien PERO destroza la reconstrucción: ha deformado el")
print("latente hasta hacerlo agrupable, sin ninguna obligación de seguir")
print("representando los datos. IDEC consigue lo mismo sin romper nada.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=8, edgecolors="none")
ax[0].set_title(f"tras preentrenar · {acuerdo(z_km_lat, y):.0%}")
for a, nombre in zip(ax[1:], resultados):
    z_f, Z_f = resultados[nombre][0], resultados[nombre][1]
    a.scatter(Z_f[:, 0], Z_f[:, 1], c=y, cmap="viridis", s=8, edgecolors="none")
    a.set_title(f"{nombre.split(' ')[0]} · {acuerdo(z_f, y):.0%}")
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
