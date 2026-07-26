"""
El estadístico de hueco (gap) de Tibshirani, y la pregunta que nadie hace:
¿y si NO hay estructura?

Gap(k) = E*[log W_k] - log W_k, donde W_k es la inercia y E* es la esperanza
sobre datos de REFERENCIA sin estructura (uniformes en la envolvente de los
datos). Se elige el menor k tal que Gap(k) >= Gap(k+1) - s_{k+1}.

La virtud del gap frente a la silueta es que puede responder "k = 1", es decir,
"aquí no hay grupos". Lo comprobamos con datos uniformes: cualquier índice de
los del fichero anterior elegiría algún k, pero el gap dice que no hay nada.

Ejecútalo con:  python code/08-validacion-clusteres/gap_y_sin_estructura.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
B_REF = 20          # número de conjuntos de referencia por cada k


def kmeans_inercia(X, k, semilla=0, iters=100):
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
    return ((X - C[z]) ** 2).sum()


def gap(X, ks, semilla=0):
    lo, hi = X.min(axis=0), X.max(axis=0)
    r = np.random.default_rng(semilla)
    gaps, sks, logW = [], [], []
    for k in ks:
        W = kmeans_inercia(X, k, semilla=0)
        logW.append(np.log(max(W, 1e-12)))
        refs = []
        for b in range(B_REF):
            Xr = r.uniform(lo, hi, size=X.shape)      # referencia sin estructura
            refs.append(np.log(max(kmeans_inercia(Xr, k, semilla=b), 1e-12)))
        refs = np.array(refs)
        gaps.append(refs.mean() - logW[-1])
        sks.append(refs.std() * np.sqrt(1 + 1.0 / B_REF))
    return np.array(gaps), np.array(sks), np.array(logW)


def elegir_k(ks, gaps, sks):
    """Regla de Tibshirani: el menor k con Gap(k) >= Gap(k+1) - s_{k+1}."""
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return ks[i]
    return ks[-1]


# ---------------- Caso 1: cuatro grupos de verdad ----------------
centros = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
X1 = np.vstack([c + rng.normal(scale=0.9, size=(75, 2)) for c in centros])

# ---------------- Caso 2: nube uniforme, SIN estructura ninguna ----------------
X2 = rng.uniform([0, 0], [10, 10], size=(300, 2))

ks = list(range(1, 9))
for nombre, X, verdad in [("cuatro grupos", X1, "4"), ("uniforme (sin grupos)", X2, "1")]:
    g, s, lw = gap(X, ks)
    print(f"\n{nombre}   (respuesta correcta: k = {verdad})")
    print(f"{'k':>3} {'log W_k':>10} {'Gap(k)':>9} {'s_k':>8}   ¿se para aquí?")
    for i, k in enumerate(ks):
        if i < len(ks) - 1:
            para = "SÍ" if g[i] >= g[i + 1] - s[i + 1] else "no"
        else:
            para = "-"
        print(f"{k:3d} {lw[i]:10.4f} {g[i]:9.4f} {s[i]:8.4f}   {para}")
    print(f"    -> el gap elige k = {elegir_k(ks, g, s)}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for f, (nombre, X) in enumerate([("cuatro grupos", X1), ("uniforme", X2)]):
    g, s, lw = gap(X, ks)
    ax[f, 0].scatter(X[:, 0], X[:, 1], s=10, c="#8b93a7")
    ax[f, 0].set_title(nombre)
    ax[f, 0].set_xticks([])
    ax[f, 0].set_yticks([])
    ax[f, 1].errorbar(ks, g, yerr=s, fmt="o-", color="#5b8def", capsize=3)
    ax[f, 1].axvline(elegir_k(ks, g, s), ls="--", c="#ef476f")
    ax[f, 1].set_xlabel("k")
    ax[f, 1].set_ylabel("Gap(k)")
    ax[f, 1].set_title(f"gap -> k = {elegir_k(ks, g, s)}")
plt.tight_layout()
plt.show()
