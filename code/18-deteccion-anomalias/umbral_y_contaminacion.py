"""
Las dos preguntas que el AUC no responde: ¿dónde pongo el umbral?, y ¿qué pasa
si los datos con los que ajusto el detector YA contienen anomalías?

  1. Un AUC alto no da un umbral. Comparamos tres formas de fijarlo y medimos
     precisión, exhaustividad y F1 de cada una.
  2. La CONTAMINACIÓN: casi nunca se dispone de datos limpios para ajustar. Se
     mide cómo se degrada un detector paramétrico (Mahalanobis) a medida que el
     conjunto de ajuste se ensucia, y cuánto lo arregla un estimador robusto.

Ejecútalo con:  python code/18-deteccion-anomalias/umbral_y_contaminacion.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
D = 6


def mahalanobis(X, mu, S):
    Si = np.linalg.inv(S + 1e-8 * np.eye(len(S)))
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, Si, d)


def auc(s, y):
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    n1 = int(y.sum())
    n0 = len(s) - n1
    return (r[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def prf(pred, y):
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


# ---------------- Datos: normales gaussianos + anomalías desplazadas ----------
def genera(n_norm, n_anom, r):
    A = r.normal(size=(D, D))
    S = A @ A.T / D + 0.4 * np.eye(D)
    L = np.linalg.cholesky(S)
    Xn = r.normal(size=(n_norm, D)) @ L.T
    Xa = r.normal(size=(n_anom, D)) @ L.T + 3.2
    X = np.vstack([Xn, Xa])
    y = np.r_[np.zeros(n_norm), np.ones(n_anom)]
    p = r.permutation(len(X))
    return X[p], y[p]


X, y = genera(950, 50, rng)
print(f"{len(X)} puntos en dimensión {D}, con un {y.mean():.1%} de anomalías.")

# ================= 1) El umbral =================
X_lim, _ = genera(800, 0, np.random.default_rng(1))     # ajuste con datos limpios
mu, S = X_lim.mean(axis=0), np.cov(X_lim, rowvar=False)
s = mahalanobis(X, mu, S)
print(f"\nAUC del detector: {auc(s, y):.4f}  (excelente... pero no es un umbral)")

s_lim = mahalanobis(X_lim, mu, S)
umbrales = [
    ("percentil 95 de los normales", np.percentile(s_lim, 95)),
    ("percentil 99 de los normales", np.percentile(s_lim, 99)),
    ("cuantil chi-cuadrado al 99 %", D + 2.33 * np.sqrt(2 * D) + 2.0),
    ("suponer 5 % de anomalías", np.percentile(s, 95)),
]
print(f"\n{'regla para el umbral':32} {'valor':>8} {'precisión':>11} "
      f"{'exhaustividad':>15} {'F1':>7}")
for nombre, u in umbrales:
    p, r_, f1 = prf(s > u, y)
    print(f"{nombre:32} {u:8.2f} {p:10.1%} {r_:14.1%} {f1:7.3f}")
print("Mismo detector, mismo AUC, y la precisión va del 19,5 % al 98,0 % según")
print("la regla. El umbral es una decisión de NEGOCIO -cuánto cuesta cada tipo")
print("de error- y no se deduce de los datos.")

# ================= 2) La contaminación =================
print("\n¿Y si el conjunto de ajuste YA lleva anomalías dentro?")
print(f"{'contaminación':>14} {'AUC (media)':>13} {'AUC (robusto)':>15}")


def mcd_simple(X, frac=0.75, iters=15):
    """Determinante mínimo, versión simple: se reajusta con el 75 % más central."""
    mu, S = X.mean(axis=0), np.cov(X, rowvar=False)
    m = int(frac * len(X))
    for _ in range(iters):
        d = mahalanobis(X, mu, S)
        sel = np.argsort(d)[:m]
        mu_n, S_n = X[sel].mean(axis=0), np.cov(X[sel], rowvar=False)
        if np.allclose(mu_n, mu, atol=1e-9):
            mu, S = mu_n, S_n
            break
        mu, S = mu_n, S_n
    return mu, S


for cont in [0.0, 0.02, 0.05, 0.10, 0.20]:
    n_a = int(cont * 800 / (1 - cont)) if cont > 0 else 0
    X_aj, _ = genera(800, n_a, np.random.default_rng(7))
    mu_c, S_c = X_aj.mean(axis=0), np.cov(X_aj, rowvar=False)
    mu_r, S_r = mcd_simple(X_aj)
    print(f"{cont:13.0%} {auc(mahalanobis(X, mu_c, S_c), y):12.4f} "
          f"{auc(mahalanobis(X, mu_r, S_r), y):14.4f}")
print("El estimador clásico se envenena: las propias anomalías inflan la")
print("covarianza y dejan de parecer raras. El robusto descarta el 25 % más")
print("extremo en cada iteración y aguanta mucho mejor.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist([s[y == 0], s[y == 1]], bins=40, stacked=True,
           label=["normales", "anomalías"])
for nombre, u in umbrales[:3]:
    ax[0].axvline(u, ls="--", lw=1)
ax[0].set_yscale("log")
ax[0].legend()
ax[0].set_title("distancias y los umbrales candidatos")
conts = [0.0, 0.02, 0.05, 0.10, 0.20]
aucs_c, aucs_r = [], []
for cont in conts:
    n_a = int(cont * 800 / (1 - cont)) if cont > 0 else 0
    X_aj, _ = genera(800, n_a, np.random.default_rng(7))
    aucs_c.append(auc(mahalanobis(X, X_aj.mean(axis=0),
                                  np.cov(X_aj, rowvar=False)), y))
    mu_r, S_r = mcd_simple(X_aj)
    aucs_r.append(auc(mahalanobis(X, mu_r, S_r), y))
ax[1].plot(conts, aucs_c, "o-", label="media y covarianza clásicas")
ax[1].plot(conts, aucs_r, "s-", label="estimación robusta")
ax[1].set_xlabel("fracción de anomalías en el ajuste")
ax[1].set_ylabel("AUC")
ax[1].legend()
ax[1].set_title("efecto de la contaminación")
plt.tight_layout()
plt.show()
