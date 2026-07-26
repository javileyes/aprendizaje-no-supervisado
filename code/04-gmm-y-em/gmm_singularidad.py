"""
La patología de la máxima verosimilitud en mixturas: NO tiene máximo global
finito. Se puede hacer la verosimilitud tan grande como se quiera.

Parte A (analítica): colocamos una componente EXACTAMENTE encima del dato x_0
con covarianza sigma^2 * I y bajamos sigma. La log-verosimilitud crece sin
límite, aproximadamente como -d*log(sigma).

Parte B (con EM): sembramos una componente sobre un dato con una covarianza
minúscula y dejamos correr EM. Colapsa en dos iteraciones. Con una
regularización de 1e-3 en la diagonal, no.

Ejecútalo con:  python code/04-gmm-y-em/gmm_singularidad.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(size=(60, 2)), rng.normal(size=(60, 2)) + [4.0, 0.0]])
n, d = X.shape
k = 2


def log_normal(X, mu, Sigma):
    L = np.linalg.cholesky(Sigma)
    dif = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum()
                   + (dif ** 2).sum(axis=0))


def logsumexp(A, axis):
    m = A.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def log_verosimilitud(mu, Sigma, pi):
    log_p = np.column_stack([np.log(pi[j]) + log_normal(X, mu[j], Sigma[j])
                             for j in range(k)])
    return logsumexp(log_p, axis=1).sum()


# ---------------- Parte A: la verosimilitud no está acotada ----------------
mu_fijo = np.array([X[0], X[1:].mean(axis=0)])
Sigma_resto = np.cov(X[1:], rowvar=False)
pi_fijo = np.array([0.5, 0.5])

print("Parte A · una componente clavada sobre x_0, con covarianza sigma^2 I")
print(f"{'sigma':>10} {'log-verosimilitud':>19} {'ll + d*log(sigma)':>19}")
sigmas = [1.0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
lls = []
for s in sigmas:
    Sigma = np.array([s ** 2 * np.eye(d), Sigma_resto])
    ll = log_verosimilitud(mu_fijo, Sigma, pi_fijo)
    lls.append(ll)
    print(f"{s:10.0e} {ll:19.3f} {ll + d * np.log(s):19.3f}")
print("La tercera columna se estabiliza: eso significa que la log-verosimilitud")
print("vale una constante MENOS d*log(sigma), y por tanto diverge a +infinito")
print("cuando sigma -> 0. No hay máximo global que alcanzar.\n")


# ---------------- Parte B: EM se lanza a esa singularidad ----------------
def em(reg, iters=12):
    """EM con una componente sembrada MALICIOSAMENTE encima de un dato."""
    mu = np.array([X[0].copy(), X.mean(axis=0)])
    Sigma = np.array([1e-4 * np.eye(d), np.cov(X, rowvar=False)])
    pi = np.array([0.5, 0.5])
    traza = []
    for _ in range(iters):
        try:
            log_p = np.column_stack([np.log(pi[j]) + log_normal(X, mu[j], Sigma[j])
                                     for j in range(k)])
        except np.linalg.LinAlgError:
            traza.append((np.inf, 0.0, traza[-1][2]))
            break
        ll = logsumexp(log_p, axis=1)
        gamma = np.exp(log_p - ll[:, None])
        traza.append((ll.sum(), np.linalg.det(Sigma[0]), gamma[:, 0].sum()))
        N = gamma.sum(axis=0)
        pi = N / n
        mu = (gamma.T @ X) / N[:, None]
        for j in range(k):
            dif = X - mu[j]
            Sigma[j] = (gamma[:, j, None] * dif).T @ dif / N[j] + reg * np.eye(d)
        if np.linalg.det(Sigma[0]) <= 0:
            traza.append((np.inf, 0.0, N[0]))
            break
    return traza


print("Parte B · EM sin regularización (reg = 0)")
print(f"{'iter':>5} {'log-verosimilitud':>19} {'det(Sigma_0)':>15} {'puntos en comp. 0':>19}")
for i, (ll, det, N0) in enumerate(em(reg=0.0)):
    ll_txt = "  +inf (colapso)" if not np.isfinite(ll) else f"{ll:19.3f}"
    print(f"{i:5d} {ll_txt:>19} {det:15.3e} {N0:19.4f}")

print("\nParte B · EM con regularización (reg = 1e-3)")
buena = em(reg=1e-3)
print(f"{'iter':>5} {'log-verosimilitud':>19} {'det(Sigma_0)':>15} {'puntos en comp. 0':>19}")
for i, (ll, det, N0) in enumerate(buena):
    if i < 4 or i == len(buena) - 1:
        print(f"{i:5d} {ll:19.3f} {det:15.3e} {N0:19.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(sigmas, lls, "o-", color="#ef476f")
ax[0].invert_xaxis()
ax[0].set_xlabel("sigma de la componente colapsada")
ax[0].set_ylabel("log-verosimilitud")
ax[0].set_title("A · crece sin límite")
ax[1].plot([t[0] for t in buena], "o-", ms=4, color="#43c59e")
ax[1].set_xlabel("iteración")
ax[1].set_ylabel("log-verosimilitud")
ax[1].set_title("B · con reg = 1e-3, EM se comporta")
plt.tight_layout()
plt.show()
