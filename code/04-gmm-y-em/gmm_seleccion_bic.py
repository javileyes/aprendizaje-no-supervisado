"""
¿Cuántas componentes? Con un modelo probabilístico, la pregunta tiene respuesta.

A diferencia de la inercia de k-means -que siempre baja al subir k y por tanto
no sirve para elegirlo-, la verosimilitud de una mixtura se puede penalizar por
el número de parámetros. Eso es el BIC (Schwarz, 1978) y el AIC (Akaike, 1974).

Generamos datos de TRES componentes con formas distintas y comprobamos si los
criterios lo aciertan.

Ejecútalo con:  python code/04-gmm-y-em/gmm_seleccion_bic.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# Tres componentes de verdad, con covarianzas deliberadamente distintas
comp = [
    (np.array([0.0, 0.0]), np.array([[1.6, 0.9], [0.9, 0.8]]), 200),
    (np.array([6.0, 1.0]), np.array([[0.4, 0.0], [0.0, 2.4]]), 150),
    (np.array([2.5, 6.0]), np.array([[1.0, -0.6], [-0.6, 1.0]]), 150),
]
X = np.vstack([mu + rng.normal(size=(m, 2)) @ np.linalg.cholesky(S).T
               for mu, S, m in comp])
n, d = X.shape
print(f"Datos: {n} puntos generados por 3 componentes gaussianas.\n")


def log_normal(X, mu, Sigma):
    L = np.linalg.cholesky(Sigma)
    dif = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (X.shape[1] * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum()
                   + (dif ** 2).sum(axis=0))


def logsumexp(A, axis):
    m = A.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def em(X, k, semilla, iters=200):
    r = np.random.default_rng(semilla)
    mu = X[r.choice(len(X), k, replace=False)]
    Sigma = np.array([np.cov(X, rowvar=False) for _ in range(k)])
    pi = np.full(k, 1.0 / k)
    ll_ant = -np.inf
    for _ in range(iters):
        log_p = np.column_stack([np.log(pi[j]) + log_normal(X, mu[j], Sigma[j])
                                 for j in range(k)])
        ll = logsumexp(log_p, axis=1)
        gamma = np.exp(log_p - ll[:, None])
        total = ll.sum()
        N = gamma.sum(axis=0) + 1e-10
        pi = N / len(X)
        mu = (gamma.T @ X) / N[:, None]
        for j in range(k):
            dif = X - mu[j]
            Sigma[j] = (gamma[:, j, None] * dif).T @ dif / N[j] + 1e-4 * np.eye(d)
        if abs(total - ll_ant) < 1e-7:
            break
        ll_ant = total
    return total


# Parámetros libres de una mixtura completa: pesos (k-1), medias (k*d)
# y covarianzas simétricas (k * d(d+1)/2).
def n_params(k, d):
    return (k - 1) + k * d + k * d * (d + 1) // 2


print(f"{'k':>3} {'nº parámetros':>14} {'log-verosim.':>14} {'BIC':>11} {'AIC':>11}")
ks = range(1, 8)
bics, aics, lls = [], [], []
for k in ks:
    mejor = max(em(X, k, s) for s in range(8))       # 8 arranques, nos quedamos el mejor
    p = n_params(k, d)
    bic = -2 * mejor + p * np.log(n)
    aic = -2 * mejor + 2 * p
    lls.append(mejor)
    bics.append(bic)
    aics.append(aic)
    print(f"{k:3d} {p:14d} {mejor:14.2f} {bic:11.2f} {aic:11.2f}")

k_bic = list(ks)[int(np.argmin(bics))]
k_aic = list(ks)[int(np.argmin(aics))]
print(f"\nBIC elige k = {k_bic}   |   AIC elige k = {k_aic}   |   verdad: k = 3")
print("La log-verosimilitud, en cambio, no deja de subir: por sí sola nunca")
print("elegiría un k finito. La penalización es lo que decide.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X[:, 0], X[:, 1], s=10, c="#8b93a7")
ax[0].set_title("los datos, sin etiquetas")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].plot(list(ks), bics, "o-", label="BIC")
ax[1].plot(list(ks), aics, "s-", label="AIC")
ax[1].axvline(3, ls=":", c="k", label="k verdadero")
ax[1].set_xlabel("número de componentes k")
ax[1].legend()
ax[1].set_title("más bajo es mejor")
plt.tight_layout()
plt.show()
