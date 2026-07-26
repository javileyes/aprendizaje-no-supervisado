"""
Mixtura de gaussianas ajustada con EM, desde cero y estable en logaritmos.

Usamos los MISMOS datos alargados con los que k-means fracasaba en el capítulo 3
(52,3 % de acuerdo, el nivel del azar) para ver qué gana el modelo al permitir
que cada componente tenga su propia matriz de covarianza.

Se imprime la log-verosimilitud de cada iteración: debe SUBIR siempre. Esa
monotonía es el teorema central de EM, no una casualidad numérica.

Ejecútalo con:  python code/04-gmm-y-em/gmm_em.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

# --- Los datos del capítulo 3, caso 1: dos cigarros paralelos ---
estira = np.array([[3.0, 0.0], [0.0, 0.6]])
th = np.deg2rad(25)
giro = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
M = giro @ estira
X = np.vstack([rng.normal(size=(150, 2)) @ M.T,
               rng.normal(size=(150, 2)) @ M.T + [-1.0, 3.2]])
y = np.repeat([0, 1], 150)
n, d = X.shape
k = 2


def log_normal(X, mu, Sigma):
    """log N(x | mu, Sigma) para todos los x a la vez, vía Cholesky."""
    L = np.linalg.cholesky(Sigma)                  # Sigma = L L^T
    dif = np.linalg.solve(L, (X - mu).T)           # L^{-1} (x - mu)
    maha = (dif ** 2).sum(axis=0)                  # Mahalanobis al cuadrado
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


def logsumexp(A, axis):
    m = A.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def acuerdo(a, b):
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


def kmeans(X, k, semilla=0, iters=100):
    """k-means++ + Lloyd, tal cual el capítulo 3. Sirve para inicializar EM."""
    r = np.random.default_rng(semilla)
    C = [X[r.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        C.append(X[r.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        z = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        Cn = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                       for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1), C


# --- Inicialización: k-means da las medias y la partición inicial ---
z0, C0 = kmeans(X, k, semilla=0)
print(f"Punto de partida (k-means): acuerdo {acuerdo(z0, y):.1%}")

mu = C0.copy()
Sigma = np.array([np.cov(X[z0 == j], rowvar=False) for j in range(k)])
pi = np.array([(z0 == j).mean() for j in range(k)])

REG = 1e-6 * np.eye(d)          # regularización: evita covarianzas singulares
historia = []

for it in range(300):
    # ---------- PASO E: responsabilidades gamma_ij = p(z=j | x_i) ----------
    log_p = np.column_stack([np.log(pi[j]) + log_normal(X, mu[j], Sigma[j])
                             for j in range(k)])
    ll = logsumexp(log_p, axis=1)                  # log p(x_i) de cada dato
    gamma = np.exp(log_p - ll[:, None])
    historia.append(ll.sum())

    # ---------- PASO M: los máximos, en forma cerrada ----------
    N = gamma.sum(axis=0)                          # masa efectiva de cada componente
    pi = N / n
    mu = (gamma.T @ X) / N[:, None]
    for j in range(k):
        dif = X - mu[j]
        Sigma[j] = (gamma[:, j, None] * dif).T @ dif / N[j] + REG

    if it > 0 and abs(historia[-1] - historia[-2]) < 1e-8:
        print(f"Convergió en la iteración {it}.")
        break

z = gamma.argmax(axis=1)
subidas = np.diff(historia)
print(f"\nLog-verosimilitud: {historia[0]:.3f} (inicio) -> {historia[-1]:.3f} (final)")
print(f"Iteraciones: {len(historia)} | ¿alguna bajada? "
      f"{'NO, monótona' if (subidas >= -1e-9).all() else 'SÍ (fallo)'}")
print(f"Pesos de la mixtura: {np.round(pi, 3)}")
print(f"Acuerdo por parejas con la verdad: {acuerdo(z, y):.1%}")

# La asignación blanda dice además CUÁNTO duda el modelo en cada punto.
maxima = gamma.max(axis=1)
print(f"\nPuntos con responsabilidad máxima por debajo de 0,9: {(maxima < 0.9).sum()} de {n}")
print(f"Entropía media de la asignación: "
      f"{-(gamma * np.log(gamma + 1e-12)).sum(axis=1).mean():.4f} nats "
      f"(el máximo posible con k=2 es {np.log(2):.4f})")

# --- Dibujo: elipses de covarianza a 2 sigma y curva de log-verosimilitud ---
t = np.linspace(0, 2 * np.pi, 200)
circulo = np.column_stack([np.cos(t), np.sin(t)])

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=z0, cmap="coolwarm", s=10, edgecolors="none")
ax[0].set_title(f"k-means: {acuerdo(z0, y):.0%}")
ax[1].scatter(X[:, 0], X[:, 1], c=gamma[:, 0], cmap="coolwarm", s=10, edgecolors="none")
for j in range(k):
    val, vec = np.linalg.eigh(Sigma[j])
    elipse = circulo * 2 * np.sqrt(val) @ vec.T + mu[j]
    ax[1].plot(elipse[:, 0], elipse[:, 1], "k-", lw=2)
ax[1].set_title(f"GMM: {acuerdo(z, y):.0%} (color = responsabilidad)")
for a in ax[:2]:
    a.set_xticks([])
    a.set_yticks([])
    a.set_aspect("equal")
ax[2].plot(historia, "o-", ms=3, color="#43c59e")
ax[2].set_xlabel("iteración")
ax[2].set_ylabel("log-verosimilitud")
ax[2].set_title("EM nunca baja")
plt.tight_layout()
plt.show()
