"""
Un autoencoder NO lineal supera a PCA cuando la variedad está curvada.

Red: d -> 64 -> 32 -> k -> 32 -> 64 -> d, con tanh. Backpropagation escrita a
mano, sin librerías de autodiferenciación, y VERIFICADA contra diferencias
finitas: si el gradiente analítico y el numérico coinciden, el código es correcto.

Datos: una hélice. Es una variedad de dimensión 1 -basta un número, la posición
a lo largo del muelle- pero curvada y metida en 20 dimensiones. Con k=1, PCA
solo puede ofrecer una RECTA, y una hélice no lo es.

Ejecútalo con:  python code/13-autoencoders/autoencoder_no_lineal.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
n, d_amb, k = 900, 20, 1

# --- Hélice (variedad 1D curvada) incrustada en 20 dimensiones ---
t = np.linspace(0, 6 * np.pi, n) + rng.normal(scale=0.01, size=n)
X3 = np.column_stack([np.cos(t), np.sin(t), t / 6])
Q = np.linalg.qr(rng.normal(size=(d_amb, 3)))[0]
X = X3 @ Q.T + 0.01 * rng.normal(size=(n, d_amb))
X -= X.mean(axis=0)
X /= X.std(axis=0).mean()
print(f"{n} puntos sobre una hélice: dimensión intrínseca 1, "
      f"incrustada en {d_amb}.")

CAPAS = [d_amb, 64, 32, k, 32, 64, d_amb]


def init(semilla=0):
    r = np.random.default_rng(semilla)
    par = []
    for a, b in zip(CAPAS[:-1], CAPAS[1:]):
        par.append([r.normal(scale=np.sqrt(2.0 / (a + b)), size=(a, b)), np.zeros(b)])
    return par


def adelante(par, X):
    """Guarda las activaciones para poder retropropagar."""
    acts = [X]
    A = X
    for i, (W, b) in enumerate(par):
        Zl = A @ W + b
        A = Zl if i == len(par) - 1 else np.tanh(Zl)   # última capa, lineal
        acts.append(A)
    return acts


def perdida_y_grad(par, X):
    acts = adelante(par, X)
    R = acts[-1]
    E = R - X
    L = (E ** 2).mean()
    grads = [None] * len(par)
    dA = 2.0 * E / X.size                              # dL/dR
    for i in range(len(par) - 1, -1, -1):
        A_ant = acts[i]
        if i != len(par) - 1:
            dA = dA * (1.0 - acts[i + 1] ** 2)         # derivada de tanh
        grads[i] = [A_ant.T @ dA, dA.sum(axis=0)]
        dA = dA @ par[i][0].T
    return L, grads


# ---------- Verificación del gradiente con diferencias finitas ----------
par = init(0)
X_mini = X[:12]
L0, g = perdida_y_grad(par, X_mini)
eps = 1e-6
errores = []
for capa in sorted({0, len(par) // 2, len(par) - 1}):
    W = par[capa][0]
    for _ in range(4):
        i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
        orig = W[i, j]
        W[i, j] = orig + eps
        Lp = perdida_y_grad(par, X_mini)[0]
        W[i, j] = orig - eps
        Lm = perdida_y_grad(par, X_mini)[0]
        W[i, j] = orig
        num = (Lp - Lm) / (2 * eps)
        errores.append(abs(num - g[capa][0][i, j]) / max(abs(num), 1e-12))
print(f"\nVerificación del gradiente (error relativo frente a diferencias finitas):")
print(f"  máximo sobre {len(errores)} pesos de {len(sorted({0, len(par)//2, len(par)-1}))} capas: {max(errores):.2e}")
print("  Por debajo de 1e-5 significa que la retropropagación es correcta.")

# ---------- Entrenamiento con Adam ----------
par = init(0)
m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in par]
v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in par]
lr, b1, b2, eps_a = 3e-3, 0.9, 0.999, 1e-8
lote, epocas = 128, 600
hist = []
paso = 0
for ep in range(epocas):
    orden = rng.permutation(n)
    for ini in range(0, n, lote):
        Xb = X[orden[ini:ini + lote]]
        L, g = perdida_y_grad(par, Xb)
        paso += 1
        lr_ep = lr * (0.5 ** (ep // 200))          # el paso se reduce a la mitad
        for i in range(len(par)):
            for j in range(2):
                m[i][j] = b1 * m[i][j] + (1 - b1) * g[i][j]
                v[i][j] = b2 * v[i][j] + (1 - b2) * g[i][j] ** 2
                mh = m[i][j] / (1 - b1 ** paso)
                vh = v[i][j] / (1 - b2 ** paso)
                par[i][j] -= lr_ep * mh / (np.sqrt(vh) + eps_a)
    if ep % 60 == 0 or ep == epocas - 1:
        hist.append((ep, perdida_y_grad(par, X)[0]))

print(f"\n{'época':>7} {'error del AE':>14}")
for ep, L in hist:
    print(f"{ep:7d} {L:14.6f}")

err_ae = perdida_y_grad(par, X)[0]

# ---------- PCA con la misma dimensión latente ----------
S = (X.T @ X) / (n - 1)
val, vec = np.linalg.eigh(S)
U = vec[:, ::-1][:, :k]
err_pca = ((X - (X @ U) @ U.T) ** 2).mean()

print(f"\nError de reconstrucción con k = {k}:")
print(f"  PCA          : {err_pca:.6f}")
print(f"  autoencoder  : {err_ae:.6f}")
print(f"  reducción    : {(1 - err_ae / err_pca):.1%}")

# ¿El código recuperó el parámetro que recorre la lámina?
Z = adelante(par, X)[len(par) // 2]   # la activación del cuello de botella


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return abs(np.corrcoef(ra, rb)[0, 1])


P = X @ U
mejor_ae = max(spearman(Z[:, j], t) for j in range(k))
mejor_pca = max(spearman(P[:, j], t) for j in range(k))


def vecinos_conservados(coord, t, kv=10):
    """Fracción de los kv vecinos en t que siguen siendo vecinos en el código."""
    dt = np.abs(t[:, None] - t[None, :])
    np.fill_diagonal(dt, np.inf)
    dz = np.abs(coord[:, None] - coord[None, :])
    np.fill_diagonal(dz, np.inf)
    vt = np.argsort(dt, axis=1)[:, :kv]
    vz = np.argsort(dz, axis=1)[:, :kv]
    return np.mean([len(set(a) & set(b)) / kv for a, b in zip(vt, vz)])


print(f"\nCalidad de la coordenada latente (k = 1):")
print(f"{'':14} {'|Spearman| con t':>18} {'vecinos conservados':>21}")
print(f"{'PCA':14} {mejor_pca:18.4f} {vecinos_conservados(P[:, 0], t):20.1%}")
print(f"{'autoencoder':14} {mejor_ae:18.4f} {vecinos_conservados(Z[:, 0], t):20.1%}")
print("Dos lecturas distintas: PCA ordena mejor GLOBALMENTE (su coordenada crece")
print("con t porque la hélice avanza en línea recta a lo largo del eje), pero el")
print("autoencoder conserva mejor la VECINDAD local, que es lo que hace falta para")
print("reconstruir. De ahí que su error sea tres veces menor.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].semilogy([h_[0] for h_ in hist], [h_[1] for h_ in hist], color="#5b8def")
ax[0].axhline(err_pca, ls="--", c="#ef476f")
ax[0].set_xlabel("época")
ax[0].set_title("error: el AE baja por debajo de PCA")
ax[1].scatter(t, P[:, 0], c=t, cmap="Spectral", s=8)
ax[1].set_title(f"PCA: coordenada frente a t (Spearman {mejor_pca:.2f})")
ax[2].scatter(t, Z[:, 0], c=t, cmap="Spectral", s=8)
ax[2].set_title(f"AE: coordenada frente a t (Spearman {mejor_ae:.2f})")
for a in ax[1:]:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
