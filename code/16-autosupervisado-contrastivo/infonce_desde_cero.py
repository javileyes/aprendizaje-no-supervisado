"""
Aprendizaje contrastivo (SimCLR / NT-Xent) en numpy, con gradientes a mano.

La idea: no hay etiquetas, así que las fabricamos. De cada dato se generan DOS
vistas con transformaciones que NO deberían cambiar su significado, y se
entrena la red para que las dos vistas de un mismo dato se parezcan y las de
datos distintos no.

Los datos son señales caracterizadas por su FRECUENCIA (6 clases). Las
transformaciones cambian la fase, la amplitud y añaden ruido: cosas que no
alteran la frecuencia. Al declarar esas invariancias le estamos diciendo a la
red qué es irrelevante, y eso es toda la supervisión que necesita.

Se evalúa con una SONDA LINEAL: una regresión logística sobre la representación
congelada. Es el estándar del campo.

Ejecútalo con:  python code/16-autosupervisado-contrastivo/infonce_desde_cero.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
T, N_CLASES, N = 64, 6, 1600
FRECS = np.array([2.0, 2.6, 3.2, 4.0, 5.0, 6.2])
t_grid = np.linspace(0, 2 * np.pi, T)


def genera(n, r):
    y = r.integers(0, N_CLASES, n)
    fase = r.uniform(0, 2 * np.pi, n)
    amp = r.uniform(0.5, 2.0, n)
    X = amp[:, None] * np.sin(FRECS[y][:, None] * t_grid[None, :] + fase[:, None])
    X += 0.45 * r.normal(size=(n, T))
    return X, y


X, y = genera(N, rng)
X_te, y_te = genera(400, np.random.default_rng(5))
print(f"{N} señales de longitud {T}; {N_CLASES} frecuencias distintas y próximas.")
print("La fase y la amplitud son ruido: NO definen la clase.")


def aumenta(X, r):
    """Dos vistas: desplazamiento circular (fase), reescalado y ruido."""
    n = len(X)
    despl = r.integers(0, T, n)
    idx = (np.arange(T)[None, :] + despl[:, None]) % T
    V = np.take_along_axis(X, idx, axis=1)
    V = V * r.uniform(0.6, 1.6, (n, 1))
    return V + 0.35 * r.normal(size=V.shape)


CAPAS = [T, 96, 64, 16]


def init(semilla=0):
    r = np.random.default_rng(semilla)
    return [[r.normal(scale=np.sqrt(2.0 / (a + b)), size=(a, b)), np.zeros(b)]
            for a, b in zip(CAPAS[:-1], CAPAS[1:])]


def adelante(par, X):
    acts, A = [X], X
    for i, (W, b) in enumerate(par):
        Zl = A @ W + b
        A = Zl if i == len(par) - 1 else np.tanh(Zl)
        acts.append(A)
    return acts


def normaliza(H):
    n = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
    return H / n, n


def nt_xent(H, tau=0.2):
    """Pérdida InfoNCE simétrica. H tiene 2b filas: [vista A ; vista B]."""
    Z, norm = normaliza(H)
    m = len(Z)
    b = m // 2                                    # tamaño del LOTE, no del conjunto
    S = (Z @ Z.T) / tau
    np.fill_diagonal(S, -1e9)                     # nadie es su propio positivo
    pos = np.concatenate([np.arange(b, m), np.arange(0, b)])
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    L = -np.log(P[np.arange(m), pos] + 1e-12).mean()
    # ---- gradiente ----
    dS = P.copy()
    dS[np.arange(m), pos] -= 1.0
    dS /= m
    np.fill_diagonal(dS, 0.0)
    dZ = ((dS + dS.T) @ Z) / tau                  # S_ij aparece en la fila i y en la j
    dH = (dZ - Z * (Z * dZ).sum(axis=1, keepdims=True)) / norm   # por la normalización
    return L, dH, Z


def atras(par, acts, dA):
    grads = []
    for i in range(len(par) - 1, -1, -1):
        if i != len(par) - 1:
            dA = dA * (1.0 - acts[i + 1] ** 2)
        grads.append([acts[i].T @ dA, dA.sum(axis=0)])
        dA = dA @ par[i][0].T
    return grads[::-1]


# ---------------- Verificación del gradiente ----------------
par = init(0)
Xa = aumenta(X[:16], rng)
Xb = aumenta(X[:16], rng)
V = np.vstack([Xa, Xb])


def perdida(par, V):
    return nt_xent(adelante(par, V)[-1])[0]


acts = adelante(par, V)
L0, dH, _ = nt_xent(acts[-1])
g = atras(par, acts, dH)
err = []
for capa in range(len(par)):
    W = par[capa][0]
    for _ in range(4):
        i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
        o = W[i, j]
        W[i, j] = o + 1e-6
        Lp = perdida(par, V)
        W[i, j] = o - 1e-6
        Lm = perdida(par, V)
        W[i, j] = o
        num = (Lp - Lm) / 2e-6
        err.append(abs(num - g[capa][0][i, j]) / max(abs(num), 1e-12))
print(f"\nGradiente analítico vs. numérico: error relativo máximo {max(err):.2e}")


# ---------------- Entrenamiento contrastivo ----------------
def entrena_contrastivo(tau=0.2, epocas=120, lote=128, lr=2e-3, semilla=0):
    par = init(semilla)
    r = np.random.default_rng(semilla + 3)
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in par]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in par]
    paso, hist = 0, []
    for ep in range(epocas):
        orden = r.permutation(N)
        for i in range(0, N, lote):
            xb = X[orden[i:i + lote]]
            V = np.vstack([aumenta(xb, r), aumenta(xb, r)])
            acts = adelante(par, V)
            L, dH, _ = nt_xent(acts[-1], tau)
            g = atras(par, acts, dH)
            paso += 1
            for a in range(len(par)):
                for b_ in range(2):
                    m[a][b_] = 0.9 * m[a][b_] + 0.1 * g[a][b_]
                    v[a][b_] = 0.999 * v[a][b_] + 0.001 * g[a][b_] ** 2
                    par[a][b_] -= lr * (m[a][b_] / (1 - 0.9 ** paso)) / (
                        np.sqrt(v[a][b_] / (1 - 0.999 ** paso)) + 1e-8)
        if ep % 20 == 0 or ep == epocas - 1:
            hist.append((ep, L))
    return par, hist


def sonda_lineal(Ztr, ytr, Zte, yte, iters=800, lr=0.5):
    """Regresión logística multiclase: mide cuánta información LINEAL hay."""
    Ztr = np.column_stack([Ztr, np.ones(len(Ztr))])
    Zte = np.column_stack([Zte, np.ones(len(Zte))])
    W = np.zeros((Ztr.shape[1], N_CLASES))
    Y = np.eye(N_CLASES)[ytr]
    for _ in range(iters):
        S = Ztr @ W
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        W -= lr * Ztr.T @ (P - Y) / len(Ztr)
    return ((Zte @ W).argmax(axis=1) == yte).mean()


print("\nEntrenando el modelo contrastivo...")
par_c, hist = entrena_contrastivo()
print(f"{'época':>7} {'InfoNCE':>10}")
for ep, L in hist:
    print(f"{ep:7d} {L:10.4f}")
print(f"  (la cifra es la del último lote de la época, que tiene 64 datos;")
print(f"   log(2*64-1) = {np.log(127):.4f} sería no haber aprendido nada)")

Z_tr = normaliza(adelante(par_c, X)[-1])[0]
Z_te = normaliza(adelante(par_c, X_te)[-1])[0]

# --- Comparación con otras representaciones ---
mu = X.mean(axis=0)
S = ((X - mu).T @ (X - mu)) / (N - 1)
val, vec = np.linalg.eigh(S)
U = vec[:, ::-1][:, :16]

print(f"\n{'representación':32} {'dimensión':>10} {'sonda lineal':>14}")
print(f"{'señal en bruto':32} {T:10d} "
      f"{sonda_lineal(X, y, X_te, y_te):13.1%}")
print(f"{'PCA (16 componentes)':32} {16:10d} "
      f"{sonda_lineal((X - mu) @ U, y, (X_te - mu) @ U, y_te):13.1%}")
print(f"{'contrastivo (InfoNCE)':32} {16:10d} "
      f"{sonda_lineal(Z_tr, y, Z_te, y_te):13.1%}")

# --- ¿Ha aprendido a IGNORAR la fase y la amplitud? ---
Xa = aumenta(X_te, np.random.default_rng(11))
Xb = aumenta(X_te, np.random.default_rng(12))
Za = normaliza(adelante(par_c, Xa)[-1])[0]
Zb = normaliza(adelante(par_c, Xb)[-1])[0]
print(f"\nInvariancia frente a las transformaciones (coseno entre dos vistas):")
print(f"  representación contrastiva : {(Za * Zb).sum(axis=1).mean():.4f}")
Pa = (Xa - mu) @ U
Pb = (Xb - mu) @ U
Pa /= np.linalg.norm(Pa, axis=1, keepdims=True)
Pb /= np.linalg.norm(Pb, axis=1, keepdims=True)
print(f"  PCA                        : {(Pa * Pb).sum(axis=1).mean():.4f}")
print("  1,0 significa 'las dos vistas del mismo dato son indistinguibles'.")

# --- El papel de la temperatura ---
print(f"\nEfecto de la temperatura tau:")
print(f"{'tau':>6} {'sonda lineal':>14}")
for tau in [0.05, 0.2, 0.5, 2.0]:
    p, _ = entrena_contrastivo(tau=tau, epocas=60)
    Ztr = normaliza(adelante(p, X)[-1])[0]
    Zte = normaliza(adelante(p, X_te)[-1])[0]
    print(f"{tau:6.2f} {sonda_lineal(Ztr, y, Zte, y_te):13.1%}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for c in range(N_CLASES):
    ax[0].plot(t_grid, X[y == c][0], lw=1)
ax[0].set_title("una señal de cada clase")
ax[0].set_xticks([])
P2 = (X_te - mu) @ U
ax[1].scatter(P2[:, 0], P2[:, 1], c=y_te, cmap="viridis", s=10, edgecolors="none")
ax[1].set_title("PCA (2 primeras)")
ax[2].scatter(Z_te[:, 0], Z_te[:, 1], c=y_te, cmap="viridis", s=10, edgecolors="none")
ax[2].set_title("contrastivo (2 primeras)")
for a in ax[1:]:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
