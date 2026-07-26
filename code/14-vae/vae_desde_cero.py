"""
Autoencoder variacional en numpy puro, con los gradientes derivados a mano.

Arquitectura:  x -> tanh(64) -> (mu, log sigma^2) -> z -> tanh(64) -> sigmoide
Pérdida:       -ELBO = reconstrucción (Bernoulli) + beta * KL(q(z|x) || N(0,I))

Se comprueban tres cosas:
  1. que el gradiente analítico coincide con el numérico;
  2. que el ELBO sube y que se puede MUESTREAR del prior y obtener datos
     plausibles (un autoencoder normal no puede);
  3. el COLAPSO POSTERIOR: si beta es grande, la KL se va a cero, el latente
     deja de llevar información y el decodificador ignora z.

Ejecútalo con:  python code/14-vae/vae_desde_cero.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
LADO, K, H = 8, 2, 64
D = LADO * LADO


def genera(n, r):
    """Imágenes binarias: 1 o 2 trazos rectos (filas o columnas) por imagen."""
    X = np.zeros((n, LADO, LADO), dtype=float)
    for i in range(n):
        for _ in range(r.integers(1, 3)):
            if r.random() < 0.5:
                f = r.integers(0, LADO)
                X[i, f, :] = 1.0
            else:
                c = r.integers(0, LADO)
                X[i, :, c] = 1.0
    return X.reshape(n, D)


X = genera(3000, rng)
X_te = genera(600, np.random.default_rng(99))
print(f"{len(X)} imágenes de {LADO}x{LADO}; píxeles encendidos: {X.mean():.1%}")


def sigmoide(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))


def init(semilla=0):
    r = np.random.default_rng(semilla)
    def W(a, b):
        return r.normal(scale=np.sqrt(2.0 / (a + b)), size=(a, b))
    return dict(W1=W(D, H), b1=np.zeros(H), Wm=W(H, K), bm=np.zeros(K),
                Wv=W(H, K), bv=np.zeros(K), W2=W(K, H), b2=np.zeros(H),
                W3=W(H, D), b3=np.zeros(D))


def adelante(p, x, eps):
    h1 = np.tanh(x @ p["W1"] + p["b1"])
    mu = h1 @ p["Wm"] + p["bm"]
    lv = np.clip(h1 @ p["Wv"] + p["bv"], -8, 8)      # log sigma^2
    sd = np.exp(0.5 * lv)
    z = mu + sd * eps                                # truco de reparametrización
    h2 = np.tanh(z @ p["W2"] + p["b2"])
    logits = h2 @ p["W3"] + p["b3"]
    return h1, mu, lv, sd, z, h2, logits


def perdida_y_grad(p, x, eps, beta=1.0):
    n = len(x)
    h1, mu, lv, sd, z, h2, logits = adelante(p, x, eps)
    xg = sigmoide(logits)
    # --- reconstrucción: -log p(x|z) con verosimilitud Bernoulli ---
    rec = np.maximum(logits, 0) - logits * x + np.log1p(np.exp(-np.abs(logits)))
    rec = rec.sum(axis=1)
    # --- KL(q||p) en forma cerrada para gaussianas diagonales ---
    kl = -0.5 * (1 + lv - mu ** 2 - np.exp(lv)).sum(axis=1)
    L = (rec + beta * kl).mean()

    # ---------------- retropropagación ----------------
    g = {}
    dlogits = (xg - x) / n                            # d(rec)/d(logits)
    g["W3"] = h2.T @ dlogits
    g["b3"] = dlogits.sum(axis=0)
    dh2 = dlogits @ p["W3"].T
    dpre2 = dh2 * (1 - h2 ** 2)
    g["W2"] = z.T @ dpre2
    g["b2"] = dpre2.sum(axis=0)
    dz = dpre2 @ p["W2"].T
    # gradientes que llegan a mu y a lv: por el decodificador y por la KL
    dmu = dz + beta * mu / n
    dlv = dz * 0.5 * sd * eps + beta * 0.5 * (np.exp(lv) - 1) / n
    g["Wm"] = h1.T @ dmu
    g["bm"] = dmu.sum(axis=0)
    g["Wv"] = h1.T @ dlv
    g["bv"] = dlv.sum(axis=0)
    dh1 = dmu @ p["Wm"].T + dlv @ p["Wv"].T
    dpre1 = dh1 * (1 - h1 ** 2)
    g["W1"] = x.T @ dpre1
    g["b1"] = dpre1.sum(axis=0)
    return L, g, rec.mean(), kl.mean()


# ---------------- 1) Verificación del gradiente ----------------
p = init(0)
xb = X[:8]
epsb = rng.normal(size=(8, K))
L0, g0, _, _ = perdida_y_grad(p, xb, epsb)
err = []
for clave in ["W1", "Wm", "Wv", "W2", "W3"]:
    A = p[clave]
    for _ in range(3):
        i, j = rng.integers(A.shape[0]), rng.integers(A.shape[1])
        o = A[i, j]
        A[i, j] = o + 1e-6
        Lp = perdida_y_grad(p, xb, epsb)[0]
        A[i, j] = o - 1e-6
        Lm = perdida_y_grad(p, xb, epsb)[0]
        A[i, j] = o
        num = (Lp - Lm) / 2e-6
        err.append(abs(num - g0[clave][i, j]) / max(abs(num), 1e-12))
print(f"\n1) Gradiente analítico vs. numérico: error relativo máximo {max(err):.2e}")
print("   (por debajo de 1e-5 la derivación es correcta)")


def entrena(beta, epocas=60, lote=100, lr=2e-3, semilla=0):
    p = init(semilla)
    r = np.random.default_rng(semilla + 1)
    m = {k_: np.zeros_like(v) for k_, v in p.items()}
    v = {k_: np.zeros_like(v_) for k_, v_ in p.items()}
    paso, hist = 0, []
    for ep in range(epocas):
        orden = r.permutation(len(X))
        for i in range(0, len(X), lote):
            xb = X[orden[i:i + lote]]
            L, g, _, _ = perdida_y_grad(p, xb, r.normal(size=(len(xb), K)), beta)
            paso += 1
            for k_ in p:
                m[k_] = 0.9 * m[k_] + 0.1 * g[k_]
                v[k_] = 0.999 * v[k_] + 0.001 * g[k_] ** 2
                p[k_] -= lr * (m[k_] / (1 - 0.9 ** paso)) / (
                    np.sqrt(v[k_] / (1 - 0.999 ** paso)) + 1e-8)
        if ep % 10 == 0 or ep == epocas - 1:
            _, _, rec, kl = perdida_y_grad(p, X_te, r.normal(size=(len(X_te), K)), beta)
            hist.append((ep, rec, kl, rec + kl))
    return p, hist


# ---------------- 2) Entrenamiento con beta = 1 ----------------
print("\n2) Entrenamiento con beta = 1")
print(f"{'época':>7} {'reconstr.':>11} {'KL':>9} {'-ELBO':>10}")
p1, hist1 = entrena(1.0)
for ep, rec, kl, tot in hist1:
    print(f"{ep:7d} {rec:11.4f} {kl:9.4f} {tot:10.4f}")

# ¿Sirve el prior para generar? Muestreamos z ~ N(0,I) y decodificamos.
r = np.random.default_rng(7)
z_prior = r.normal(size=(600, K))
h2 = np.tanh(z_prior @ p1["W2"] + p1["b2"])
muestras = sigmoide(h2 @ p1["W3"] + p1["b3"])
print(f"\n   Muestreando z ~ N(0,I) y decodificando:")
print(f"   fracción media de píxeles encendidos: generado {muestras.mean():.3f} "
      f"| datos reales {X.mean():.3f}")
# Un trazo recto hace que la varianza POR FILAS sea alta. Lo medimos.
def estructura(A):
    im = A.reshape(-1, LADO, LADO)
    return im.mean(axis=2).std(axis=1).mean()
print(f"   estructura (dispersión entre filas): generado {estructura(muestras):.3f} "
      f"| real {estructura(X):.3f} | ruido puro {estructura(r.random((600, D))):.3f}")

# ---------------- 3) Colapso posterior ----------------
print("\n3) Colapso posterior: qué pasa al subir beta")
print(f"{'beta':>6} {'reconstr.':>11} {'KL':>9} {'unidades activas':>18}")
for beta in [0.2, 1.0, 4.0, 20.0]:
    pb, hb = entrena(beta, epocas=40)
    h1 = np.tanh(X_te @ pb["W1"] + pb["b1"])
    mu = h1 @ pb["Wm"] + pb["bm"]
    lv = np.clip(h1 @ pb["Wv"] + pb["bv"], -8, 8)
    kl_por_dim = (-0.5 * (1 + lv - mu ** 2 - np.exp(lv))).mean(axis=0)
    activas = int((kl_por_dim > 0.01).sum())
    print(f"{beta:6.2f} {hb[-1][1]:11.4f} {hb[-1][2]:9.4f} {activas:15d}/{K}")
print("   Con beta grande, la KL se hunde: q(z|x) se pega al prior, deja de")
print("   depender de x y el latente no transporta NADA. El decodificador")
print("   responde siempre lo mismo y la reconstrucción se degrada.")

fig, ax = plt.subplots(2, 4, figsize=(13, 6))
for a, i in zip(ax[0], range(4)):
    a.imshow(X_te[i].reshape(LADO, LADO), cmap="magma")
    a.set_title("real", fontsize=10)
for a, i in zip(ax[1], range(4)):
    a.imshow(muestras[i].reshape(LADO, LADO), cmap="magma")
    a.set_title("generado del prior", fontsize=10)
for fila in ax:
    for a in fila:
        a.set_xticks([])
        a.set_yticks([])
plt.tight_layout()
plt.show()
