"""
Regularización de consistencia y pseudo-etiquetas al estilo FixMatch.

Receta (Sohn et al., 2020):
  1. Con los pocos datos etiquetados, pérdida supervisada normal.
  2. Con los NO etiquetados: se predice sobre una vista DÉBILmente aumentada;
     si la confianza supera un umbral, esa predicción se convierte en
     pseudo-etiqueta y se usa para supervisar una vista FUERTEmente aumentada.

El umbral es lo que separa FixMatch de un autoentrenamiento ingenuo, y lo
medimos: sin él, el modelo se confirma en sus propios errores.

Datos: las señales del capítulo 16 (seis frecuencias). 1500 sin etiquetar y un
puñado etiquetadas.

Ejecútalo con:  python code/17-semisupervisado/consistencia_fixmatch.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
T, C, N_U = 64, 6, 1500
FRECS = np.array([2.0, 2.6, 3.2, 4.0, 5.0, 6.2])
grid = np.linspace(0, 2 * np.pi, T)


def genera(n, r):
    y = r.integers(0, C, n)
    X = (r.uniform(0.5, 2.0, (n, 1))
         * np.sin(FRECS[y][:, None] * grid[None, :] + r.uniform(0, 2 * np.pi, (n, 1))))
    return X + 0.45 * r.normal(size=(n, T)), y


X_u, y_u = genera(N_U, rng)
X_te, y_te = genera(600, np.random.default_rng(5))


def aumenta(X, r, fuerte):
    n = len(X)
    d = r.integers(0, T, n)
    idx = (np.arange(T)[None, :] + d[:, None]) % T
    V = np.take_along_axis(X, idx, axis=1)
    if fuerte:
        V = V * r.uniform(0.4, 2.0, (n, 1)) + 0.45 * r.normal(size=V.shape)
        mascara = r.random(V.shape) < 0.2          # borrado aleatorio de muestras
        V = np.where(mascara, 0.0, V)
    else:
        V = V * r.uniform(0.85, 1.15, (n, 1)) + 0.12 * r.normal(size=V.shape)
    return V


CAPAS = [T, 96, 64, C]


def init(s=0):
    r = np.random.default_rng(s)
    return [[r.normal(scale=np.sqrt(2.0 / (a + b)), size=(a, b)), np.zeros(b)]
            for a, b in zip(CAPAS[:-1], CAPAS[1:])]


def adelante(par, X):
    acts, A = [X], X
    for i, (W, b) in enumerate(par):
        Z = A @ W + b
        A = Z if i == len(par) - 1 else np.tanh(Z)
        acts.append(A)
    return acts


def softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(axis=1, keepdims=True)


def atras(par, acts, dlogits):
    grads, dA = [], dlogits
    for i in range(len(par) - 1, -1, -1):
        if i != len(par) - 1:
            dA = dA * (1.0 - acts[i + 1] ** 2)
        grads.append([acts[i].T @ dA, dA.sum(axis=0)])
        dA = dA @ par[i][0].T
    return grads[::-1]


def suma_grads(a, b, peso=1.0):
    return [[a[i][j] + peso * b[i][j] for j in range(2)] for i in range(len(a))]


def entrena(X_l, y_l, usar_no_etiquetados, umbral, pasos=1500, lote=64,
            lr=2e-3, lam=1.0, semilla=0):
    par = init(semilla)
    r = np.random.default_rng(semilla + 11)
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in par]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in par]
    usados, aciertos_pseudo = [], []
    for paso in range(1, pasos + 1):
        idx = r.integers(0, len(X_l), min(lote, len(X_l)))
        xb = aumenta(X_l[idx], r, fuerte=False)
        acts = adelante(par, xb)
        P = softmax(acts[-1])
        dl = (P - np.eye(C)[y_l[idx]]) / len(xb)
        g = atras(par, acts, dl)

        if usar_no_etiquetados:
            iu = r.integers(0, N_U, lote)
            debil = aumenta(X_u[iu], r, fuerte=False)
            Pw = softmax(adelante(par, debil)[-1])
            conf = Pw.max(axis=1)
            pseudo = Pw.argmax(axis=1)
            sel = conf >= umbral
            usados.append(sel.mean())
            if sel.any():
                aciertos_pseudo.append((pseudo[sel] == y_u[iu][sel]).mean())
                fuerte = aumenta(X_u[iu][sel], r, fuerte=True)
                acts_f = adelante(par, fuerte)
                Pf = softmax(acts_f[-1])
                dlf = (Pf - np.eye(C)[pseudo[sel]]) / sel.sum()
                g = suma_grads(g, atras(par, acts_f, dlf), lam)

        for i in range(len(par)):
            for j in range(2):
                m[i][j] = 0.9 * m[i][j] + 0.1 * g[i][j]
                v[i][j] = 0.999 * v[i][j] + 0.001 * g[i][j] ** 2
                par[i][j] -= lr * (m[i][j] / (1 - 0.9 ** paso)) / (
                    np.sqrt(v[i][j] / (1 - 0.999 ** paso)) + 1e-8)
    acc = (adelante(par, X_te)[-1].argmax(axis=1) == y_te).mean()
    return (acc, np.mean(usados) if usados else 0.0,
            np.mean(aciertos_pseudo) if aciertos_pseudo else 0.0)


print(f"{N_U} señales sin etiquetar, {len(X_te)} de prueba, {C} clases "
      f"(azar = {1/C:.1%}).\n")
print(f"{'etiquetas':>10} {'solo supervisado':>18} {'FixMatch (u=0,95)':>19} "
      f"{'mejora':>9} {'% no etq. usados':>18}")
resumen = []
for n_l in [6, 12, 30, 60]:
    r = np.random.default_rng(3)
    sel = np.concatenate([r.choice(np.flatnonzero(y_u == c), n_l // C, replace=False)
                          for c in range(C)])
    X_l, y_l = X_u[sel], y_u[sel]
    a_sup = np.mean([entrena(X_l, y_l, False, 0.0, semilla=s)[0] for s in range(2)])
    res = [entrena(X_l, y_l, True, 0.95, semilla=s) for s in range(2)]
    a_fix = np.mean([x[0] for x in res])
    uso = np.mean([x[1] for x in res])
    resumen.append((n_l, a_sup, a_fix))
    print(f"{n_l:10d} {a_sup:17.1%} {a_fix:18.1%} {a_fix - a_sup:+8.1%} {uso:17.1%}")

# --- El papel del umbral de confianza ---
r = np.random.default_rng(3)
sel = np.concatenate([r.choice(np.flatnonzero(y_u == c), 1, replace=False)
                      for c in range(C)])
X_l, y_l = X_u[sel], y_u[sel]
print(f"\nCon solo {len(X_l)} etiquetas, ¿cuánto importa el umbral de confianza?")
print(f"{'umbral':>8} {'acierto final':>15} {'% no etq. usados':>18} "
      f"{'pseudo-etq. correctas':>23}")
for u in [0.0, 0.5, 0.8, 0.95, 0.99]:
    res = [entrena(X_l, y_l, True, u, semilla=s) for s in range(2)]
    print(f"{u:8.2f} {np.mean([x[0] for x in res]):14.1%} "
          f"{np.mean([x[1] for x in res]):17.1%} {np.mean([x[2] for x in res]):22.1%}")
print("El umbral cambia la CANTIDAD por la CALIDAD de las pseudo-etiquetas: al")
print("subirlo se usan muchas menos, pero son más fiables. Cuál gana depende del")
print("problema, y aquí -con pseudo-etiquetas ya bastante buenas- gana la cantidad.")
print("No des por hecho que un umbral alto es siempre mejor: mídelo.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ns = [x[0] for x in resumen]
ax[0].plot(ns, [x[1] for x in resumen], "o-", label="solo supervisado")
ax[0].plot(ns, [x[2] for x in resumen], "s-", label="FixMatch")
ax[0].axhline(1 / C, ls=":", c="k", label="azar")
ax[0].set_xlabel("nº de etiquetas")
ax[0].set_ylabel("acierto en prueba")
ax[0].legend()
ax[0].set_title("los datos sin etiquetar valen más cuanto menos etiquetas hay")
r2 = np.random.default_rng(9)
ax[1].plot(grid, X_u[0], lw=1, label="original")
ax[1].plot(grid, aumenta(X_u[:1], r2, False)[0], lw=1, label="aumentación débil")
ax[1].plot(grid, aumenta(X_u[:1], r2, True)[0], lw=1, label="aumentación fuerte")
ax[1].legend(fontsize=8)
ax[1].set_xticks([])
ax[1].set_title("las dos aumentaciones")
plt.tight_layout()
plt.show()
