"""
t-SNE desde cero: afinidades gaussianas arriba, t de Student abajo.

Implementamos el algoritmo completo:
  - busqueda binaria de sigma_i para que cada punto tenga la PERPLEJIDAD pedida;
  - p_ij simetrizadas; q_ij con t de Student de 1 grado de libertad;
  - el gradiente exacto de KL(P||Q), con momento y "exageración temprana".

Y luego medimos las dos advertencias que todo el mundo repite y casi nadie
comprueba: que las DISTANCIAS entre grupos en el dibujo no significan nada, y
que los TAMAÑOS de los grupos tampoco.

Ejecútalo con:  python code/12-manifold-tsne-umap/tsne_desde_cero.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# --- Tres grupos DELIBERADAMENTE desiguales, en 30 dimensiones ---
#   A: 120 puntos, muy compacto      (sd 0,3)
#   B: 120 puntos, muy disperso      (sd 2,0)
#   C: 120 puntos, compacto y LEJÍSIMOS de los otros dos
d = 30
A = rng.normal(scale=0.3, size=(120, d)) + np.r_[0.0, np.zeros(d - 1)]
B = rng.normal(scale=2.0, size=(120, d)) + np.r_[10.0, np.zeros(d - 1)]
C = rng.normal(scale=0.3, size=(120, d)) + np.r_[60.0, np.zeros(d - 1)]
X = np.vstack([A, B, C])
y = np.repeat([0, 1, 2], 120)
n = len(X)


def dist2(X):
    sq = (X ** 2).sum(axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)


D2 = dist2(X)
print(f"{n} puntos en {d} dimensiones, tres grupos.")
print("Distancias medias REALES entre centros:")
cen = np.array([A.mean(0), B.mean(0), C.mean(0)])
for i, j in [(0, 1), (1, 2), (0, 2)]:
    print(f"  centro {i} - centro {j}: {np.linalg.norm(cen[i] - cen[j]):7.2f}")
print(f"Radio medio de cada grupo: " +
      " ".join(f"{np.linalg.norm(G - G.mean(0), axis=1).mean():.2f}"
               for G in (A, B, C)))


def p_condicionales(D2, perplejidad, tol=1e-5, iters=60):
    """Para cada i, busca sigma_i tal que 2^H(P_i) = perplejidad (bisección)."""
    n = len(D2)
    P = np.zeros((n, n))
    objetivo = np.log(perplejidad)
    for i in range(n):
        lo, hi = 1e-20, 1e20
        beta = 1.0                                   # beta = 1/(2 sigma^2)
        Di = np.delete(D2[i], i)
        for _ in range(iters):
            Pi = np.exp(-Di * beta)
            suma = Pi.sum()
            if suma <= 0:
                H = 0.0
                Pi = np.ones_like(Di) / len(Di)
            else:
                H = np.log(suma) + beta * (Di * Pi).sum() / suma
                Pi = Pi / suma
            if abs(H - objetivo) < tol:
                break
            if H > objetivo:                         # demasiada entropía: subir beta
                lo = beta
                beta = beta * 2 if hi == 1e20 else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == 1e-20 else (beta + lo) / 2
        P[i, np.arange(n) != i] = Pi
    return P


def tsne(D2, perplejidad=30.0, iters=600, semilla=0, dim=2):
    n = len(D2)
    P = p_condicionales(D2, perplejidad)
    P = (P + P.T) / (2 * n)                          # simetrizar
    P = np.maximum(P, 1e-12)
    r = np.random.default_rng(semilla)
    Y = r.normal(scale=1e-4, size=(n, dim))
    velocidad = np.zeros_like(Y)
    historia = []
    for it in range(iters):
        exagera = 4.0 if it < 100 else 1.0            # exageración temprana
        num = 1.0 / (1.0 + dist2(Y))                  # t de Student, 1 g.l.
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        # gradiente exacto: 4 * sum_j (p_ij - q_ij) (y_i - y_j) (1+||y_i-y_j||^2)^-1
        L = (exagera * P - Q) * num
        grad = 4.0 * (np.diag(L.sum(axis=1)) - L) @ Y
        momento = 0.5 if it < 250 else 0.8
        velocidad = momento * velocidad - 200.0 * grad
        Y = Y + velocidad
        Y -= Y.mean(axis=0)
        if it % 50 == 0 or it == iters - 1:
            # OJO: la KL se mide SIEMPRE con la P de verdad, no con la exagerada.
            # sum(4P·log(4P/Q)) no es una divergencia, porque 4P suma 4 y no 1.
            historia.append((it, (P * np.log(P / Q)).sum()))
    return Y, historia


print("\nAjustando t-SNE (perplejidad 30)...")
Y, hist = tsne(D2, perplejidad=30.0)
print(f"{'iteración':>10} {'KL(P||Q)':>12}")
for it, kl in hist:
    print(f"{it:10d} {kl:12.5f}")

# --- ¿Se conservan las distancias entre grupos? ---
cen_y = np.array([Y[y == j].mean(axis=0) for j in range(3)])
print("\nDistancias entre centros: real (en 30D) frente a la del dibujo")
print(f"{'par':>10} {'real':>10} {'en t-SNE':>11} {'cociente':>10}")
reales, dibujadas = [], []
for i, j in [(0, 1), (1, 2), (0, 2)]:
    r_ = np.linalg.norm(cen[i] - cen[j])
    d_ = np.linalg.norm(cen_y[i] - cen_y[j])
    reales.append(r_)
    dibujadas.append(d_)
    print(f"{i}-{j:>8} {r_:10.2f} {d_:11.2f} {d_ / r_:10.4f}")
print(f"  Si t-SNE conservara distancias, los tres cocientes serían iguales.")
print(f"  Aquí van de {min(np.array(dibujadas)/np.array(reales)):.4f} a "
      f"{max(np.array(dibujadas)/np.array(reales)):.4f}: "
      f"un factor {max(np.array(dibujadas)/np.array(reales))/min(np.array(dibujadas)/np.array(reales)):.1f}.")

# --- ¿Se conservan los tamaños? ---
print("\nRadio medio de cada grupo: real frente al del dibujo")
print(f"{'grupo':>7} {'real':>9} {'en t-SNE':>11}")
for j, G in enumerate((A, B, C)):
    rr = np.linalg.norm(G - G.mean(0), axis=1).mean()
    rd = np.linalg.norm(Y[y == j] - cen_y[j], axis=1).mean()
    print(f"{j:7d} {rr:9.2f} {rd:11.2f}")
r_real = [np.linalg.norm(G - G.mean(0), axis=1).mean() for G in (A, B, C)]
r_dib = [np.linalg.norm(Y[y == j] - cen_y[j], axis=1).mean() for j in range(3)]
print(f"  En los datos, el grupo más ancho lo es {max(r_real)/min(r_real):.1f} veces más "
      f"que el más estrecho.")
print(f"  En el dibujo, esa razón se ha comprimido a {max(r_dib)/min(r_dib):.1f}.")

# --- El efecto de la perplejidad ---
print("\nMisma nube, distintas perplejidades:")
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, perp in zip(ax, [5.0, 15.0, 30.0, 80.0]):
    Yp, _ = tsne(D2, perplejidad=perp, iters=400)
    cen_p = np.array([Yp[y == j].mean(axis=0) for j in range(3)])
    sep = np.linalg.norm(cen_p[0] - cen_p[1]) / np.linalg.norm(cen_p[1] - cen_p[2])
    print(f"  perplejidad {perp:5.1f} -> cociente de separaciones d(0,1)/d(1,2) = {sep:.3f}")
    a.scatter(Yp[:, 0], Yp[:, 1], c=y, cmap="viridis", s=8, edgecolors="none")
    a.set_title(f"perplejidad {perp:.0f}")
    a.set_xticks([])
    a.set_yticks([])
print(f"  El cociente REAL es {reales[0]/reales[1]:.3f}; ninguna perplejidad lo reproduce,")
print("  y el valor que sale depende más de la perplejidad que de los datos.")
plt.tight_layout()
plt.show()
