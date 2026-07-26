"""
Qué significa "cerca": la misma nube de puntos vista con cuatro reglas distintas.

Comparamos la distancia euclídea, la de Manhattan, la coseno y la de Mahalanobis
sobre unos datos con dos variables muy correlacionadas, y comprobamos que el
vecino más próximo de un punto CAMBIA según la regla que uses.

Ejecútalo con:  python code/02-geometria-y-dimensionalidad/distancias.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

# --- Datos: una nube alargada (altura y peso, muy correlacionados) ---
n = 400
A = np.array([[1.0, 0.0], [1.8, 0.55]])       # matriz que estira y rota la nube
X = rng.normal(size=(n, 2)) @ A.T
X -= X.mean(axis=0)
S = np.cov(X, rowvar=False)
print("Matriz de covarianza:\n", np.round(S, 3))
print(f"Correlación entre las dos columnas: {S[0,1]/np.sqrt(S[0,0]*S[1,1]):.3f}")

# --- Las cuatro reglas, cada una como función de (punto, conjunto) ---
S_inv = np.linalg.inv(S)

def d_euclidea(q, X):
    return np.sqrt(((X - q) ** 2).sum(axis=1))

def d_manhattan(q, X):
    return np.abs(X - q).sum(axis=1)

def d_coseno(q, X):
    cos = (X @ q) / (np.linalg.norm(X, axis=1) * np.linalg.norm(q) + 1e-12)
    return 1.0 - cos

def d_mahalanobis(q, X):
    D = X - q
    return np.sqrt(np.einsum("ij,jk,ik->i", D, S_inv, D))

reglas = [("euclídea", d_euclidea), ("manhattan", d_manhattan),
          ("coseno", d_coseno), ("mahalanobis", d_mahalanobis)]

# --- ¿Quién es el vecino más próximo del punto 0 según cada regla? ---
q = X[0]
print(f"\nPunto de consulta: ({q[0]:+.2f}, {q[1]:+.2f})")
vecinos = {}
for nombre, d in reglas:
    dist = d(q, X)
    dist[0] = np.inf                                  # él mismo no cuenta
    orden = np.argsort(dist)[:5]
    vecinos[nombre] = set(orden)
    print(f"  {nombre:12s} -> 5 vecinos más próximos: {orden}")

base = vecinos["euclídea"]
for nombre in ["manhattan", "coseno", "mahalanobis"]:
    comunes = len(base & vecinos[nombre])
    print(f"  coincidencias euclídea/{nombre}: {comunes}/5")

# --- Mapa de las bolas unidad de cada regla alrededor del punto de consulta ---
gx, gy = np.meshgrid(np.linspace(-6, 6, 260), np.linspace(-6, 6, 260))
G = np.column_stack([gx.ravel(), gy.ravel()])

fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
for a, (nombre, d) in zip(ax, reglas):
    Z = d(q, G).reshape(gx.shape)
    a.scatter(X[:, 0], X[:, 1], s=6, c="#8b93a7", alpha=0.5)
    a.contour(gx, gy, Z, levels=8, cmap="viridis", linewidths=1.2)
    a.plot(q[0], q[1], "r*", ms=15)
    a.set_title(f"«cerca» según {nombre}")
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
