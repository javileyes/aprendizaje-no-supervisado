"""
Panorama del aprendizaje no supervisado: las tres preguntas que se le pueden
hacer a una nube de puntos sin etiquetas.

  1) ¿Dónde está la masa?      -> estimación de densidad
  2) ¿En cuántos grupos cae?   -> clústering
  3) ¿Cuántas direcciones hay? -> aprendizaje de representación

Ejecútalo con:  python code/01-aprender-sin-etiquetas/panorama.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# --- Los datos: tres grupos, pero NADIE nos dice que son tres ni cuáles ---
centros = np.array([[-2.5, 0.0], [2.0, 1.5], [0.5, -2.5]])
n_por_grupo = 120
X = np.vstack([c + rng.normal(scale=0.75, size=(n_por_grupo, 2)) for c in centros])
X = X[rng.permutation(len(X))]          # barajamos: perdemos todo rastro del origen
print(f"Datos: {X.shape[0]} puntos en {X.shape[1]} dimensiones, sin etiquetas.")

# --- Pregunta 1: ¿dónde está la masa? Densidad por histograma 2D ---
H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=24)
densidad = H / (H.sum() * (xe[1] - xe[0]) * (ye[1] - ye[0]))
print(f"Densidad máxima estimada: {densidad.max():.4f} puntos por unidad de área.")

# --- Pregunta 2: ¿en cuántos grupos cae? k-means en 12 líneas ---
def kmeans(X, k, semilla=0, iters=50):
    r = np.random.default_rng(semilla)
    C = X[r.choice(len(X), k, replace=False)]          # centros iniciales al azar
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        z = d2.argmin(axis=1)                          # asignar cada punto al centro más cercano
        C_nuevo = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                            for j in range(k)])
        if np.allclose(C_nuevo, C):
            break
        C = C_nuevo
    inercia = ((X - C[z]) ** 2).sum()
    return z, C, inercia

z, C, inercia = kmeans(X, k=3, semilla=0)
print(f"k-means con k=3 -> tamaños {np.bincount(z)}, inercia {inercia:.2f}")

# --- Pregunta 3: ¿cuántas direcciones hacen falta? La primera componente ---
Xc = X - X.mean(axis=0)
S = (Xc.T @ Xc) / (len(Xc) - 1)                        # matriz de covarianza 2x2
valores, vectores = np.linalg.eigh(S)
orden = valores.argsort()[::-1]
valores, vectores = valores[orden], vectores[:, orden]
u1 = vectores[:, 0]
proy = Xc @ u1                                         # coordenada sobre la 1ª componente
frac = valores[0] / valores.sum()
print(f"Varianza explicada por la 1ª componente: {frac:.1%}")

# --- Las cuatro miradas, una al lado de la otra ---
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))

ax[0].scatter(X[:, 0], X[:, 1], s=12, c="#8b93a7")
ax[0].set_title("1. Lo que ve la máquina")

ax[1].imshow(densidad.T, origin="lower", aspect="auto", cmap="magma",
             extent=[xe[0], xe[-1], ye[0], ye[-1]])
ax[1].set_title("2. Densidad: ¿dónde hay masa?")

ax[2].scatter(X[:, 0], X[:, 1], s=12, c=z, cmap="viridis")
ax[2].scatter(C[:, 0], C[:, 1], s=200, marker="X", c="red", edgecolors="k")
ax[2].set_title("3. Clústering: ¿qué grupos hay?")

ax[3].hist(proy, bins=30, color="#5b8def")
ax[3].set_title(f"4. 1ª componente ({frac:.0%} de varianza)")

for a in ax[:3]:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
