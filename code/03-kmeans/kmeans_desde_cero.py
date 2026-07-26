"""
k-means desde cero, con inicialización k-means++ y traza de la inercia.

El objetivo es ver dos cosas con los ojos:
  - que la inercia NO SUBE en ningún medio paso (asignar y recolocar), y que
    baja siempre que algo se mueve;
  - que el algoritmo se para solo, en pocas iteraciones, porque solo hay un
    número finito de particiones posibles.

Ejecútalo con:  python code/03-kmeans/kmeans_desde_cero.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- Datos: cuatro grupos esféricos, el caso favorable para k-means ---
centros_reales = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.5], [5.5, 0.5]])
X = np.vstack([c + rng.normal(scale=1.0, size=(90, 2)) for c in centros_reales])
X = X[rng.permutation(len(X))]
k = 4


def dist2(X, C):
    """Distancias al cuadrado de cada punto a cada centro, sin bucles."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def inercia(X, C, z):
    return ((X - C[z]) ** 2).sum()


def init_pp(X, k, rng):
    """k-means++: el primer centro al azar; cada siguiente, con probabilidad
    proporcional a la distancia al cuadrado al centro más próximo ya elegido."""
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = dist2(X, np.array(C)).min(axis=1)
        p = d2 / d2.sum()
        C.append(X[rng.choice(len(X), p=p)])
    return np.array(C)


def lloyd(X, C, iters=20):
    """Descenso por bloques de coordenadas: asignar, recolocar, repetir."""
    historia = []
    z = dist2(X, C).argmin(axis=1)
    J0 = inercia(X, C, z)
    historia.append(("inicial", J0, J0, C.copy(), z.copy()))
    for t in range(iters):
        # PASO 1 (asignación): con C fijo, la z óptima es el centro más próximo
        z = dist2(X, C).argmin(axis=1)
        J_asig = inercia(X, C, z)
        # PASO 2 (actualización): con z fijo, la C óptima es la media del grupo
        C_nuevo = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                            for j in range(k)])
        J_act = inercia(X, C_nuevo, z)
        historia.append((f"iter {t+1}", J_asig, J_act, C_nuevo.copy(), z.copy()))
        if np.allclose(C_nuevo, C):
            print(f"Convergió en la iteración {t+1}: los centros ya no se mueven.")
            C = C_nuevo
            break
        C = C_nuevo
    return C, z, historia


C0 = init_pp(X, k, rng)
C, z, historia = lloyd(X, C0)

print(f"\n{'paso':>10} {'tras asignar':>14} {'tras recolocar':>16}")
for nombre, J_a, J_b, _, _ in historia:
    print(f"{nombre:>10} {J_a:14.3f} {J_b:16.3f}")
print(f"\nInercia final: {inercia(X, C, z):.3f}")
print(f"Tamaños de los grupos: {np.bincount(z, minlength=k)}")

# --- Identidad de la varianza: total = intra + inter ---
mu = X.mean(axis=0)
total = ((X - mu) ** 2).sum()
intra = inercia(X, C, z)
inter = sum(np.sum(z == j) * ((C[j] - mu) ** 2).sum() for j in range(k))
print(f"\nDispersión total  : {total:10.3f}")
print(f"  intra (inercia) : {intra:10.3f}")
print(f"  inter (entre)   : {inter:10.3f}")
print(f"  intra + inter   : {intra + inter:10.3f}  <- coincide con el total")

# --- Dibujo: tres instantáneas y la curva de la inercia ---
fig, ax = plt.subplots(1, 4, figsize=(15, 3.7))
for a, idx in zip(ax[:3], [0, 1, len(historia) - 1]):
    nombre, _, J, Ct, zt = historia[idx]
    a.scatter(X[:, 0], X[:, 1], c=zt, cmap="viridis", s=14, edgecolors="none")
    a.scatter(Ct[:, 0], Ct[:, 1], marker="X", s=180, c="red", edgecolors="k")
    a.set_title(f"{nombre} · J = {J:.0f}")
    a.set_xticks([])
    a.set_yticks([])
ax[3].plot([h[2] for h in historia], "o-", color="#ef476f")
ax[3].set_xlabel("iteración")
ax[3].set_ylabel("inercia J")
ax[3].set_title("J nunca sube")
plt.tight_layout()
plt.show()
