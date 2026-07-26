"""
Los tres fracasos característicos de k-means, uno al lado del otro.

  1. Grupos alargados (anisótropos).
  2. Grupos de tamaño y densidad muy distintos.
  3. Grupos no convexos (dos lunas).

Y, sobre todo, el DIAGNÓSTICO: en los tres casos medimos la inercia de la
partición VERDADERA y la comparamos con la que encuentra k-means. Si la
verdadera tiene peor inercia, k-means no está fallando al optimizar: está
optimizando bien un objetivo que prefiere la respuesta equivocada.

Medimos también, como eje aparte, cuánto de la partición verdadera se puede
siquiera EXPRESAR con dos centroides (colocándolos en las medias verdaderas
y asignando por cercanía). Ahí es donde las lunas se distinguen del resto.

Ejecútalo con:  python code/03-kmeans/kmeans_limites.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)


def dist2(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def kmeans(X, k, semilla=0, iters=100):
    r = np.random.default_rng(semilla)
    C = [X[r.integers(len(X))]]
    for _ in range(k - 1):                       # k-means++
        d2 = dist2(X, np.array(C)).min(axis=1)
        C.append(X[r.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        z = dist2(X, C).argmin(axis=1)
        Cn = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                       for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return dist2(X, C).argmin(axis=1), C


def acuerdo(a, b):
    """Acuerdo por parejas: inmune a la permutación de las etiquetas."""
    ia = a[:, None] == a[None, :]
    ib = b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


def inercia_de(X, z, k=2):
    """Inercia de una partición cualquiera, con sus propios centroides óptimos."""
    C = np.array([X[z == j].mean(axis=0) for j in range(k)])
    return ((X - C[z]) ** 2).sum(), C


def expresable(X, y, k=2):
    """¿Podría k-means REPRESENTAR la partición verdadera? Colocamos los
    centroides en las medias verdaderas y asignamos por cercanía."""
    _, C = inercia_de(X, y, k)
    z = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return max((z == y).mean(), (z != y).mean())


# --- Caso 1: dos cigarros paralelos, estirados 5:1 y girados 25 grados ---
estira = np.array([[3.0, 0.0], [0.0, 0.6]])
th = np.deg2rad(25)
giro = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
M = giro @ estira
X1 = np.vstack([rng.normal(size=(150, 2)) @ M.T,
                rng.normal(size=(150, 2)) @ M.T + [-1.0, 3.2]])
y1 = np.repeat([0, 1], 150)

# --- Caso 2: un grupo enorme y disperso, otro pequeño y compacto ---
X2 = np.vstack([rng.normal(scale=2.6, size=(280, 2)),
                rng.normal(scale=0.35, size=(40, 2)) + [5.5, 0.0]])
y2 = np.concatenate([np.zeros(280, int), np.ones(40, int)])

# --- Caso 3: dos lunas entrelazadas (no convexas) ---
t = rng.uniform(0, np.pi, 200)
luna_a = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(scale=0.09, size=(200, 2))
luna_b = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)]) + rng.normal(scale=0.09, size=(200, 2))
X3 = np.vstack([luna_a, luna_b])
y3 = np.repeat([0, 1], 200)

casos = [("1. alargados", X1, y1, "-> mixturas gaussianas, cap. 4"),
         ("2. tamaños dispares", X2, y2, "-> densidad, cap. 6"),
         ("3. no convexos", X3, y3, "-> espectral, cap. 7")]

print(f"{'caso':22s} {'acuerdo':>8} {'J de k-means':>13} {'J de la verdad':>15} "
      f"{'¿expresable?':>13}")
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for col, (nombre, X, y, salida) in enumerate(casos):
    z, C = kmeans(X, 2, semilla=0)
    a = acuerdo(z, y)
    J_km = inercia_de(X, z)[0]
    J_ver = inercia_de(X, y)[0]
    print(f"{nombre:22s} {a:7.1%} {J_km:13.1f} {J_ver:15.1f} {expresable(X, y):12.1%}")
    ax[0, col].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, edgecolors="none")
    ax[0, col].set_title(f"{nombre}: verdad")
    ax[1, col].scatter(X[:, 0], X[:, 1], c=z, cmap="coolwarm", s=10, edgecolors="none")
    ax[1, col].scatter(C[:, 0], C[:, 1], marker="X", s=160, c="k")
    ax[1, col].set_title(f"k-means: {a:.0%} de acuerdo")
    for fila in range(2):
        ax[fila, col].set_xticks([])
        ax[fila, col].set_yticks([])

print("\nEn los TRES casos la partición verdadera tiene PEOR inercia que la que")
print("encuentra k-means. Es decir: k-means no se equivoca al optimizar, optimiza")
print("bien; lo que pasa es que su objetivo prefiere la respuesta que no queremos.")
print("Cambiar de algoritmo no arregla eso: hay que cambiar de OBJETIVO, y eso es")
print("lo que hacen los capítulos 4, 6 y 7.")
print("\nLa última columna es otra cosa: cuánto de la partición correcta se puede")
print("siquiera expresar con dos centroides. Ahí las lunas son el caso más severo,")
print("porque ninguna frontera recta las sigue.")
for nombre, X, y, salida in casos:
    print(f"  {nombre:22s} {salida}")
plt.tight_layout()
plt.show()
