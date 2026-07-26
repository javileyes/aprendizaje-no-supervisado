"""
¿Cuánto importa la inicialización? Azar frente a k-means++, con 30 semillas.

Una sola ejecución no dice nada: k-means es un algoritmo de descenso local y su
resultado depende de dónde empiece. Repetimos las dos estrategias 30 veces sobre
los MISMOS datos y comparamos la distribución de la inercia final.

Ejecútalo con:  python code/03-kmeans/kmeans_inicializacion.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng_datos = np.random.default_rng(1)

# Datos deliberadamente traicioneros: grupos muy separados, de forma que un
# arranque malo (dos centros en el mismo grupo) se queda atrapado para siempre.
centros = np.array([[0, 0], [12, 0], [24, 0], [0, 12], [12, 12], [24, 12]])
X = np.vstack([c + rng_datos.normal(scale=1.2, size=(60, 2)) for c in centros])
k = 6


def dist2(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def lloyd(X, C, iters=100):
    for _ in range(iters):
        z = dist2(X, C).argmin(axis=1)
        C_nuevo = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                            for j in range(len(C))])
        if np.allclose(C_nuevo, C):
            break
        C = C_nuevo
    z = dist2(X, C).argmin(axis=1)
    return ((X - C[z]) ** 2).sum()


def init_azar(X, k, rng):
    return X[rng.choice(len(X), k, replace=False)]


def init_pp(X, k, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = dist2(X, np.array(C)).min(axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(C)


resultados = {}
for nombre, init in [("azar", init_azar), ("k-means++", init_pp)]:
    Js = np.array([lloyd(X, init(X, k, np.random.default_rng(s))) for s in range(30)])
    resultados[nombre] = Js
    print(f"{nombre:10s} -> mediana {np.median(Js):8.1f} | mejor {Js.min():8.1f} "
          f"| peor {Js.max():8.1f} | desviación {Js.std():7.1f}")

mejor_global = min(J.min() for J in resultados.values())
print(f"\nMejor inercia encontrada por cualquiera: {mejor_global:.1f}")
for nombre, J in resultados.items():
    print(f"  {nombre:10s}: {np.mean(J <= 1.02 * mejor_global):.0%} de las 30 semillas "
          f"llegan a menos del 2% del óptimo")

fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([resultados["azar"], resultados["k-means++"]])
ax.set_xticks([1, 2])
ax.set_xticklabels(["azar", "k-means++"])
ax.set_ylabel("inercia final (30 semillas)")
ax.set_title("La inicialización decide si te quedas en un mínimo local")
plt.tight_layout()
plt.show()
