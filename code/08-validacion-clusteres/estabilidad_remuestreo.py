"""
Estabilidad por remuestreo: la tercera familia de criterios (Ben-Hur, Elisseeff
y Guyon, 2002), la que no supone NINGUNA forma de grupo.

La idea: una estructura real se repite; un artefacto, no. Se toman dos
submuestras solapadas, se agrupa cada una por separado, y se mide si coinciden
en los puntos que ambas comparten. Se repite muchas veces y para cada k.

Y medimos también su punto ciego, que el texto anuncia y conviene ver: sobre
datos SIN NINGUNA estructura la estabilidad sigue siendo alta, porque un sesgo
sistemático del algoritmo es perfectamente reproducible.

Ejecútalo con:  python code/08-validacion-clusteres/estabilidad_remuestreo.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def kmeans(X, k, semilla=0, iters=200):
    r = np.random.default_rng(semilla)
    C = [X[r.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        C.append(X[r.choice(len(X), p=d2 / max(d2.sum(), 1e-12))])
    C = np.array(C)
    for _ in range(iters):
        z = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        Cn = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                       for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return C


def asigna(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def acuerdo(a, b):
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


def estabilidad(X, k, reps=30, frac=0.8, semilla=0):
    """Dos submuestras del 80 %, se agrupa cada una y se comparan EN LOS PUNTOS
    QUE COMPARTEN. El modelo aprendido en una se aplica a los puntos de la otra."""
    r = np.random.default_rng(semilla)
    n, m = len(X), int(frac * len(X))
    out = []
    for rep in range(reps):
        i1 = r.choice(n, m, replace=False)
        i2 = r.choice(n, m, replace=False)
        comun = np.intersect1d(i1, i2)
        if len(comun) < 10:
            continue
        C1 = kmeans(X[i1], k, rep)
        C2 = kmeans(X[i2], k, rep + 500)
        out.append(acuerdo(asigna(X[comun], C1), asigna(X[comun], C2)))
    return np.mean(out), np.std(out)


# --- Tres conjuntos: uno fácil, uno con forma, y uno SIN estructura ---
centros = np.array([[0.0, 0.0], [5.0, 0.5], [2.5, 5.0]])
XA = np.vstack([c + rng.normal(scale=0.8, size=(100, 2)) for c in centros])
t = rng.uniform(0, np.pi, 150)
XB = np.vstack([np.column_stack([np.cos(t), np.sin(t)]),
                np.column_stack([1 - np.cos(t), 0.4 - np.sin(t)])]) \
    + rng.normal(scale=0.07, size=(300, 2))
XC = rng.uniform(0, 10, size=(300, 2))

casos = [("A · tres grupos (k=3)", XA), ("B · dos lunas (k=2)", XB),
         ("C · uniforme (sin grupos)", XC)]

print(f"{'k':>3} " + " ".join(f"{n:>24}" for n, _ in casos))
tabla = {}
for k in range(2, 8):
    fila = []
    for nombre, X in casos:
        m, s = estabilidad(X, k)
        tabla[(nombre, k)] = m
        fila.append(f"{m:.3f} ± {s:.3f}")
    print(f"{k:3d} " + " ".join(f"{v:>24}" for v in fila))

print()
for nombre, _ in casos:
    ks = list(range(2, 8))
    mejor = max(ks, key=lambda k: tabla[(nombre, k)])
    print(f"  {nombre:26} la estabilidad elige k = {mejor} "
          f"({tabla[(nombre, mejor)]:.3f})")

print("\nA: acierta, y con un pico nítido. B: elige 4 donde hay 2, igual que")
print("los índices internos: las lunas rotas en trozos son MUY reproducibles.")
print("C: y aquí está el punto ciego. Sobre 300 puntos uniformes, sin ninguna")
print("estructura, la estabilidad no baja de 0,76 y llega a 0,94: k-means corta")
print("el cuadrado siempre por el mismo sitio, y eso es perfectamente estable.")
print("Reproducible no es lo mismo que real. Por eso la estabilidad complementa")
print("al gap statistic, que sí sabe decir «aquí no hay nada», pero no lo sustituye.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for nombre, _ in casos:
    ax[0].plot(range(2, 8), [tabla[(nombre, k)] for k in range(2, 8)], "o-", label=nombre)
ax[0].set_xlabel("k")
ax[0].set_ylabel("estabilidad (acuerdo entre submuestras)")
ax[0].legend(fontsize=8)
ax[0].set_title("más alto = más reproducible")
ax[1].scatter(XC[:, 0], XC[:, 1], s=10, c=asigna(XC, kmeans(XC, 4, 0)), cmap="viridis")
ax[1].set_title("C: k=4 sobre ruido, estabilidad 0,94")
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()
plt.show()
