"""
La misma nube de puntos, dos respuestas igual de válidas.

Un conjunto de clientes descritos por (gasto en euros, antigüedad en años).
Cambiamos SOLO las unidades -normalizando o no- y el mismo algoritmo devuelve
una partición completamente distinta. Ninguna de las dos es "la correcta":
la elección de la escala ES parte de la pregunta.

Ejecútalo con:  python code/01-aprender-sin-etiquetas/dos_respuestas.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

# Dos "bandas" en gasto (300 y 900 euros) y dos en antigüedad (1 y 9 años).
gasto = np.concatenate([rng.normal(300, 60, 100), rng.normal(900, 60, 100)])
antig = np.concatenate([rng.normal(1.0, 0.6, 50), rng.normal(9.0, 0.6, 50)] * 2)
X = np.column_stack([gasto, antig])


def kmeans(X, k, semilla=0, iters=100):
    r = np.random.default_rng(semilla)
    C = X[r.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        z = d2.argmin(axis=1)
        C_nuevo = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                            for j in range(k)])
        if np.allclose(C_nuevo, C):
            break
        C = C_nuevo
    return z


# --- Respuesta A: unidades originales. El euro aplasta al año ---
z_bruto = kmeans(X, k=2, semilla=0)

# --- Respuesta B: cada columna con media 0 y desviación 1 ---
Z = (X - X.mean(axis=0)) / X.std(axis=0)
z_norm = kmeans(Z, k=2, semilla=0)

# ¿Cuánto se parecen las dos particiones? Como las etiquetas 0/1 son arbitrarias,
# medimos el acuerdo por PAREJAS: ¿los mismos dos puntos caen juntos en ambas?
def acuerdo_por_parejas(a, b):
    ia = a[:, None] == a[None, :]
    ib = b[:, None] == b[None, :]
    triu = np.triu_indices(len(a), k=1)
    return (ia[triu] == ib[triu]).mean()

print(f"Reparto con unidades originales : {np.bincount(z_bruto)}")
print(f"Reparto tras normalizar         : {np.bincount(z_norm)}")
print(f"Acuerdo por parejas entre ambas : {acuerdo_por_parejas(z_bruto, z_norm):.1%}")

# ¿Qué separa cada una? Diferencia entre los dos centros, en unidades originales.
c_bruto = np.array([X[z_bruto == j].mean(axis=0) for j in range(2)])
c_norm = np.array([X[z_norm == j].mean(axis=0) for j in range(2)])
print(f"Sin normalizar, los centros difieren en {abs(c_bruto[0,0]-c_bruto[1,0]):7.1f} EUR "
      f"y {abs(c_bruto[0,1]-c_bruto[1,1]):.2f} anios")
print(f"Normalizando,   los centros difieren en {abs(c_norm[0,0]-c_norm[1,0]):7.1f} EUR "
      f"y {abs(c_norm[0,1]-c_norm[1,1]):.2f} anios")

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for a, z, t in [(ax[0], z_bruto, "A: unidades originales"),
                (ax[1], z_norm, "B: columnas normalizadas")]:
    a.scatter(X[:, 0], X[:, 1], c=z, cmap="coolwarm", s=18, edgecolors="none")
    a.set_title(t)
    a.set_xlabel("gasto (EUR)")
ax[0].set_ylabel("antiguedad (anios)")
plt.tight_layout()
plt.show()
