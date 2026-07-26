"""
La maldición de la dimensionalidad, medida en vez de contada.

Tres efectos, los tres verificables con numpy:
  1. Concentración de distancias: en dimensión alta todos los puntos están a la
     MISMA distancia, así que "el vecino más próximo" deja de significar algo.
  2. El volumen se va a la cáscara: casi todo el volumen de una bola está pegado
     a su superficie.
  3. Hubness: unos pocos puntos se vuelven vecinos de casi todo el mundo.

Comprobamos además la predicción teórica exacta para el cubo unidad: la media de
la distancia crece como sqrt(d/6) mientras su desviación tiende a la constante
0,5*sqrt(7/30) = 0,2415. La dispersión RELATIVA, por tanto, cae como 1/sqrt(d).
El contraste (dmax-dmin)/dmin hereda ese ritmo solo asintóticamente: es un
estadístico de extremos y al principio cae bastante más deprisa.

Ejecútalo con:  python code/02-geometria-y-dimensionalidad/maldicion_dimensionalidad.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 500
dims = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]

contrastes, medias, desviaciones = [], [], []
print(f"{'d':>5} {'media dist':>11} {'sd dist':>9} {'sd teórica':>11} {'contraste':>10}")
for d in dims:
    X = rng.random((n, d))                       # uniforme en el cubo unidad
    # matriz de distancias por la identidad ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq = (X ** 2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    D = np.sqrt(D2)
    np.fill_diagonal(D, np.inf)
    dmin = D.min(axis=1)
    dmax = np.where(np.isinf(D), -np.inf, D).max(axis=1)
    contraste = ((dmax - dmin) / dmin).mean()    # contraste relativo por punto
    fuera = D[np.triu_indices(n, k=1)]
    medias.append(fuera.mean())
    desviaciones.append(fuera.std())
    contrastes.append(contraste)
    print(f"{d:5d} {fuera.mean():11.3f} {fuera.std():9.4f} "
          f"{0.5*np.sqrt(7/30):11.4f} {contraste:10.3f}")

# --- Efecto 2: ¿qué fracción del volumen de la bola está en el 10% exterior? ---
print("\nFracción del volumen de una bola que vive en el 10% exterior del radio:")
for d in [1, 2, 3, 10, 50, 100]:
    print(f"  d={d:4d} -> {1 - 0.9 ** d:6.1%}")

# --- Efecto 3: hubness. ¿Cuántas veces aparece cada punto como vecino de otro? ---
print("\nHubness (nº de veces que un punto es 1er vecino de otro; ideal ~1):")
for d in [2, 20, 200]:
    X = rng.random((n, d))
    sq = (X ** 2).sum(axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    np.fill_diagonal(D, np.inf)
    cuentas = np.bincount(D.argmin(axis=1), minlength=n)
    centro = np.linalg.norm(X - X.mean(axis=0), axis=1)   # lejanía al centro de masas
    print(f"  d={d:4d} -> máximo {cuentas.max():3d}, "
          f"puntos que no son vecinos de nadie: {(cuentas == 0).mean():.0%}, "
          f"corr(hubness, dist. al centro)={np.corrcoef(cuentas, centro)[0, 1]:+.3f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(dims, contrastes, "o-", color="#ef476f")
ax[0].set_xscale("log")
ax[0].set_xlabel("dimensión d")
ax[0].set_title("contraste (dmax-dmin)/dmin")

ax[1].plot(dims, medias, "o-", label="media medida")
ax[1].plot(dims, [np.sqrt(d / 6) for d in dims], "--", label="sqrt(d/6) teórica")
ax[1].plot(dims, desviaciones, "s-", label="desviación medida")
ax[1].axhline(0.5 * np.sqrt(7 / 30), ls=":", color="k", label="0,2415 teórica")
ax[1].set_xscale("log")
ax[1].set_xlabel("dimensión d")
ax[1].legend(fontsize=8)
ax[1].set_title("distancia media y su dispersión")

dd = np.arange(1, 101)
ax[2].plot(dd, 1 - 0.9 ** dd, color="#43c59e")
ax[2].set_xlabel("dimensión d")
ax[2].set_title("volumen en el 10% exterior")
plt.tight_layout()
plt.show()
