"""
La hipótesis de la variedad, medida en vez de contada.

Tomamos m coordenadas latentes uniformes y las embebemos linealmente en R^1000
con una matriz aleatoria. El espacio ambiente es SIEMPRE de dimensión 1000; lo
único que cambia es m. Si la maldición contase columnas, el contraste debería
ser igual de malo en todas las filas. Si cuenta grados de libertad, debería
seguir al contraste de una nube uniforme en R^m (la columna de la derecha).

Ejecútalo con:  python code/02-geometria-y-dimensionalidad/variedad.py
"""
import numpy as np

rng = np.random.default_rng(0)
n, D = 500, 1000


def contraste(X):
    """Contraste relativo medio (dmax - dmin) / dmin, promediado sobre puntos."""
    sq = (X ** 2).sum(axis=1)
    M = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    np.fill_diagonal(M, np.inf)
    dmin = M.min(axis=1)
    dmax = np.where(np.isinf(M), -np.inf, M).max(axis=1)
    return ((dmax - dmin) / dmin).mean()


print("La hipótesis de la variedad, medida: m coordenadas latentes en R^1000")
print(f"{'m':>5} {'contraste en R^1000':>21} {'contraste en R^m':>18}")
for m in [2, 5, 10, 50, 200, 1000]:
    Z = rng.random((n, m))                 # las m coordenadas de la variedad
    W = rng.normal(size=(m, D))            # el embebido en el espacio ambiente
    print(f"{m:5d} {contraste(Z @ W):21.3f} {contraste(rng.random((n, m))):18.3f}")
