"""
Factorización no negativa (NMF) con las reglas multiplicativas de Lee y Seung.

Buscamos X ~ W H con TODO no negativo. La restricción de no negatividad
prohíbe las cancelaciones y obliga a que la representación sea ADITIVA:
por eso las bases que aprende son "partes" y no "fantasmas" como en PCA.

Construimos 300 imágenes de 12x12 sumando 4 partes elementales (barras y
bloques) con pesos aleatorios, y comprobamos si NMF recupera esas partes.
Comparamos con PCA sobre los mismos datos.

Ejecútalo con:  python code/11-nmf-e-ica/nmf_lee_seung.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
lado, k_real, n = 12, 4, 300

# --- Las cuatro "partes" verdaderas: dos barras y dos bloques ---
partes = np.zeros((k_real, lado, lado))
partes[0, 2:4, :] = 1.0            # barra horizontal
partes[1, :, 8:10] = 1.0           # barra vertical
partes[2, 7:11, 1:5] = 1.0         # bloque abajo a la izquierda
partes[3, 0:4, 0:3] = 1.0          # bloque arriba a la izquierda
partes = partes.reshape(k_real, -1)

pesos = rng.random((n, k_real)) * (rng.random((n, k_real)) > 0.4)   # activaciones dispersas
X = pesos @ partes + 0.02 * rng.random((n, lado * lado))            # ruido no negativo
d = X.shape[1]
print(f"{n} imágenes de {lado}x{lado} = {d} píxeles, generadas por {k_real} partes.")
print(f"Mínimo de X: {X.min():.4f} (todo no negativo, como debe ser)")


def nmf(X, k, iters=400, semilla=0):
    """Reglas multiplicativas de Lee y Seung para minimizar ||X - WH||_F^2."""
    r = np.random.default_rng(semilla)
    n, d = X.shape
    W = r.random((n, k)) + 0.1
    H = r.random((k, d)) + 0.1
    eps = 1e-10
    errores = []
    for _ in range(iters):
        # H <- H * (W^T X) / (W^T W H)
        H *= (W.T @ X) / (W.T @ W @ H + eps)
        # W <- W * (X H^T) / (W H H^T)
        W *= (X @ H.T) / (W @ H @ H.T + eps)
        errores.append(np.linalg.norm(X - W @ H) ** 2)
    return W, H, np.array(errores)


W, H, err = nmf(X, k_real)
print(f"\nNMF: error {err[0]:.3f} -> {err[-1]:.3f} en {len(err)} iteraciones")
print(f"  ¿monótono? {'sí' if (np.diff(err) <= 1e-6).all() else 'NO'}")
print(f"  mínimos de W y H: {W.min():.3e}, {H.min():.3e}  (nunca negativos)")

# --- ¿Ha recuperado las partes? Emparejamos por correlación ---
# OJO: en VALOR ABSOLUTO. El signo de una componente de PCA es arbitrario
# (si u es autovector, -u también lo es), así que comparar con signo penalizaría
# a PCA por algo que no es un error. Con NMF el problema no se plantea: todo
# es no negativo.
def empareja(A, B):
    """Empareja cada fila de A con una de B maximizando la suma de |correlación|.
    Con k=4 se pueden probar las 24 permutaciones y quedarse con la mejor."""
    from itertools import permutations
    C = np.array([[abs(np.corrcoef(a, b)[0, 1]) for b in B] for a in A])
    mejor = max(permutations(range(len(B))), key=lambda p: sum(C[i, p[i]] for i in range(len(A))))
    return [(i, mejor[i], C[i, mejor[i]]) for i in range(len(A))]


print("\n|Correlación| entre cada componente aprendida y la parte real que le toca:")
for i, j, c in sorted(empareja(H, partes), key=lambda p: p[1]):
    print(f"  componente NMF {i} <-> parte real {j}: {c:.4f}")
print(f"  media: {np.mean([c for _, _, c in empareja(H, partes)]):.4f}")

# --- PCA sobre los mismos datos, para comparar ---
mu = X.mean(axis=0)
Uc, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
comp_pca = Vt[:k_real]
print("\nLo mismo con PCA:")
for i, j, c in sorted(empareja(comp_pca, partes), key=lambda p: p[1]):
    print(f"  componente PCA {i} <-> parte real {j}: {c:.4f}")
print(f"  media: {np.mean([c for _, _, c in empareja(comp_pca, partes)]):.4f}")
print("\nPCA también las encuentra. La diferencia NO está en acertar las partes,")
print("sino en CÓMO las representa:")
print(f"  fracción de píxeles negativos en las componentes de PCA: "
      f"{(comp_pca < 0).mean():.1%}")
print(f"  fracción de píxeles negativos en las de NMF            : "
      f"{(H < 0).mean():.1%}")

fig, ax = plt.subplots(3, k_real, figsize=(10, 7.5))
for j in range(k_real):
    ax[0, j].imshow(partes[j].reshape(lado, lado), cmap="viridis")
    ax[0, j].set_title(f"parte real {j}", fontsize=10)
for i, j, _ in empareja(H, partes):
    ax[1, j].imshow(H[i].reshape(lado, lado), cmap="viridis")
    ax[1, j].set_title(f"NMF {i}", fontsize=10)
for i, j, _ in empareja(comp_pca, partes):
    ax[2, j].imshow(comp_pca[i].reshape(lado, lado), cmap="coolwarm")
    ax[2, j].set_title(f"PCA {i}", fontsize=10)
for fila in ax:
    for a in fila:
        a.set_xticks([])
        a.set_yticks([])
plt.tight_layout()
plt.show()
