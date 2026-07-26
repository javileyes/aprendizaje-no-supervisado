"""
Un autoencoder LINEAL entrenado con error cuadrático reencuentra PCA.

Teorema (Baldi y Hornik, 1989): si el codificador y el decodificador son
lineales y la pérdida es el error cuadrático, todos los mínimos del problema
generan el MISMO subespacio que las k primeras componentes principales. Los
pesos no coinciden con los autovectores -pueden estar rotados dentro del
subespacio- pero el subespacio sí.

Lo comprobamos entrenando con descenso de gradiente escrito a mano y midiendo
la distancia entre subespacios y el error de reconstrucción.

Ejecútalo con:  python code/13-autoencoders/autoencoder_lineal_es_pca.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
n, d, k = 500, 20, 3

# --- Datos con estructura de rango 3 más ruido ---
base = np.linalg.qr(rng.normal(size=(d, k)))[0]
X = rng.normal(size=(n, k)) @ np.diag([4.0, 2.5, 1.5]) @ base.T
X += 0.6 * rng.normal(size=(n, d))
X -= X.mean(axis=0)
print(f"{n} puntos en dimensión {d}, con {k} direcciones dominantes.")

# --- Referencia: PCA ---
S = (X.T @ X) / (n - 1)
val, vec = np.linalg.eigh(S)
val, vec = val[::-1], vec[:, ::-1]
U_pca = vec[:, :k]
err_pca = ((X - (X @ U_pca) @ U_pca.T) ** 2).mean()
print(f"PCA con {k} componentes -> error cuadrático medio {err_pca:.6f}")


def dist_subespacios(A, B):
    """0 si A y B generan el mismo subespacio."""
    QA = np.linalg.qr(A)[0]
    QB = np.linalg.qr(B)[0]
    return np.linalg.norm(QA @ QA.T - QB @ QB.T)


# --- Autoencoder lineal: X -> XW1 -> XW1W2, con gradientes a mano ---
# Pérdida: L = (1/(n*d)) ||X - X W1 W2||^2
W1 = 0.1 * rng.normal(size=(d, k))       # codificador
W2 = 0.1 * rng.normal(size=(k, d))       # decodificador
lr, iters = 0.05, 4000
historia, distancias = [], []

for it in range(iters):
    Z = X @ W1                            # código
    R = Z @ W2                            # reconstrucción
    E = R - X                             # residuo
    L = (E ** 2).mean()
    # dL/dW2 = 2/(n*d) Z^T E   ;   dL/dW1 = 2/(n*d) X^T E W2^T
    gW2 = 2.0 / (n * d) * (Z.T @ E)
    gW1 = 2.0 / (n * d) * (X.T @ E @ W2.T)
    W1 -= lr * gW1
    W2 -= lr * gW2
    if it % 100 == 0 or it == iters - 1:
        historia.append((it, L))
        distancias.append((it, dist_subespacios(W1, U_pca)))

print(f"\n{'iteración':>10} {'error del AE':>14} {'dist. a PCA':>13}")
for (it, L), (_, dd) in zip(historia, distancias):
    if it % 500 == 0 or it == iters - 1:
        print(f"{it:10d} {L:14.6f} {dd:13.6f}")

Z = X @ W1
R = Z @ W2
print(f"\nError final del autoencoder : {((R - X) ** 2).mean():.6f}")
print(f"Error de PCA                : {err_pca:.6f}")
print(f"Diferencia                  : {abs(((R - X) ** 2).mean() - err_pca):.2e}")
print(f"\nDistancia entre subespacios (0 = idénticos): "
      f"{dist_subespacios(W1, U_pca):.6f}")
print(f"¿Son los pesos iguales a los autovectores? "
      f"||W1 - U_pca|| = {np.linalg.norm(W1 - U_pca):.4f}")
print("NO: el autoencoder encuentra el mismo SUBESPACIO con otra base. El")
print("descenso de gradiente no tiene ningún motivo para ordenar ni ortogonalizar.")

# --- ¿Y están ordenadas por varianza, como en PCA? ---
print(f"\nVarianza de cada coordenada del código:")
print(f"  autoencoder : {np.round(Z.var(axis=0, ddof=1), 3)}")
print(f"  PCA         : {np.round((X @ U_pca).var(axis=0, ddof=1), 3)}  "
      f"(= autovalores {np.round(val[:k], 3)})")
print("El AE reparte la varianza entre sus tres coordenadas sin ordenarlas ni")
print("concentrarlas: cualquier rotación dentro del subespacio le da igual.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].semilogy([h[0] for h in historia], [h[1] for h in historia], color="#5b8def")
ax[0].axhline(err_pca, ls="--", c="#ef476f")
ax[0].set_xlabel("iteración")
ax[0].set_ylabel("error cuadrático medio")
ax[0].set_title("el AE baja hasta el error de PCA")
ax[1].semilogy([d_[0] for d_ in distancias], [d_[1] for d_ in distancias],
               color="#43c59e")
ax[1].set_xlabel("iteración")
ax[1].set_ylabel("distancia entre subespacios")
ax[1].set_title("y converge al subespacio de PCA")
P_pca = X @ U_pca
ax[2].scatter(P_pca[:, 0], P_pca[:, 1], s=8, c="#8b93a7", label="PCA")
ax[2].scatter(Z[:, 0], Z[:, 1], s=8, c="#e2b53d", alpha=0.6, label="autoencoder")
ax[2].legend()
ax[2].set_title("los códigos: mismo subespacio, otra base")
ax[2].set_xticks([])
ax[2].set_yticks([])
plt.tight_layout()
plt.show()
