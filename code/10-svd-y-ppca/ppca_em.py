"""
PCA probabilístico (Tipping y Bishop, 1999): PCA como modelo generativo.

Modelo:  z ~ N(0, I_k),   x = W z + mu + eps,   eps ~ N(0, sigma^2 I)
de donde  x ~ N(mu, W W^T + sigma^2 I).

Comprobamos cuatro cosas:
  1. la solución de máxima verosimilitud en forma CERRADA:
        sigma^2 = (1/(d-k)) * suma de los autovalores descartados
        W = U_k (Lambda_k - sigma^2 I)^{1/2}
  2. que EM converge a esa misma solución (y que la log-verosimilitud sube);
  3. que el subespacio que encuentra es el de PCA;
  4. que cuando sigma^2 -> 0 la posterior E[z|x] tiende a la proyección de PCA
     BLANQUEADA, (x . u_j)/sqrt(lambda_j).

La ventaja de tener un modelo: da una densidad p(x) con la que puntuar datos
nuevos, y EM permite tratar valores ausentes.

Ejecútalo con:  python code/10-svd-y-ppca/ppca_em.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(9)
n, d, k = 600, 12, 3

# --- Datos generados EXACTAMENTE por el modelo, para saber la verdad ---
W_real = rng.normal(size=(d, k))
mu_real = rng.normal(size=d) * 2
sigma2_real = 0.7
Z = rng.normal(size=(n, k))
X = Z @ W_real.T + mu_real + rng.normal(scale=np.sqrt(sigma2_real), size=(n, d))
print(f"{n} puntos en dimensión {d}, generados por {k} factores latentes "
      f"y ruido sigma^2 = {sigma2_real}.")

mu = X.mean(axis=0)
Xc = X - mu
S = (Xc.T @ Xc) / n
val, vec = np.linalg.eigh(S)
val, vec = val[::-1], vec[:, ::-1]


def log_verosimilitud(Xc, W, sigma2):
    """log N(x | 0, W W^T + sigma^2 I) sumada sobre los datos."""
    C = W @ W.T + sigma2 * np.eye(W.shape[0])
    L = np.linalg.cholesky(C)
    dif = np.linalg.solve(L, Xc.T)
    return -0.5 * (len(Xc) * (W.shape[0] * np.log(2 * np.pi)
                              + 2 * np.log(np.diag(L)).sum())
                   + (dif ** 2).sum())


# ---------------- 1) Solución cerrada ----------------
sigma2_ml = val[k:].mean()
W_ml = vec[:, :k] @ np.diag(np.sqrt(np.maximum(val[:k] - sigma2_ml, 0)))
print(f"\n1) Solución en forma cerrada")
print(f"   sigma^2 estimado = {sigma2_ml:.6f}   (real: {sigma2_real})")
print(f"   media de los {d - k} autovalores descartados = {val[k:].mean():.6f}")
print(f"   log-verosimilitud = {log_verosimilitud(Xc, W_ml, sigma2_ml):.4f}")


# ---------------- 2) EM ----------------
def em_ppca(Xc, k, iters=300):
    n, d = Xc.shape
    r = np.random.default_rng(0)
    W = r.normal(size=(d, k)) * 0.1
    sigma2 = 1.0
    historia = []
    for _ in range(iters):
        # PASO E: posterior de z dado x. M = W^T W + sigma^2 I
        M = W.T @ W + sigma2 * np.eye(k)
        Minv = np.linalg.inv(M)
        Ez = Xc @ W @ Minv                                 # n x k
        Ezz = n * sigma2 * Minv + Ez.T @ Ez                # k x k, suma de E[z z^T]
        # PASO M
        W = (Xc.T @ Ez) @ np.linalg.inv(Ezz)
        sigma2 = ((Xc ** 2).sum()
                  - 2 * np.trace(Ez.T @ Xc @ W)
                  + np.trace(Ezz @ (W.T @ W))) / (n * d)
        historia.append(log_verosimilitud(Xc, W, sigma2))
    return W, sigma2, np.array(historia)


W_em, sigma2_em, hist = em_ppca(Xc, k)
print(f"\n2) EM (300 iteraciones)")
print(f"   sigma^2 por EM = {sigma2_em:.6f}   (cerrada: {sigma2_ml:.6f})")
print(f"   log-verosimilitud: {hist[0]:.4f} -> {hist[-1]:.4f}")
print(f"   ¿monótona? {'sí' if (np.diff(hist) >= -1e-6).all() else 'NO'}")
print(f"   diferencia con la cerrada: "
      f"{abs(log_verosimilitud(Xc, W_ml, sigma2_ml) - hist[-1]):.2e}")

# ---------------- 3) ¿Es el mismo subespacio que PCA? ----------------
def dist_subespacios(A, B):
    """Distancia entre subespacios: 0 si generan el mismo."""
    QA = np.linalg.qr(A)[0]
    QB = np.linalg.qr(B)[0]
    return np.linalg.norm(QA @ QA.T - QB @ QB.T)


print(f"\n3) ¿Coincide el subespacio con el de PCA?")
print(f"   dist(subespacio de W_EM, subespacio de PCA)     = "
      f"{dist_subespacios(W_em, vec[:, :k]):.2e}")
print(f"   dist(subespacio cerrado, subespacio de PCA)     = "
      f"{dist_subespacios(W_ml, vec[:, :k]):.2e}")
print("   (W no es único: cualquier rotación W R sirve. Lo que se determina")
print("    es el SUBESPACIO, no la base.)")

# ---------------- 4) Qué aporta sigma^2: la confianza de la posterior ----------------
# Con W = U_k (Lambda_k - sigma^2 I)^{1/2} resulta M = W^T W + s2 I, de modo que
#   E[z|x]_j  = (x . u_j) * sqrt(lambda_j - sigma^2_ml) / (lambda_j - sigma^2_ml + s2)
#   Cov[z|x]  = s2 * M^{-1}      -> la INCERTIDUMBRE, que PCA no tiene
# Comparamos con la proyección blanqueada de PCA, (x . u_j)/sqrt(lambda_j).
z_pca_blanco = Xc @ vec[:, :k] / np.sqrt(val[:k])
print(f"\n4) La posterior p(z|x): escala e incertidumbre")
print(f"{'s2 usado':>10} {'escala medida':>15} {'escala teórica':>16} {'sd posterior':>14}")
for s2 in [3.0, 1.0, 0.3, 0.1, 0.01, 1e-4]:
    M = W_ml.T @ W_ml + s2 * np.eye(k)
    Ez = Xc @ W_ml @ np.linalg.inv(M)
    medido = np.std(Ez[:, 0]) / np.std(z_pca_blanco[:, 0])
    teorico = np.sqrt(val[0] - sigma2_ml) * np.sqrt(val[0]) / (val[0] - sigma2_ml + s2)
    sd_post = np.sqrt(s2 * np.linalg.inv(M)[0, 0])
    print(f"{s2:10.4f} {medido:15.6f} {teorico:16.6f} {sd_post:14.6f}")
print("   La media posterior es siempre la proyección de PCA reescalada: por eso")
print("   las dos primeras columnas coinciden. Lo NUEVO es la cuarta, la anchura")
print("   de p(z|x): cuando sigma^2 -> 0 la posterior COLAPSA sobre un punto y")
print("   PPCA deja de ser probabilístico. PCA es ese límite determinista.")

# ---------------- 5) La ventaja del modelo: una densidad ----------------
X_dentro = Z[:200] @ W_real.T + mu_real + rng.normal(scale=np.sqrt(sigma2_real),
                                                     size=(200, d))
X_fuera = mu_real + rng.normal(scale=3.0, size=(200, d))     # fuera del subespacio
C = W_ml @ W_ml.T + sigma2_ml * np.eye(d)
L = np.linalg.cholesky(C)


def log_dens(A):
    dif = np.linalg.solve(L, (A - mu).T)
    return -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum()
                   + (dif ** 2).sum(axis=0))


print(f"\n5) Con el modelo se pueden PUNTUAR datos nuevos (esto es el cap. 18)")
print(f"   log-densidad media de puntos del modelo : {log_dens(X_dentro).mean():9.3f}")
print(f"   log-densidad media de puntos fuera      : {log_dens(X_fuera).mean():9.3f}")
umbral = np.percentile(log_dens(X_dentro), 5)
print(f"   con umbral al percentil 5 de los normales, se detecta el "
      f"{(log_dens(X_fuera) < umbral).mean():.1%} de los anómalos")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(hist, color="#43c59e")
ax[0].axhline(log_verosimilitud(Xc, W_ml, sigma2_ml), ls="--", c="#ef476f")
ax[0].set_xlabel("iteración de EM")
ax[0].set_ylabel("log-verosimilitud")
ax[0].set_title("EM converge a la solución cerrada")
ax[1].plot(val, "o-", color="#5b8def")
ax[1].axhline(sigma2_ml, ls=":", c="k")
ax[1].set_xlabel("componente")
ax[1].set_title(f"espectro; la línea es sigma^2 = {sigma2_ml:.2f}")
Ez = Xc @ W_ml @ np.linalg.inv(W_ml.T @ W_ml + sigma2_ml * np.eye(k))
ax[2].scatter(Ez[:, 0], z_pca_blanco[:, 0], s=8, c="#8b93a7")
ax[2].set_xlabel("E[z|x] (PPCA)")
ax[2].set_ylabel("PCA blanqueado")
ax[2].set_title("las dos coordenadas, una contra otra")
plt.tight_layout()
plt.show()
