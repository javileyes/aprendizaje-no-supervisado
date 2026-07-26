"""
k-means es el límite duro de EM cuando las gaussianas se hacen infinitamente
estrechas.

Fijamos todas las covarianzas a sigma^2 * I y pesos iguales, y vamos bajando
sigma. Medimos tres cosas:
  - la responsabilidad máxima media (cuánto se parece la asignación blanda a
    una dura: 1/k si el modelo duda, 1 si está seguro);
  - la entropía media de la asignación (debe caer a 0);
  - el porcentaje de puntos cuyo argmax coincide con la asignación de k-means.

Ejecútalo con:  python code/04-gmm-y-em/gmm_limite_kmeans.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

centros = np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 4.0]])
X = np.vstack([c + rng.normal(scale=1.1, size=(120, 2)) for c in centros])
n, d = X.shape
k = 3


def kmeans(X, k, semilla=0, iters=100):
    r = np.random.default_rng(semilla)
    C = [X[r.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        C.append(X[r.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        z = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        Cn = np.array([X[z == j].mean(axis=0) if np.any(z == j) else C[j]
                       for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    z = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return z, C, ((X - C[z]) ** 2).sum()


z_km, C, inercia = kmeans(X, k)
print(f"k-means: inercia {inercia:.3f}, tamaños {np.bincount(z_km)}\n")

# Con Sigma = sigma^2 I y pesos iguales, el paso E se reduce a un softmax de
# las distancias al cuadrado dividido por 2*sigma^2. Esa es toda la historia.
d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

sigmas = [3.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
resp_max, entropias, decididos = [], [], []
print(f"{'sigma':>7} {'resp. máxima':>13} {'entropía':>10} {'ya decididos':>13}")
for s in sigmas:
    logits = -d2 / (2 * s ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    gamma = np.exp(logits)
    gamma /= gamma.sum(axis=1, keepdims=True)
    H = -(gamma * np.log(gamma + 1e-300)).sum(axis=1).mean()
    seguros = (gamma.max(axis=1) > 0.99).mean()     # puntos sin ninguna duda
    resp_max.append(gamma.max(axis=1).mean())
    entropias.append(H)
    decididos.append(seguros)
    print(f"{s:7.2f} {gamma.max(axis=1).mean():13.4f} {H:10.4f} {seguros:12.1%}")

print(f"\nCon k=3, la entropía máxima (duda total) es log(3) = {np.log(3):.4f} nats.")
print("Cuando sigma -> 0 la entropía tiende a 0: la asignación blanda se vuelve dura.")
print(f"El argmax, en cambio, NO cambia nunca con sigma: coincide con k-means en el "
      f"{(gamma.argmax(axis=1) == z_km).mean():.0%} de los puntos para todo sigma,")
print("porque dividir todos los logits por una constante positiva no altera cuál es")
print("el mayor. Lo que sigma controla es la CONFIANZA, no la decisión.")

# Y el objetivo también converge: E[log-verosimilitud completa] -> -J/(2 sigma^2)
for s in [1.0, 0.1, 0.02]:
    logits = -d2 / (2 * s ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    gamma = np.exp(logits)
    gamma /= gamma.sum(axis=1, keepdims=True)
    esperado = (gamma * d2).sum()
    print(f"sigma={s:5.2f} -> suma ponderada de distancias^2 = {esperado:9.3f} "
          f"(inercia de k-means: {inercia:.3f})")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].semilogx(sigmas, resp_max, "o-", color="#5b8def")
ax[0].axhline(1 / k, ls=":", c="k")
ax[0].set_xlabel("sigma")
ax[0].set_title("responsabilidad máxima media")
ax[1].semilogx(sigmas, entropias, "o-", color="#ef476f")
ax[1].axhline(np.log(k), ls=":", c="k")
ax[1].set_xlabel("sigma")
ax[1].set_title("entropía de la asignación (nats)")
ax[2].semilogx(sigmas, decididos, "o-", color="#43c59e")
ax[2].set_ylim(-0.05, 1.05)
ax[2].set_xlabel("sigma")
ax[2].set_title("puntos con responsabilidad > 0,99")
plt.tight_layout()
plt.show()
