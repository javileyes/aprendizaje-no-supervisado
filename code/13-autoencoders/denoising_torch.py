"""
Autoencoder DENOISING con PyTorch. Solo terminal: el navegador no puede
ejecutar torch, así que este fichero no lleva botón de "Ejecutar".

Idea (Vincent et al., 2008): corrompe la entrada y pide reconstruir la ORIGINAL.
Así el autoencoder no puede limitarse a copiar; tiene que aprender de qué está
hecho el dato para poder rellenar lo que falta. Es el mismo principio que años
después dio lugar a BERT ("adivina la palabra tapada") y a los modelos de
difusión ("quita el ruido").

Comparamos tres autoencoders con la MISMA arquitectura y el mismo presupuesto
de entrenamiento, cambiando solo el ruido de entrada, y los evaluamos sobre
datos limpios que ninguno ha visto.

Requiere:  pip install torch
Ejecútalo con:  python code/13-autoencoders/denoising_torch.py
"""
import numpy as np
import torch
import torch.nn as nn

SEMILLA = 0
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)
rng = np.random.default_rng(SEMILLA)

# --- Datos: dígitos sintéticos de 12x12 formados por trazos ---
LADO = 12
D = LADO * LADO


def genera(n):
    """Cada imagen es la suma de 2-4 trazos rectos con grosor y posición al azar."""
    X = np.zeros((n, LADO, LADO), dtype=np.float32)
    for i in range(n):
        for _ in range(rng.integers(2, 5)):
            if rng.random() < 0.5:
                f = rng.integers(0, LADO - 1)
                c0, c1 = sorted(rng.integers(0, LADO, 2))
                X[i, f:f + 2, c0:c1 + 1] = 1.0
            else:
                c = rng.integers(0, LADO - 1)
                f0, f1 = sorted(rng.integers(0, LADO, 2))
                X[i, f0:f1 + 1, c:c + 2] = 1.0
    return torch.tensor(X.reshape(n, D))


X_tr, X_te = genera(4000), genera(1000)
print(f"Entrenamiento {tuple(X_tr.shape)}, prueba {tuple(X_te.shape)}. "
      f"Píxeles encendidos: {X_tr.mean():.1%}")


class AE(nn.Module):
    def __init__(self, latente=12):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 128), nn.ReLU(),
                                 nn.Linear(128, latente))
        self.dec = nn.Sequential(nn.Linear(latente, 128), nn.ReLU(),
                                 nn.Linear(128, D))

    def forward(self, x):
        return self.dec(self.enc(x))


def entrena(nivel_ruido, epocas=40, lote=128):
    torch.manual_seed(SEMILLA)
    modelo = AE()
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perdida = nn.BCEWithLogitsLoss()
    for _ in range(epocas):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), lote):
            xb = X_tr[perm[i:i + lote]]
            ruidoso = xb * (torch.rand_like(xb) > nivel_ruido).float()
            opt.zero_grad()
            L = perdida(modelo(ruidoso), xb)      # se reconstruye la ORIGINAL
            L.backward()
            opt.step()
    return modelo


@torch.no_grad()
def evalua(modelo, nivel_prueba):
    ruidoso = X_te * (torch.rand_like(X_te) > nivel_prueba).float()
    rec = torch.sigmoid(modelo(ruidoso))
    return ((rec - X_te) ** 2).mean().item()


print(f"\n{'ruido de entrenamiento':>24} {'error con 0% ruido':>20} "
      f"{'con 30%':>10} {'con 50%':>10}")
for nivel in [0.0, 0.2, 0.4]:
    m = entrena(nivel)
    fila = [evalua(m, p) for p in (0.0, 0.3, 0.5)]
    print(f"{nivel:23.0%} {fila[0]:20.5f} {fila[1]:10.5f} {fila[2]:10.5f}")

print("\nEl autoencoder entrenado SIN ruido gana cuando la prueba tampoco lo tiene,")
print("pero se degrada mucho más deprisa al corromper la entrada. El entrenado con")
print("ruido ha tenido que aprender la estructura de los trazos, no a copiar.")
