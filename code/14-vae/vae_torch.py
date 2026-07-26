"""
VAE con PyTorch. Solo terminal (el navegador no puede ejecutar torch).

Aquí sí podemos permitirnos un latente mayor y comparar de forma limpia el VAE
con un autoencoder normal en las dos cosas que los distinguen:

  A) MUESTREAR: se toma z ~ N(0,I) y se decodifica. El VAE produce datos
     plausibles; el autoencoder normal, basura, porque nadie le obligó a que
     su latente estuviera bien poblado.
  B) INTERPOLAR: se recorre el segmento entre los códigos de dos datos. En el
     VAE el camino pasa por imágenes válidas; en el AE, por zonas muertas.

Requiere:  pip install torch
Ejecútalo con:  python code/14-vae/vae_torch.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
rng = np.random.default_rng(0)
LADO, D, K = 12, 144, 8


def genera(n, r):
    X = np.zeros((n, LADO, LADO), dtype=np.float32)
    for i in range(n):
        for _ in range(r.integers(1, 4)):
            if r.random() < 0.5:
                X[i, r.integers(0, LADO), :] = 1.0
            else:
                X[i, :, r.integers(0, LADO)] = 1.0
    return torch.tensor(X.reshape(n, D))


X_tr = genera(6000, rng)
X_te = genera(1000, np.random.default_rng(5))
print(f"Entrenamiento {tuple(X_tr.shape)}; píxeles encendidos {X_tr.mean():.1%}")


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 256), nn.ReLU())
        self.mu = nn.Linear(256, K)
        self.lv = nn.Linear(256, K)
        self.dec = nn.Sequential(nn.Linear(K, 256), nn.ReLU(), nn.Linear(256, D))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.lv(h).clamp(-8, 8)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)   # reparametrización
        return self.dec(z), mu, lv


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, K))
        self.dec = nn.Sequential(nn.Linear(K, 256), nn.ReLU(), nn.Linear(256, D))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z, None


def entrena(modelo, es_vae, epocas=60, lote=128):
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    for _ in range(epocas):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), lote):
            xb = X_tr[perm[i:i + lote]]
            logits, mu, lv = modelo(xb)
            rec = F.binary_cross_entropy_with_logits(logits, xb, reduction="sum") / len(xb)
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean() if es_vae else 0.0
            opt.zero_grad()
            (rec + kl).backward()
            opt.step()
    return modelo


vae = entrena(VAE(), True)
ae = entrena(AE(), False)


@torch.no_grad()
def evalua(modelo, es_vae):
    logits, mu, lv = modelo(X_te)
    rec = F.binary_cross_entropy_with_logits(logits, X_te, reduction="sum") / len(X_te)
    kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean().item() if es_vae else 0.0
    return rec.item(), kl


for nombre, m, es in [("VAE", vae, True), ("autoencoder", ae, False)]:
    rec, kl = evalua(m, es)
    print(f"{nombre:12s} reconstrucción {rec:8.3f}  KL {kl:7.3f}")


# --- A) Muestrear del prior ---
@torch.no_grad()
def muestrea(modelo, n=1000):
    z = torch.randn(n, K)
    return torch.sigmoid(modelo.dec(z))


def estructura(A):
    im = A.reshape(-1, LADO, LADO)
    return im.mean(dim=2).std(dim=1).mean().item()


@torch.no_grad()
def dist_al_dato_mas_cercano(A):
    """Cuánto se parece cada imagen generada al dato REAL más próximo."""
    d = torch.cdist(A, X_te)
    return d.min(dim=1).values.mean().item()


print(f"\nA) Muestreo del prior z ~ N(0,I)")
print(f"{'':14} {'píxeles ON':>12} {'estructura':>12} {'dist. al dato real':>20}")
otros = genera(1000, np.random.default_rng(77))
print(f"{'datos reales':14} {X_te.mean().item():12.3f} {estructura(X_te):12.3f} "
      f"{dist_al_dato_mas_cercano(otros):20.3f}")
for nombre, m in [("VAE", vae), ("autoencoder", ae)]:
    S = muestrea(m)
    print(f"{nombre:14} {S.mean().item():12.3f} {estructura(S):12.3f} "
          f"{dist_al_dato_mas_cercano(S):20.3f}")
print("   'estructura' mide la dispersión entre filas: alta si hay trazos, baja si")
print("   la imagen es una mancha. La última columna es la distancia a la imagen")
print("   REAL más parecida: la fila de arriba es la referencia (datos de verdad).")


# --- B) Interpolar entre dos datos ---
@torch.no_grad()
def interpola(modelo, es_vae, pasos=9):
    x0, x1 = X_te[0:1], X_te[1:2]
    if es_vae:
        z0 = modelo.mu(modelo.enc(x0))
        z1 = modelo.mu(modelo.enc(x1))
    else:
        z0, z1 = modelo.enc(x0), modelo.enc(x1)
    alfas = torch.linspace(0, 1, pasos).unsqueeze(1)
    Z = (1 - alfas) * z0 + alfas * z1
    return torch.sigmoid(modelo.dec(Z))


print(f"\nB) Interpolación entre los códigos de dos datos (9 pasos)")
print(f"{'':14} {'dist. media al dato real más próximo':>38}")
for nombre, m, es in [("VAE", vae, True), ("autoencoder", ae, False)]:
    C = interpola(m, es)
    print(f"{nombre:14} {dist_al_dato_mas_cercano(C):38.3f}")
print("   Más baja = el camino entre dos datos pasa por imágenes plausibles.")
