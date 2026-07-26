"""
HDBSCAN desde cero: DBSCAN para todos los valores de epsilon a la vez.

Los cinco pasos del algoritmo de Campello, Moulavi y Sander (2013):
  1. distancia núcleo: d_core(x) = distancia al k-ésimo vecino de x;
  2. distancia de alcance mutuo: max(d_core(a), d_core(b), d(a,b));
  3. árbol de recubrimiento mínimo sobre esa distancia (Prim);
  4. jerarquía de enlace simple y CONDENSACIÓN por tamaño mínimo de grupo;
  5. selección de los grupos de mayor ESTABILIDAD.

Lo probamos sobre datos con dos densidades muy distintas, que es exactamente
el caso donde un epsilon único de DBSCAN no puede funcionar.

Ejecútalo con:  python code/06-densidad-dbscan/hdbscan_estabilidad.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# --- Datos: un grupo apretado, uno flojo y otro intermedio, más ruido ---
X = np.vstack([
    rng.normal(scale=0.14, size=(120, 2)) + [0.0, 0.0],      # denso
    rng.normal(scale=0.14, size=(120, 2)) + [1.1, 0.0],      # denso, pegado al anterior
    rng.normal(scale=1.50, size=(120, 2)) + [7.0, 1.0],      # muy disperso
    rng.uniform([-2, -4], [12, 6], size=(45, 2)),            # ruido uniforme
])
y = np.concatenate([np.zeros(120, int), np.ones(120, int),
                    np.full(120, 2), np.full(45, -1)])
n = len(X)
sq = (X ** 2).sum(axis=1)
D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
print(f"{n} puntos: tres grupos de 120 con dispersiones 0,14 / 0,14 / 1,50 y 45 de ruido.")


def expandir(z):
    """Cada punto marcado como ruido pasa a ser su propio grupo, para que
    agrupar todo el ruido junto NO se premie al medir el acuerdo."""
    z = z.copy()
    libre = z.max() + 1
    for i in np.flatnonzero(z < 0):
        z[i] = libre
        libre += 1
    return z


def acuerdo(a, b):
    a, b = expandir(a), expandir(b)
    ia, ib = a[:, None] == a[None, :], b[:, None] == b[None, :]
    t = np.triu_indices(len(a), k=1)
    return (ia[t] == ib[t]).mean()


def dbscan(D, eps, min_pts):
    n = len(D)
    vec = [np.flatnonzero(D[i] <= eps) for i in range(n)]
    nucleo = np.array([len(v) >= min_pts for v in vec])
    et = np.full(n, -1)
    c = 0
    for i in range(n):
        if et[i] != -1 or not nucleo[i]:
            continue
        et[i] = c
        cola = list(vec[i])
        while cola:
            j = cola.pop()
            if et[j] == -1:
                et[j] = c
                if nucleo[j]:
                    cola.extend(vec[j])
        c += 1
    return et


# ---------- Paso 1 y 2: distancia núcleo y alcance mutuo ----------
MIN_SAMPLES = 6
MIN_CLUSTER = 25
d_core = np.sort(D, axis=1)[:, MIN_SAMPLES - 1]
MR = np.maximum(np.maximum(D, d_core[:, None]), d_core[None, :])
print(f"\nDistancia núcleo (k={MIN_SAMPLES}): del grupo denso {d_core[:120].mean():.3f}, "
      f"del disperso {d_core[240:360].mean():.3f}, del ruido {d_core[360:].mean():.3f}")

# ---------- Paso 3: árbol de recubrimiento mínimo (algoritmo de Prim) ----------
dentro = np.zeros(n, bool)
dentro[0] = True
mejor_d = MR[0].copy()
mejor_o = np.zeros(n, int)
aristas = []
for _ in range(n - 1):
    mejor_d_masked = np.where(dentro, np.inf, mejor_d)
    j = int(np.argmin(mejor_d_masked))
    aristas.append((mejor_o[j], j, mejor_d[j]))
    dentro[j] = True
    menor = MR[j] < mejor_d
    mejor_d = np.where(menor, MR[j], mejor_d)
    mejor_o = np.where(menor, j, mejor_o)
aristas.sort(key=lambda e: e[2])

# ---------- Paso 4: jerarquía de enlace simple mediante union-find ----------
padre = list(range(2 * n))
tam = [1] * n + [0] * n


def raiz(x):
    while padre[x] != x:
        padre[x] = padre[padre[x]]
        x = padre[x]
    return x


hijos = {}
alturas = {}
nodo = n
for u, v, d in aristas:
    ru, rv = raiz(u), raiz(v)
    hijos[nodo] = (ru, rv)
    alturas[nodo] = d
    tam[nodo] = tam[ru] + tam[rv]
    padre[ru] = padre[rv] = padre[nodo] = nodo
    nodo += 1
raiz_arbol = nodo - 1


def hojas_de(x):
    pila, out = [x], []
    while pila:
        a = pila.pop()
        if a < n:
            out.append(a)
        else:
            pila.extend(hijos[a])
    return out


# ---------- Paso 4b: condensación ----------
# Recorremos de la raíz hacia abajo. Un hijo con menos de MIN_CLUSTER puntos
# "se cae" del grupo actual (pasa a ser ruido a ese nivel); si los dos hijos son
# suficientemente grandes, hay una división genuina y nacen dos grupos nuevos.
lam_caida = np.zeros(n)          # lambda a la que cada punto abandona su grupo
lam_nac, puntos_grupo, hijos_grupo = {}, {}, {}
sig = 0
lam_nac[sig] = 0.0               # el grupo 0 es la raíz: contiene todo
puntos_grupo[sig] = []
hijos_grupo[sig] = []
asignado = {raiz_arbol: sig}
sig += 1

pila = [raiz_arbol]
while pila:
    nodo = pila.pop()
    g = asignado[nodo]
    if nodo < n:
        continue
    lam = 1.0 / max(alturas[nodo], 1e-12)
    a, b = hijos[nodo]
    ta = tam[a] if a >= n else 1
    tb = tam[b] if b >= n else 1
    grandes = [(c, t) for c, t in ((a, ta), (b, tb)) if t >= MIN_CLUSTER]
    if len(grandes) == 2:                       # división genuina: dos grupos nuevos
        for c, _ in grandes:
            lam_nac[sig] = lam
            puntos_grupo[sig] = []
            hijos_grupo[sig] = []
            hijos_grupo[g].append(sig)
            asignado[c] = sig
            sig += 1
            pila.append(c)
    else:                                       # los pequeños se caen del grupo g
        for c, t in ((a, ta), (b, tb)):
            if t >= MIN_CLUSTER:
                asignado[c] = g
                pila.append(c)
            else:
                for h in hojas_de(c):
                    lam_caida[h] = lam
                    puntos_grupo[g].append(h)

# ---------- Paso 5: estabilidad y selección por exceso de masa ----------
def estabilidad(g):
    s = sum(lam_caida[h] - lam_nac[g] for h in puntos_grupo[g])
    for c in hijos_grupo[g]:
        s += len(hojas_totales(c)) * (lam_nac[c] - lam_nac[g])
    return s


def hojas_totales(g):
    out = list(puntos_grupo[g])
    for c in hijos_grupo[g]:
        out += hojas_totales(c)
    return out


S = {g: estabilidad(g) for g in lam_nac}
seleccionado, S_hat = {}, {}
for g in sorted(lam_nac, reverse=True):          # de las hojas hacia la raíz
    suma_hijos = sum(S_hat[c] for c in hijos_grupo[g])
    if g == 0:                                   # la raíz nunca se selecciona:
        S_hat[g] = suma_hijos                    # "todo en un grupo" no es respuesta
        seleccionado[g] = False
        continue
    if hijos_grupo[g] and suma_hijos > S[g]:
        S_hat[g] = suma_hijos
        seleccionado[g] = False
    else:
        S_hat[g] = S[g]
        seleccionado[g] = True
        for c in hijos_grupo[g]:                 # desactiva toda su descendencia
            pila2 = [c]
            while pila2:
                x = pila2.pop()
                seleccionado[x] = False
                pila2.extend(hijos_grupo[x])

elegidos = [g for g in lam_nac if seleccionado[g]]
z_h = np.full(n, -1)
for c, g in enumerate(elegidos):
    for h in hojas_totales(g):
        z_h[h] = c

print(f"\nÁrbol condensado: {len(lam_nac)} grupos candidatos; "
      f"seleccionados {len(elegidos)} por estabilidad.")
print(f"{'grupo':>6} {'nace en lambda':>15} {'puntos':>8} {'estabilidad':>13} {'elegido':>9}")
for g in sorted(lam_nac):
    marca = "sí" if seleccionado[g] else "no"
    print(f"{g:6d} {lam_nac[g]:15.4f} {len(hojas_totales(g)):8d} {S[g]:13.2f} {marca:>9}")

print(f"\nHDBSCAN -> {len(elegidos)} grupos, {(z_h == -1).mean():.1%} de ruido "
      f"(la verdad tiene {(y == -1).mean():.1%}), acuerdo {acuerdo(z_h, y):.1%}")

print("\nDBSCAN con un epsilon único, para comparar:")
print(f"{'eps':>6} {'grupos':>7} {'ruido':>7} {'acuerdo':>9}")
for eps in [0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.8, 2.4]:
    z = dbscan(D, eps, MIN_SAMPLES)
    print(f"{eps:6.2f} {len(set(z[z >= 0])):7d} {(z == -1).mean():7.1%} "
          f"{acuerdo(z, y):8.1%}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=12, edgecolors="none")
ax[0].set_title("verdad")
z_db = dbscan(D, 0.4, MIN_SAMPLES)
for a, z, t in [(ax[1], z_db, f"DBSCAN eps=0.4: {acuerdo(z_db, y):.0%}"),
                (ax[2], z_h, f"HDBSCAN: {acuerdo(z_h, y):.0%}")]:
    a.scatter(X[z < 0, 0], X[z < 0, 1], c="#aab", s=14, marker="x")
    a.scatter(X[z >= 0, 0], X[z >= 0, 1], c=z[z >= 0], cmap="viridis",
              s=12, edgecolors="none")
    a.set_title(t)
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
