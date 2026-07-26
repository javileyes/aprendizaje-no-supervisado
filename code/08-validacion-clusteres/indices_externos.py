"""
Cuando SÍ hay etiquetas de referencia: Rand, Rand ajustado e información mutua
normalizada, implementados desde cero.

El mensaje central: el índice de Rand SIN ajustar está inflado. Dos particiones
al azar ya coinciden en un montón de parejas por pura combinatoria, y el Rand
crudo lo cuenta como acierto. El ARI resta esa coincidencia esperada, de forma
que vale 0 de media cuando no hay ninguna relación.

Ejecútalo con:  python code/08-validacion-clusteres/indices_externos.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)


def contingencia(a, b):
    ka, kb = a.max() + 1, b.max() + 1
    M = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(M, (a, b), 1)
    return M


def comb2(x):
    return x * (x - 1) // 2


def rand_index(a, b):
    """Fracción de PAREJAS clasificadas igual (juntas-juntas o separadas-separadas)."""
    M = contingencia(a, b)
    n = len(a)
    suma_ij = comb2(M).sum()
    suma_i = comb2(M.sum(axis=1)).sum()
    suma_j = comb2(M.sum(axis=0)).sum()
    acuerdos = comb2(n) + 2 * suma_ij - suma_i - suma_j
    return acuerdos / comb2(n)


def ari(a, b):
    """Rand ajustado: (índice - esperado) / (máximo - esperado)."""
    M = contingencia(a, b)
    n = len(a)
    suma_ij = comb2(M).sum()
    suma_i = comb2(M.sum(axis=1)).sum()
    suma_j = comb2(M.sum(axis=0)).sum()
    esperado = suma_i * suma_j / comb2(n)
    maximo = 0.5 * (suma_i + suma_j)
    return (suma_ij - esperado) / (maximo - esperado)


def nmi(a, b):
    """Información mutua normalizada: 2 I(A;B) / (H(A) + H(B))."""
    M = contingencia(a, b).astype(float)
    n = M.sum()
    P = M / n
    pa, pb = P.sum(axis=1), P.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.nansum(np.where(P > 0, P * np.log(P / np.outer(pa, pb)), 0.0))
    Ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    Hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    return 2 * I / (Ha + Hb) if (Ha + Hb) > 0 else 1.0


# ---------------- 1) Línea base: dos particiones AL AZAR ----------------
print("1) Dos particiones aleatorias e independientes (no hay ninguna relación real)")
print(f"{'k':>4} {'Rand':>9} {'ARI':>10} {'NMI':>9}")
for k in [2, 5, 10, 20]:
    R, A, N = [], [], []
    for _ in range(200):
        a = rng.integers(0, k, 300)
        b = rng.integers(0, k, 300)
        R.append(rand_index(a, b))
        A.append(ari(a, b))
        N.append(nmi(a, b))
    print(f"{k:4d} {np.mean(R):9.4f} {np.mean(A):10.4f} {np.mean(N):9.4f}")
print("   El Rand crudo sube con k hasta rozar 1 sin que haya NADA que descubrir.")
print("   El ARI se queda pegado a 0, que es lo correcto. El NMI sube un poco:")
print("   también está sesgado al alza cuando hay muchos grupos y pocos datos.\n")

# ---------------- 2) Degradación controlada ----------------
print("2) Partimos de la verdad y estropeamos una fracción creciente de etiquetas")
verdad = np.repeat([0, 1, 2], 100)
print(f"{'% cambiado':>11} {'Rand':>9} {'ARI':>10} {'NMI':>9}")
frac, Rs, As, Ns = [], [], [], []
for p in [0.0, 0.1, 0.2, 0.35, 0.5, 0.67, 0.85, 1.0]:
    z = verdad.copy()
    idx = rng.choice(len(z), int(p * len(z)), replace=False)
    z[idx] = rng.integers(0, 3, len(idx))
    frac.append(p)
    Rs.append(rand_index(verdad, z))
    As.append(ari(verdad, z))
    Ns.append(nmi(verdad, z))
    print(f"{p:10.0%} {Rs[-1]:10.4f} {As[-1]:10.4f} {Ns[-1]:9.4f}")

# ---------------- 3) Un caso instructivo: partir un grupo en dos ----------------
print("\n3) ¿Y si la partición es CORRECTA pero más fina que la verdad?")
fina = verdad.copy()
fina[:50] = 3                      # partimos el grupo 0 en dos mitades
print(f"   verdad: 3 grupos de 100. Propuesta: 4 grupos (el primero, partido).")
print(f"   Rand {rand_index(verdad, fina):.4f} | ARI {ari(verdad, fina):.4f} "
      f"| NMI {nmi(verdad, fina):.4f}")

mal = verdad.copy()
mal[:50] = 1                       # los MISMOS 50 puntos, pero mezclados con el grupo 1
print("   Y ahora un error de verdad sobre esos mismos 50 puntos: al grupo 1.")
print(f"   Rand {rand_index(verdad, mal):.4f} | ARI {ari(verdad, mal):.4f} "
      f"| NMI {nmi(verdad, mal):.4f}")
print("   Ningún índice llega a 1 con la subdivisión, así que tampoco la aprueban.")
print(f"   Pero el error cuesta más: el ARI baja de {ari(verdad, fina):.4f} a "
      f"{ari(verdad, mal):.4f}. Los índices")
print("   distinguen refinamiento de error en grado, no en clase.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([2, 5, 10, 20],
           [np.mean([rand_index(rng.integers(0, k, 300), rng.integers(0, k, 300))
                     for _ in range(60)]) for k in [2, 5, 10, 20]], "o-", label="Rand")
ax[0].plot([2, 5, 10, 20],
           [np.mean([ari(rng.integers(0, k, 300), rng.integers(0, k, 300))
                     for _ in range(60)]) for k in [2, 5, 10, 20]], "s-", label="ARI")
ax[0].set_xlabel("número de grupos")
ax[0].set_ylabel("valor con etiquetas al azar")
ax[0].legend()
ax[0].set_title("línea base: debería ser 0")
ax[1].plot(frac, Rs, "o-", label="Rand")
ax[1].plot(frac, As, "s-", label="ARI")
ax[1].plot(frac, Ns, "^-", label="NMI")
ax[1].set_xlabel("fracción de etiquetas estropeadas")
ax[1].legend()
ax[1].set_title("degradación controlada")
plt.tight_layout()
plt.show()
