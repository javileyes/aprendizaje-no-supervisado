# 🧩 Aprendizaje no supervisado

de la intuición a las matemáticas

Un manual web **interactivo**: explicaciones **de la intuición a las matemáticas**, con
ejemplos en Python que puedes **ejecutar en el propio navegador** (gracias a
[Pyodide](https://pyodide.org)) o descargar y correr en tu **terminal**.

---

## 🚀 Cómo abrir el manual (el sitio web)

Sitio **estático** (HTML + CSS + JS, sin backend). Recomendado:

```bash
python3 -m http.server 8000
# Abre http://localhost:8000 en tu navegador
```

> ℹ️ La primera vez que pulses **«▶ Ejecutar»** en un ejemplo, el navegador descarga el
> intérprete de Python (~10 MB). A partir de ahí es instantáneo. Necesita conexión a
> internet para esa primera descarga y para las librerías.

---

## ✍️ El manual es tuyo: tocar el código y anotar

No es una web para mirar. Mientras lees puedes cambiarla, y lo que cambies se queda.

- **El código se edita.** Cualquier bloque de ejemplo es editable: pincha dentro, cambia
  un número —un `alpha`, una semilla, el número de pasos— y vuelve a pulsar **«▶ Ejecutar»**.
  `Tab` indenta; `Esc` o `Mayús+Tab` sacan el foco del bloque. Es lo que piden los recuadros
  **🧪 Experimenta**: se hacen ahí mismo, sin descargar nada.
- **«↺ Original» por bloque.** Aparece en cuanto tocas un ejemplo y lo devuelve a como
  estaba. Solo afecta a ese bloque.
- **Puedes anotar casi todo:** párrafos, fórmulas, títulos de sección, puntos de lista,
  recuadros completos y filas de tabla. Pasa el ratón por encima y sale un **«✎ crear nota»**;
  lo que ya tiene nota lleva un **📝** en el margen, y al posar el cursor se ve el texto.
- **Se guarda solo, en tu navegador.** El código editado y las notas sobreviven a recargar
  la página y a cambiar de capítulo. No se envía nada a ningún servidor.
- **«💾 Mis cambios»** (en la barra lateral) baja **todo** —el código editado de todos los
  capítulos y todas tus notas— a un único fichero `.json`, y lo vuelve a importar. Sirve
  para llevártelo a otro ordenador, a otro navegador, o para guardar varias tandas de pruebas.

Nada de esto se ancla a la posición: cada ejemplo se identifica por su **nombre de fichero**,
cada sección por su **id**, cada fórmula por su **LaTeX**, y los párrafos, listas, recuadros
y filas por una huella de su texto. Si el manual se actualiza —párrafos nuevos, secciones
reordenadas—, tus notas siguen donde las dejaste. Solo se desengancha una nota si cambia
el texto concreto que anotaste, y ni siquiera entonces se pierde: sigue listada en
«Mis cambios», avisando de que el párrafo ha cambiado.

---

## 🐍 Cómo ejecutar los ejemplos en tu terminal (entorno virtual)

### 1. Requisito de versión de Python

Usa **Python 3.10 – 3.13**. Algunas librerías pesadas (p. ej. PyTorch) aún no publican
ruedas para las versiones más nuevas; si tu `python3` es 3.14+, crea el entorno con una
versión soportada: `python3.13 -m venv .venv`.

### 2. Crea el entorno virtual e instala las dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux   (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecuta cualquier ejemplo

```bash
python code/01-aprender-sin-etiquetas/ejemplo_1.py
# sin ventana de gráficas (servidores):  MPLBACKEND=Agg python code/...
```

Los ejemplos `*.py` en NumPy son los que también corren en el navegador; los que usan
librerías pesadas se ejecutan en la terminal.

---

## 🗺️ Estructura del proyecto

```
aprendizaje-no-supervisado/
├── index.html                 # Portada + índice completo
├── README.md
├── requirements.txt
├── assets/                    # CSS y JS (diseño, navegación, runner de Pyodide)
├── chapters/                  # Un archivo HTML por capítulo
└── code/                      # Los mismos ejemplos como scripts .py por capítulo
```

---

## 📚 Índice

**Parte I · Fundamentos** — 1) Aprender sin etiquetas · 2) Geometría, distancias y la maldición de la dimensionalidad
**Parte II · Clústering** — 3) k-means: cuantización vectorial · 4) Mixturas gaussianas y el algoritmo EM · 5) Clústering jerárquico y dendrogramas · 6) Clústering por densidad: DBSCAN y HDBSCAN · 7) Clústering espectral · 8) ¿Cuántos clústeres? Validar sin etiquetas
**Parte III · Análisis de componentes** — 9) PCA: la dirección de máxima varianza · 10) SVD, Eckart-Young y PCA probabilístico · 11) NMF e ICA: más allá de la varianza · 12) Variedades: kernel PCA, t-SNE y UMAP
**Parte IV · No supervisado profundo** — 13) Autoencoders: compresión aprendida · 14) Autoencoders variacionales · 15) Deep clustering: DEC, IDEC y el colapso · 16) Autosupervisado y contrastivo: SimCLR e InfoNCE
**Parte V · La frontera con lo supervisado** — 17) Semisupervisado: pocas etiquetas, muchos datos · 18) Detección de anomalías

Empieza por [`chapters/01-aprender-sin-etiquetas.html`](chapters/01-aprender-sin-etiquetas.html).

---

## 🛠️ Tecnología

- **Sin frameworks ni build**: HTML/CSS/JS puro.
- **Matemáticas**: [MathJax 3](https://www.mathjax.org/).
- **Python en el navegador**: [Pyodide](https://pyodide.org) (WebAssembly).

Hecho para aprender haciendo. 🧠
