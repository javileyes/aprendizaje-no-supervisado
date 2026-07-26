/* =========================================================================
   nav.js — Fuente única de verdad del índice del manual.
   Construye: barra lateral, navegación anterior/siguiente e índice de portada.
   (Generado por la skill crear-manual; los datos se inyectan al hacer scaffold.)
   ========================================================================= */
(function () {
  "use strict";

  const PARTS = [
    {
        "id": "I",
        "title": "Parte I · Fundamentos"
    },
    {
        "id": "II",
        "title": "Parte II · Clústering"
    },
    {
        "id": "III",
        "title": "Parte III · Análisis de componentes"
    },
    {
        "id": "IV",
        "title": "Parte IV · No supervisado profundo"
    },
    {
        "id": "V",
        "title": "Parte V · La frontera con lo supervisado"
    }
];

  const CH = [
    {
        "n": 1,
        "part": "I",
        "slug": "01-aprender-sin-etiquetas",
        "title": "Aprender sin etiquetas",
        "desc": "Qué significa aprender sin supervisión, las tres tareas canónicas (densidad, estructura, representación) y por qué no hay una respuesta correcta."
    },
    {
        "n": 2,
        "part": "I",
        "slug": "02-geometria-y-dimensionalidad",
        "title": "Geometría, distancias y la maldición de la dimensionalidad",
        "desc": "Métricas y similitudes, escalado, y la demostración de por qué en dimensión alta todas las distancias se parecen."
    },
    {
        "n": 3,
        "part": "II",
        "slug": "03-kmeans",
        "title": "k-means: cuantización vectorial",
        "desc": "La inercia como objetivo, Lloyd como descenso por bloques de coordenadas, la demostración de convergencia monótona y k-means++."
    },
    {
        "n": 4,
        "part": "II",
        "slug": "04-gmm-y-em",
        "title": "Mixturas gaussianas y el algoritmo EM",
        "desc": "De la asignación dura a la blanda: el ELBO por la desigualdad de Jensen, los pasos E y M derivados, y k-means como caso límite."
    },
    {
        "n": 5,
        "part": "II",
        "slug": "05-clustering-jerarquico",
        "title": "Clústering jerárquico y dendrogramas",
        "desc": "Enlaces simple, completo, medio y de Ward; la fórmula de Lance-Williams que los unifica; cómo se lee y se corta un dendrograma."
    },
    {
        "n": 6,
        "part": "II",
        "slug": "06-densidad-dbscan",
        "title": "Clústering por densidad: DBSCAN y HDBSCAN",
        "desc": "Clústeres de forma arbitraria y ruido explícito: alcanzabilidad por densidad, distancia mutua de alcance y jerarquía de estabilidad."
    },
    {
        "n": 7,
        "part": "II",
        "slug": "07-clustering-espectral",
        "title": "Clústering espectral",
        "desc": "El laplaciano del grafo, el corte normalizado, la relajación de Rayleigh y por qué los autovectores resuelven un problema combinatorio."
    },
    {
        "n": 8,
        "part": "II",
        "slug": "08-validacion-clusteres",
        "title": "¿Cuántos clústeres? Validar sin etiquetas",
        "desc": "Silueta, gap statistic, Calinski-Harabasz, estabilidad por remuestreo, y ARI/NMI cuando sí hay etiquetas de referencia."
    },
    {
        "n": 9,
        "part": "III",
        "slug": "09-pca",
        "title": "PCA: la dirección de máxima varianza",
        "desc": "Derivación por multiplicadores de Lagrange, la doble lectura (máxima varianza = mínimo error de reconstrucción) y la varianza explicada."
    },
    {
        "n": 10,
        "part": "III",
        "slug": "10-svd-y-ppca",
        "title": "SVD, Eckart-Young y PCA probabilístico",
        "desc": "La descomposición en valores singulares como teorema central, la mejor aproximación de rango bajo y PCA como modelo generativo con EM."
    },
    {
        "n": 11,
        "part": "III",
        "slug": "11-nmf-e-ica",
        "title": "NMF e ICA: más allá de la varianza",
        "desc": "Factorización no negativa y partes-de-un-todo con las reglas de Lee-Seung; independencia y no-gaussianidad con FastICA."
    },
    {
        "n": 12,
        "part": "III",
        "slug": "12-manifold-tsne-umap",
        "title": "Variedades: kernel PCA, t-SNE y UMAP",
        "desc": "Cuando la estructura no es lineal: el truco del kernel, MDS e Isomap, el gradiente de t-SNE derivado y la intuición de UMAP."
    },
    {
        "n": 13,
        "part": "IV",
        "slug": "13-autoencoders",
        "title": "Autoencoders: compresión aprendida",
        "desc": "Del cuello de botella lineal (que reencuentra PCA) al no lineal; backpropagation desde cero, denoising y autoencoders dispersos."
    },
    {
        "n": 14,
        "part": "IV",
        "slug": "14-vae",
        "title": "Autoencoders variacionales",
        "desc": "El ELBO derivado desde la verosimilitud, el truco de reparametrización, la KL en forma cerrada y qué significa realmente el espacio latente."
    },
    {
        "n": 15,
        "part": "IV",
        "slug": "15-deep-clustering",
        "title": "Deep clustering: DEC, IDEC y el colapso",
        "desc": "Agrupar en el espacio latente y aprenderlo a la vez: la distribución objetivo autoentrenada, por qué colapsa y cómo se evita."
    },
    {
        "n": 16,
        "part": "IV",
        "slug": "16-autosupervisado-contrastivo",
        "title": "Autosupervisado y contrastivo: SimCLR e InfoNCE",
        "desc": "Fabricar la supervisión desde los propios datos: aumentaciones, InfoNCE como cota inferior de la información mutua, colapso y BYOL."
    },
    {
        "n": 17,
        "part": "V",
        "slug": "17-semisupervisado",
        "title": "Semisupervisado: pocas etiquetas, muchos datos",
        "desc": "La hipótesis del clúster, propagación de etiquetas en grafos con solución cerrada, autoentrenamiento, regularización de consistencia y FixMatch."
    },
    {
        "n": 18,
        "part": "V",
        "slug": "18-deteccion-anomalias",
        "title": "Detección de anomalías",
        "desc": "Densidad, distancia y aislamiento: LOF, Isolation Forest, la envolvente gaussiana y el error de reconstrucción como puntuación."
    }
];

  window.RL_CH = CH;
  window.RL_PARTS = PARTS;

  // ¿Estamos en la raíz o dentro de /chapters/?
  const inChapters = location.pathname.includes("/chapters/");
  const chHref = (slug) => (inChapters ? slug + ".html" : "chapters/" + slug + ".html");
  const homeHref = inChapters ? "../index.html" : "index.html";

  function buildSidebar() {
    const sb = document.getElementById("sidebar");
    if (!sb) return;
    let html = `
      <div class="brand">
        <a href="${homeHref}">
          <span class="logo"><span class="dot">🧩</span> No supervisado</span>
        </a>
        <span class="tagline">de la intuición a las matemáticas</span>
      </div>
      <div class="sidebar-tools">
        <button class="theme-btn" data-theme-btn>🌙 <span>Oscuro</span></button>
      </div>
      <nav class="nav">
        <a href="${homeHref}"><span class="num">★</span> Portada e índice</a>`;
    PARTS.forEach((p) => {
      html += `<div class="nav-part">${p.title}</div>`;
      CH.filter((c) => c.part === p.id).forEach((c) => {
        html += `<a href="${chHref(c.slug)}"><span class="num">${c.n}</span> ${c.title}</a>`;
      });
    });
    html += `</nav>`;
    sb.innerHTML = html;
  }

  function buildChapterNav() {
    const holder = document.getElementById("chapter-nav");
    if (!holder) return;
    const cur = parseInt(document.body.dataset.chapter || "0", 10);
    const prev = CH.find((c) => c.n === cur - 1);
    const next = CH.find((c) => c.n === cur + 1);
    let html = "";
    if (prev) html += `<a class="prev" href="${chHref(prev.slug)}"><div class="cn-label">← Anterior</div><div class="cn-title">${prev.n}. ${prev.title}</div></a>`;
    else html += `<a class="prev" href="${homeHref}"><div class="cn-label">←</div><div class="cn-title">Portada e índice</div></a>`;
    if (next) html += `<a class="next" href="${chHref(next.slug)}"><div class="cn-label">Siguiente →</div><div class="cn-title">${next.n}. ${next.title}</div></a>`;
    holder.innerHTML = html;
  }

  /* Índice interno del capítulo ("En este capítulo").
     Se genera solo, a partir de los <h2 id> del contenido, así que ningún capítulo
     necesita mantenerlo a mano. Se salta las páginas cortas y la portada. */
  function buildChapterTOC() {
    const content = document.querySelector(".content");
    if (!content || !document.body.dataset.chapter || document.body.dataset.chapter === "0") return;

    const hs = [...content.querySelectorAll("h2[id]")];
    if (hs.length < 4) return; // con pocas secciones el índice estorba más que ayuda

    const nav = document.createElement("nav");
    nav.className = "chapter-toc";
    nav.setAttribute("aria-label", "Índice del capítulo");
    nav.innerHTML =
      `<div class="ct-title">En este capítulo</div><ol class="ct-list">` +
      hs.map((h) => {
        // El texto del título puede llevar $LaTeX$; para el índice lo dejamos tal cual
        // (MathJax lo renderiza también aquí) pero quitamos etiquetas sueltas.
        const t = h.innerHTML.replace(/<a\b[^>]*>|<\/a>/g, "");
        return `<li><a href="#${h.id}">${t}</a></li>`;
      }).join("") +
      `</ol>`;

    const lead = content.querySelector(".lead");
    (lead || content.querySelector("h1")).insertAdjacentElement("afterend", nav);

    // Marca la sección que se está leyendo.
    const links = new Map(hs.map((h) => [h.id, nav.querySelector(`a[href="#${CSS.escape(h.id)}"]`)]));
    let activa = null;
    const marcar = (id) => {
      if (id === activa) return;
      if (activa && links.get(activa)) links.get(activa).classList.remove("active");
      activa = id;
      if (links.get(id)) links.get(id).classList.add("active");
    };
    if ("IntersectionObserver" in window) {
      const vistos = new Set();
      const obs = new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => (e.isIntersecting ? vistos.add(e.target.id) : vistos.delete(e.target.id)));
          // la primera sección visible en orden de documento manda
          const actual = hs.find((h) => vistos.has(h.id));
          if (actual) marcar(actual.id);
        },
        { rootMargin: "-80px 0px -70% 0px" }
      );
      hs.forEach((h) => obs.observe(h));
    }
  }

  function buildIndexTOC() {
    const holder = document.getElementById("index-toc");
    if (!holder) return;
    let html = "";
    PARTS.forEach((p) => {
      html += `<div class="toc-part"><div class="toc-part-title">${p.title}</div><div class="toc-grid">`;
      CH.filter((c) => c.part === p.id).forEach((c) => {
        html += `<a class="toc-card" href="${chHref(c.slug)}">
          <span class="n">${c.n}</span>
          <span><span class="tc-title">${c.title}</span><span class="tc-desc">${c.desc}</span></span>
        </a>`;
      });
      html += `</div></div>`;
    });
    holder.innerHTML = html;
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildSidebar();
    buildChapterNav();
    buildChapterTOC();
    buildIndexTOC();
  });
})();
