// SKILL-GRAPH UI — state machine driving the 4 screens, fetch + SSE, cytoscape.

const screens = ["screen-cv", "screen-extracting", "screen-review", "screen-build", "screen-graph"];
const stepFromScreen = {
  "screen-cv": 1, "screen-extracting": 1,
  "screen-review": 2,
  "screen-build": 3,
  "screen-graph": 4,
};

function show(screenId) {
  screens.forEach(id => document.getElementById(id).classList.toggle("active", id === screenId));
  document.querySelectorAll("#stepper .step").forEach(el => {
    el.classList.toggle("active", Number(el.dataset.step) === stepFromScreen[screenId]);
  });
}

function $(id) { return document.getElementById(id); }

async function checkStatus() {
  const r = await fetch("/api/status");
  return r.ok ? r.json() : { has_cv_profile: false, has_postings: false, has_graph: false };
}

// --- Screen 1: CV ---

$("cv-next").addEventListener("click", async () => {
  const text = $("cv-input").value.trim();
  if (!text) {
    $("cv-error").textContent = "Paste a CV first.";
    return;
  }
  $("cv-error").textContent = "";
  $("cv-next").disabled = true;
  show("screen-extracting");
  try {
    const resp = await fetch("/api/cv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cv_text: text }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderReview(data);
    show("screen-review");
  } catch (e) {
    $("cv-error").textContent = e.message;
    show("screen-cv");
  } finally {
    $("cv-next").disabled = false;
  }
});

// --- Screen 2: Review ---

function renderReview({ profile, targeting }) {
  const skillsList = $("skills-list");
  skillsList.innerHTML = "";
  (profile.skills || []).forEach(s => {
    const row = document.createElement("div");
    row.className = "skill-row";
    row.innerHTML = `
      <span class="skill-name">${escapeHtml(s.name)}</span>
      <span class="badges">
        <span class="badge kind-${s.kind}">${s.kind}</span>
        <span class="badge">${s.proficiency || ""}</span>
      </span>`;
    skillsList.appendChild(row);
  });
  // The persisted shape is the merged scraping params dict:
  //   { adzuna: { queries, countries, ... }, jobspy: { queries, locations, ... } }
  // (Not the raw JobSearchTargeting shape — see pipelines/cv_extraction/nodes.py:derive_targeted_scraping_params.)
  const adzuna = targeting.adzuna || {};
  const jobspy = targeting.jobspy || {};
  const queries = Array.from(new Set([...(adzuna.queries || []), ...(jobspy.queries || [])]));
  renderChips($("queries-list"), queries);
  renderChips($("countries-list"), adzuna.countries || []);
  renderChips(
    $("locations-list"),
    (jobspy.locations || []).map(l =>
      typeof l === "string" ? l : `${l.name} (${l.country_indeed})`
    )
  );
}

function renderChips(container, items) {
  container.innerHTML = "";
  items.forEach(text => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = text;
    container.appendChild(chip);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

$("build-next").addEventListener("click", startBuild);

// --- Screen 3: Build (SSE) ---

let currentEventSource = null;

async function startBuild() {
  $("build-error").textContent = "";
  $("build-next").disabled = true;
  $("counter-postings").textContent = "0";
  $("counter-skills").textContent = "0";
  $("build-status").textContent = "Scraping job boards…";
  $("build-fail").textContent = "";
  show("screen-build");
  try {
    const resp = await fetch("/api/build", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const { job_id } = await resp.json();
    connectEvents(job_id);
  } catch (e) {
    $("build-error").textContent = e.message;
    show("screen-review");
  } finally {
    $("build-next").disabled = false;
  }
}

function connectEvents(jobId) {
  if (currentEventSource) currentEventSource.close();
  const es = new EventSource(`/api/build/events?job_id=${encodeURIComponent(jobId)}`);
  currentEventSource = es;
  es.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch { return; }
    switch (event.type) {
      case "postings_count":
        $("counter-postings").textContent = event.value;
        break;
      case "skills_count":
        $("counter-skills").textContent = event.value;
        $("build-status").textContent = "Extracting skills…";
        break;
      case "done":
        es.close();
        currentEventSource = null;
        loadGraph();
        break;
      case "error":
        es.close();
        currentEventSource = null;
        $("build-fail").textContent = event.value || "Build failed.";
        break;
    }
  };
  es.onerror = () => {
    es.close();
    currentEventSource = null;
    $("build-fail").textContent = "Lost connection to the server.";
  };
}

// --- Screen 4: Graph (cytoscape) ---

async function loadGraph() {
  try {
    const r = await fetch("/api/graph");
    if (!r.ok) throw new Error(`graph fetch ${r.status}`);
    const graph = await r.json();
    renderGraph(graph);
    show("screen-graph");
  } catch (e) {
    $("build-fail").textContent = `Could not load graph: ${e.message}`;
  }
}

// 20-tab style palette tuned for the dark background. Index = stable hash mod len.
const CATEGORY_PALETTE = [
  "#7FB069", "#E6A0C4", "#A89BD4", "#F4A261", "#5DA9E9",
  "#C9D6E6", "#D9544D", "#F0D879", "#79B4B7", "#B07AA1",
  "#7DCEA0", "#FFB347", "#9BB5C9", "#E59866", "#85929E",
  "#D7BDE2", "#A3E4D7", "#F8C471", "#BB8FCE", "#82E0AA",
];

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h) + s.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

function colourForCategory(cat) {
  if (!cat) return "#888888";
  return CATEGORY_PALETTE[hashStr(cat) % CATEGORY_PALETTE.length];
}

// Spring positions are unit-ish [-1, 1]; spread them across the viewport.
const VIEWPORT_SPREAD = 900;

function renderGraph(graph) {
  const elements = [];
  const counts = new Map();
  const categories = new Set();
  (graph.nodes || []).forEach(n => {
    counts.set(n.id, n.count);
    if (n.category) categories.add(n.category);
    elements.push({
      data: {
        id: n.id,
        label: n.id,
        count: n.count,
        user_has: !!n.user_has,
        category: n.category || null,
      },
      position: {
        x: (n.position?.x || 0) * VIEWPORT_SPREAD,
        y: (n.position?.y || 0) * VIEWPORT_SPREAD,
      },
    });
  });
  (graph.edges || []).forEach(e => {
    elements.push({
      data: {
        id: `${e.source}__${e.target}`,
        source: e.source,
        target: e.target,
        weight: e.weight,
        cooccurrence: e.cooccurrence,
      }
    });
  });

  const maxCount = Math.max(1, ...Array.from(counts.values()));

  const cy = cytoscape({
    container: $("cy"),
    elements,
    layout: { name: "preset" },  // positions baked into the JSON by build_graph
    minZoom: 0.2,
    maxZoom: 4,
    wheelSensitivity: 0.2,
    style: [
      {
        selector: "node",
        style: {
          "background-color": (ele) => colourForCategory(ele.data("category")),
          "width":  (ele) => 10 + 38 * Math.sqrt(ele.data("count") / maxCount),
          "height": (ele) => 10 + 38 * Math.sqrt(ele.data("count") / maxCount),
          "label": "",
          "border-width": (ele) => ele.data("user_has") ? 3 : 0,
          "border-color": "#3F704D",
        }
      },
      {
        selector: "node:selected",
        style: {
          "border-width": 4,
          "border-color": "#E4E4E4",
        }
      },
      {
        selector: "edge",
        style: {
          "line-color": "#4A4D51",
          "opacity": 0.35,
          "width": (ele) => 0.6 + 1.5 * Math.min(2.5, (ele.data("weight") || 0)),
          "curve-style": "haystack",
        }
      },
      // Label only top-30 by count when nothing is selected.
      {
        selector: "node[?topLabel]",
        style: {
          "label": "data(label)",
          "color": "#E4E4E4",
          "font-size": 10,
          "text-margin-y": -4,
          "text-outline-color": "#2A2D31",
          "text-outline-width": 2,
        }
      },
    ],
  });

  // Annotate top-30 by count for labeling.
  const sorted = cy.nodes().sort((a, b) => b.data("count") - a.data("count"));
  sorted.slice(0, 30).forEach(n => n.data("topLabel", true));

  cy.fit(undefined, 40);

  renderCategoryLegend(Array.from(categories).sort());

  cy.on("tap", "node", (evt) => {
    const n = evt.target;
    showNodeInfo({
      name: n.data("id"),
      count: n.data("count"),
      user_has: n.data("user_has"),
      category: n.data("category"),
    });
    loadAndRenderPostings(n.data("id"));
  });

  cy.on("tap", (evt) => {
    if (evt.target === cy) {
      $("node-info").innerHTML = "";
      $("postings-list").innerHTML = "";
      $("postings-heading").textContent = "";
      document.querySelector("#node-panel p.muted").style.display = "block";
    }
  });
}

function renderCategoryLegend(categories) {
  const legend = $("category-legend");
  if (!legend) return;
  legend.innerHTML = "";
  categories.forEach(cat => {
    const chip = document.createElement("span");
    chip.className = "legend-chip";
    chip.innerHTML = `<span class="dot" style="background:${colourForCategory(cat)}"></span>${escapeHtml(cat)}`;
    legend.appendChild(chip);
  });
}

function showNodeInfo({ name, count, user_has, category }) {
  document.querySelector("#node-panel p.muted").style.display = "none";
  const catColour = colourForCategory(category);
  $("node-info").innerHTML = `
    <dt>Name</dt><dd>${escapeHtml(name)}</dd>
    <dt>Popularity</dt><dd>${count} postings</dd>
    <dt>Status</dt><dd>${user_has ? "You already have this" : "Gap"}</dd>
    <dt>Category</dt><dd><span class="dot" style="background:${catColour}"></span>${escapeHtml(category || "—")}</dd>`;
}

async function loadAndRenderPostings(name) {
  $("postings-heading").textContent = "Loading postings…";
  $("postings-list").innerHTML = "";
  try {
    const r = await fetch(`/api/skill/${encodeURIComponent(name)}/postings?limit=50`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    const shown = data.postings.length;
    const total = data.total;
    $("postings-heading").textContent =
      total === 0
        ? "No postings reference this skill"
        : `Postings (${shown} shown · ${total} total)`;
    const ol = $("postings-list");
    ol.innerHTML = "";
    data.postings.forEach(p => {
      const li = document.createElement("li");
      const title = escapeHtml(p.title || "(no title)");
      const sub = [p.company, p.location, p.source].filter(Boolean).map(escapeHtml).join(" · ");
      li.innerHTML = p.url
        ? `<a href="${escapeHtml(p.url)}" target="_blank" rel="noopener">${title}</a><span class="muted">${sub}</span>`
        : `<span>${title}</span><span class="muted">${sub}</span>`;
      ol.appendChild(li);
    });
  } catch (e) {
    $("postings-heading").textContent = `Could not load postings: ${e.message}`;
  }
}

// --- Init ---

(async function init() {
  show("screen-cv");
})();
