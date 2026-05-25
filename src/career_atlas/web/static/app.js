// CAREER-ATLAS UI — state machine driving the 4 screens, fetch + SSE, cytoscape.

const screens = ["screen-cv", "screen-extracting", "screen-review", "screen-build", "screen-map"];
const stepFromScreen = {
  "screen-cv": 1, "screen-extracting": 1,
  "screen-review": 2,
  "screen-build": 3,
  "screen-map": 4,
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
  return r.ok ? r.json() : { has_cv_profile: false, has_postings: false, has_map: false };
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
        loadMap();
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

// --- Screen 4: Map (cytoscape) ---

async function loadMap() {
  try {
    const r = await fetch("/api/map");
    if (!r.ok) throw new Error(`map fetch ${r.status}`);
    const graph = await r.json();
    // Reveal the section FIRST so #cy has measurable dimensions; fcose
    // can't lay out into a zero-sized container.
    show("screen-map");
    // requestAnimationFrame lets the browser flush layout for the newly-
    // visible section before cytoscape measures it.
    await new Promise(requestAnimationFrame);
    renderMap(graph);
  } catch (e) {
    $("build-fail").textContent = `Could not load map: ${e.message}`;
  }
}

// Category-driven fill: hash(category) → one of 20 palette colours. Skills
// the user owns get full alpha; the rest are dimmed to recede.
const CATEGORY_PALETTE = [
  "#7FB069", "#E6A0C4", "#A89BD4", "#F4A261", "#5DA9E9",
  "#C9D6E6", "#D9544D", "#F0D879", "#79B4B7", "#B07AA1",
  "#7DCEA0", "#FFB347", "#9BB5C9", "#E59866", "#85929E",
  "#D7BDE2", "#A3E4D7", "#F8C471", "#BB8FCE", "#82E0AA",
];
const NEUTRAL_COLOUR = "#888888";

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h) + s.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

function colourForCategory(cat) {
  if (!cat) return NEUTRAL_COLOUR;
  return CATEGORY_PALETTE[hashStr(cat) % CATEGORY_PALETTE.length];
}

// t-SNE positions land in roughly [-1, 1]; scale them into a roomy pixel grid
// so cy.fit() ends at a comfortable zoom for a 700+ point semantic map.
const VIEWPORT_SPREAD = 2400;

// Cytoscape instance for the currently-rendered map. The rotate buttons
// (wired once at module load) read from this ref so they always act on the
// latest map without re-binding listeners on every renderMap().
let currentCy = null;

function rotateMap(angleDeg) {
  if (!currentCy) return;
  const a = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  currentCy.batch(() => {
    currentCy.nodes().forEach(n => {
      const p = n.position();
      n.position({
        x: p.x * cos - p.y * sin,
        y: p.x * sin + p.y * cos,
      });
    });
  });
  currentCy.fit(undefined, 40);
}

const rotL = $("rotate-left");
const rotR = $("rotate-right");
if (rotL) rotL.addEventListener("click", () => rotateMap(-5));
if (rotR) rotR.addEventListener("click", () => rotateMap(5));

function renderMap(graph) {
  const elements = [];
  const counts = new Map();
  (graph.nodes || []).forEach(n => {
    counts.set(n.id, n.count);
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
  // Edges intentionally dropped — this is a semantic map (position = meaning),
  // not a co-occurrence network. skill_map.json no longer emits them.

  const maxCount = Math.max(1, ...Array.from(counts.values()));

  const cy = cytoscape({
    container: $("cy"),
    elements,
    layout: { name: "preset" },  // positions baked in by build_map (t-SNE)
    minZoom: 0.15,
    maxZoom: 5,
    wheelSensitivity: 0.2,
    style: [
      {
        selector: "node",
        style: {
          "background-color": (ele) => colourForCategory(ele.data("category")),
          // Wider frequency variation: lower baseline (12) and bigger
          // multiplier (160). Top-N skills now dominate the field; tail
          // skills are small dots but still visible.
          "width":  (ele) => 12 + 160 * Math.sqrt(ele.data("count") / maxCount),
          "height": (ele) => 12 + 160 * Math.sqrt(ele.data("count") / maxCount),
          // Owned skills pop, market skills recede.
          "background-opacity": (ele) => ele.data("user_has") ? 1.0 : 0.3,
          "label": "",
          "border-width": 0,
        }
      },
      {
        selector: "node:selected",
        style: {
          "border-width": 16,
          "border-color": "#FFFFFF",
          "border-opacity": 1.0,
          // Always label the selected node, even if it isn't in the top-15.
          // Same styling as the persistent labels; z-index lifts it above
          // any overlapping dot in dense regions.
          "label": "data(label)",
          "color": "#E4E4E4",
          "font-size": 84,
          "text-margin-y": -16,
          "text-outline-color": "#2A2D31",
          "text-outline-width": 5,
          "z-index": 999,
        }
      },
      // Label only top-15 by count.
      {
        selector: "node[?topLabel]",
        style: {
          "label": "data(label)",
          "color": "#E4E4E4",
          "font-size": 84,
          "text-margin-y": -16,
          "text-outline-color": "#2A2D31",
          "text-outline-width": 5,
        }
      },
    ],
  });

  // Annotate top-15 by count for labeling.
  const sorted = cy.nodes().sort((a, b) => b.data("count") - a.data("count"));
  sorted.slice(0, 15).forEach(n => n.data("topLabel", true));

  // Expose for the rotate buttons.
  currentCy = cy;

  cy.fit(undefined, 40);

  // Cache the rank-by-count order for the top-N slider.
  const ranked = sorted.map(n => n.id());
  setupTopNSlider(cy, ranked);

  cy.on("tap", "node", (evt) => {
    const n = evt.target;
    showNodeInfo({
      name: n.data("id"),
      count: n.data("count"),
      user_has: n.data("user_has"),
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

function setupTopNSlider(cy, rankedNodeIds) {
  const slider = $("top-n-slider");
  const valueOut = $("top-n-value");
  const totalSpan = $("top-n-total");
  if (!slider || !valueOut || !totalSpan) return;

  const total = rankedNodeIds.length;
  totalSpan.textContent = total;
  // Cap max at the actual node count so the slider can't ask for more than exist.
  slider.max = total;
  if (parseInt(slider.value, 10) > total) slider.value = total;

  function applyTopN(n) {
    const cutoff = Math.max(1, Math.min(total, n));
    const keep = new Set(rankedNodeIds.slice(0, cutoff));
    cy.batch(() => {
      cy.nodes().forEach(node => {
        const visible = keep.has(node.id()) || node.data("user_has");
        node.style("display", visible ? "element" : "none");
      });
    });
    valueOut.textContent = cutoff;
  }

  slider.addEventListener("input", () => applyTopN(parseInt(slider.value, 10)));
  applyTopN(parseInt(slider.value, 10));
}

function showNodeInfo({ name, count, user_has }) {
  document.querySelector("#node-panel p.muted").style.display = "none";
  $("node-info").innerHTML = `
    <dt>Name</dt><dd>${escapeHtml(name)}</dd>
    <dt>Popularity</dt><dd>${count} postings</dd>
    <dt>Status</dt><dd>${user_has ? "On your CV" : "Missing"}</dd>`;
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
// URL hash router: `#map` jumps straight to the map if the artefact
// exists. Falls back to the CV screen otherwise. Hash also updates as the
// user navigates so a refresh keeps them on the same screen.

const HASH_TO_SCREEN = {
  "#cv":         "screen-cv",
  "#review":     "screen-review",
  "#build":      "screen-build",
  "#map":      "screen-map",
};

async function init() {
  const requested = HASH_TO_SCREEN[location.hash];
  if (requested === "screen-map") {
    try {
      const status = await fetch("/api/status").then(r => r.json());
      if (status.has_map) {
        await loadMap();
        return;
      }
    } catch (_) { /* fall through */ }
  }
  show(requested || "screen-cv");
}

init();
