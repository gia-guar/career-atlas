# skillgraph

[![Built on Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

<center>

<img src="assets/logo.png" alt="skill-graph-logo" style="width: 300px; height: 300px;">

## What am I looking at?

*Skillgraph* helps you identify the strenghts and gaps in your career experience
by matching them against what recruiters are asking in recent job openings. This allows you
to target specific positions that ask for rare and nieche skills, and lets you work to
grow highly requested skills.

*Powered by*:<br><br>

<p float="left">

<img src="assets/linkedinlogo.png" style="width: 120px; height: 30px;"> &nbsp; <img src="assets/Indeed_logo.svg.png" style="width: 120px; height: 30px;">
</p>


<p float="left">
<img src="assets/Gemma4logo.png" style="width: 80px; height: 80px; vertical-align: middle;">
<img src="assets/gemma-text.png" style="width: 230px; height: 50px; vertical-align: middle;">
</p>

## How it works?

First, job are scraped based on your experience (e.g. your CV): A local Gemma 4 model (via Ollama) reads your experience, derives compatible
job-search queries, scrapes with `Adzuna` + `JobSpy`
(LinkedIn / Indeed) into a Parquet store. Then extracts per-posting skills and renders
a popularity-weighted skill graph with the skills and tools you master
highlighted in green.


## Setup

Requirements: Python 3.12, [uv](https://github.com/astral-sh/uv),
[Ollama](https://ollama.com), an Adzuna account (free).

```
uv pip install -e ".[dev]"
```

Copy the credentials template and fill in your Adzuna keys
([free at adzuna.com/developer](https://developer.adzuna.com/)):

```
cp credentials.template.yml conf/local/credentials.yml
# then edit conf/local/credentials.yml
```

`conf/local/credentials.yml` is gitignored. Never commit it.

Pull the Gemma 4 tag matching your hardware (one-time):

```
ollama pull gemma4:e2b    # smallest; works on CPU / 8 GB GPU
```
<center>

| Tier   | Default Ollama tag | Hardware                       |
|--------|--------------------|--------------------------------|
| `low`  | `gemma4:e2b`       | CPU / 8 GB GPU                 |
| `mid`  | `gemma4:e4b`       | 12 GB GPU (default)            |
| `high` | `gemma4:26b`       | 24 GB GPU (MoE, 3.8B active)   |
| `max`  | `gemma4:31b`       | 32 GB+ GPU (dense)             |

</center>

Pick a tier in `conf/local/parameters/cv_extraction.yml`:
<div style="text-align: left;">

```yaml
cv_extraction:
  hardware_tier: low   # one of: low | mid | high | max
```
</div>

Custom tags / remote hosts go in the same local YAML by overriding
`model_registry.<tier>.ollama_tag` or `ollama.host`.

## Run

### Web UI

Launch the local browser app, paste your CV, watch live counters during
scraping + extraction, and explore the interactive skill graph:

<div style="text-align: left;">

```
job-universe-ui
# → open http://127.0.0.1:8000/
```

</div>

Walks the same three pipelines under the hood (`cv_extraction` → `scraping`
→ `skill_graph`), reading and writing the exact same files as the CLI.

### CLI

Drop your CV as `data/01_raw/cv/cv.md` (plain text or markdown), then:

<div style="text-align: left;">

```
kedro run --pipeline=cv_extraction    # CV → cv_profile.json + cv_derived_scraping_params.json
kedro run --pipeline=scraping         # consume the derived params → job_postings.parquet
kedro run --pipeline=skill_graph      # per-posting skills → graph PNG
kedro run                             # all three, via __default__
```
</div>

First run produces `data/03_primary/job_postings.parquet`. Subsequent runs
update `last_seen_at` for ids already in the store and append rows for new
ids — the file never duplicates. `skill_graph` caches per-posting skill
extractions in `data/02_intermediate/posting_skills.parquet` so re-runs
only LLM-call newly-scraped postings.

## What the LLM produces

Intermediate JSON artefacts in `data/02_intermediate/`:

* `cv_profile.json` — structured `CVProfile`: skills/tools (each with kind,
  proficiency, and an evidence snippet), role titles, seniority, years of
  experience, preferred locations, summary.
* `cv_derived_scraping_params.json` — a `JobSearchTargeting` overlaid on
  the tech knobs from `params:scraping`: search queries, Adzuna country
  codes (constrained to Adzuna's 19-country list by the JSON schema),
  JobSpy `{name, country_indeed}` pairs (constrained to JobSpy's
  supported set).
* `posting_skills.parquet` — long-format cache of per-posting skill
  extractions (`posting_id, name, kind`). `kind` is one of `skill`,
  `tool`, or `requirement`.
* `canonical_skill_map.json` — `{raw_normalized_name: canonical_name}`
  produced by sentence-transformer embeddings + agglomerative clustering.
* `skill_graph.json` — filtered node/edge graph with PMI weights and
  per-node `user_has` flags.

And in `data/08_reporting/`:

* `skill_graph.png` — the visualization (black background, grey nodes
  sized by frequency, your skills in `#3F704D`, top-30 labeled).
* `skill_graph_nodes.csv` — full ranked node list for offline inspection.

## Run the tests

```
pytest tests/
```

All tests mock the LLM and HTTP layers — no Ollama or network access is
required to run the suite.

## Tuning

Three YAML files in `conf/base/parameters/`:

* `scraping.yml` — *only* technical knobs (rate limit, max pages, sites,
  hours_old). No queries, no countries, no locations — those come from
  the CV at runtime.
* `cv_extraction.yml` — `hardware_tier`, the `model_registry` mapping
  tiers → Ollama tags, Ollama host/timeout, and generation options
  (temperature, num_predict, num_ctx). To broaden geographic scope (e.g.
  Europe-wide instead of the LLM's 1-4 country pick), drop a
  `targeting_overrides` block into `conf/local/parameters/cv_extraction.yml`
  — a commented-out template lives in the base config.
* `skill_graph.yml` — embedding model id (default
  `nomic-ai/nomic-embed-text-v1.5`), clustering distance threshold,
  graph filters (`min_node_count`, `min_cooccurrence`, `min_pmi`), and
  visualization style. The Stage-3 LLM tier is read from
  `cv_extraction.yml` — one source of truth.
</center>

## Project layout

```
src/job_universe/
  schemas.py                       # JobPosting + CVProfile + JobSearchTargeting + PostingSkills + identity helpers
  scraping.py                      # fetch / normalize / dedupe / append helpers (provider-agnostic)
  canonicalize.py                  # embedding-driven skill clustering + canonical-name selection
  skill_graph.py                   # frequency, co-occurrence, PMI, matplotlib rendering
  hooks.py                         # CredentialsHook + OllamaClientHook + SkillEmbedderHook + ProgressHook
  datasets.py                      # SeededParquetDataset (empty frame on missing file)
  web/                             # optional FastAPI UI (job-universe-ui)
    app.py                         # endpoints: /api/cv, /api/build (SSE), /api/graph
    runner.py                      # threaded KedroSession driver
    progress.py                    # ProgressEmitter (thread-safe, asyncio-queue-backed)
    static/                        # index.html + style.css + app.js + cytoscape
  clients/
    adzuna.py                      # rate-limited httpx client + tenacity retry
    jobspy_wrapper.py              # per-site isolated JobSpy wrapper
  llm/
    client.py                      # OllamaClient: /api/chat w/ JSON-schema format
    prompts.py                     # SKILL_EXTRACTION_SYSTEM, TARGETING_SYSTEM, POSTING_SKILL_EXTRACTION_SYSTEM
  pipelines/
    cv_extraction/                 # CV → CVProfile → JobSearchTargeting
    scraping/                      # thin wiring around src/job_universe/scraping.py
    skill_graph/                   # per-posting skills → canonicalize → graph → PNG
conf/base/
  catalog.yml                      # persisted datasets
  parameters/scraping.yml          # tech knobs only
  parameters/cv_extraction.yml     # hardware_tier, model registry, Ollama host
  parameters/skill_graph.yml       # embedder, distance threshold, graph filters, viz style
conf/local/
  credentials.yml                  # ADZUNA_APP_ID, ADZUNA_APP_KEY (gitignored)
data/
  01_raw/cv/cv.md                  # the user's CV (gitignored)
  02_intermediate/                 # postings_normalized.parquet, cv_profile.json,
                                   # cv_derived_scraping_params.json,
                                   # posting_skills.parquet, canonical_skill_map.json,
                                   # skill_graph.json
  03_primary/                      # job_postings.parquet (cumulative)
  08_reporting/                    # skill_graph.png, skill_graph_nodes.csv
tests/                             # pytest cases
```
