# job-universe

CV-driven job scraping for **any profession**. A local Gemma 4 model (via
Ollama) reads your CV, derives the right job-search queries and target
geographies, then scrapes Adzuna + JobSpy (LinkedIn / Indeed) into a
deduplicated, append-only Parquet store with first-/last-seen tracking.
Downstream stages (skill canonicalization, embeddings, graph construction,
gap visualization) read from this store but are out of scope here.

Built on [Kedro 1.4](https://docs.kedro.org).

## Setup

Requirements: Python 3.12, [uv](https://github.com/astral-sh/uv),
[Ollama](https://ollama.com), an Adzuna developer account.

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

| Tier   | Default Ollama tag | Hardware                       |
|--------|--------------------|--------------------------------|
| `low`  | `gemma4:e2b`       | CPU / 8 GB GPU                 |
| `mid`  | `gemma4:e4b`       | 12 GB GPU (default)            |
| `high` | `gemma4:26b`       | 24 GB GPU (MoE, 3.8B active)   |
| `max`  | `gemma4:31b`       | 32 GB+ GPU (dense)             |

Pick a tier in `conf/local/parameters/cv_extraction.yml`:
```yaml
cv_extraction:
  hardware_tier: low   # one of: low | mid | high | max
```

Custom tags / remote hosts go in the same local YAML by overriding
`model_registry.<tier>.ollama_tag` or `ollama.host`.

## Run

Drop your CV as `data/01_raw/cv/cv.md` (plain text or markdown), then:

```
kedro run --pipeline=cv_extraction    # CV → cv_profile.json + cv_derived_scraping_params.json
kedro run --pipeline=scraping         # consume the derived params → job_postings.parquet
kedro run                             # both, via __default__
```

First run produces `data/03_primary/job_postings.parquet`. Subsequent runs
update `last_seen_at` for ids already in the store and append rows for new
ids — the file never duplicates.

## What the LLM produces

Two JSON artefacts in `data/02_intermediate/`:

* `cv_profile.json` — structured `CVProfile`: skills/tools (each with kind,
  proficiency, and an evidence snippet), role titles, seniority, years of
  experience, preferred locations, summary.
* `cv_derived_scraping_params.json` — a `JobSearchTargeting` overlaid on
  the tech knobs from `params:scraping`: search queries, Adzuna country
  codes (constrained to Adzuna's 19-country list by the JSON schema),
  JobSpy `{name, country_indeed}` pairs (constrained to JobSpy's
  supported set).

## Run the tests

```
pytest tests/
```

All tests mock the LLM and HTTP layers — no Ollama or network access is
required to run the suite.

## Tuning

Two YAML files in `conf/base/parameters/`:

* `scraping.yml` — *only* technical knobs (rate limit, max pages, sites,
  hours_old). No queries, no countries, no locations — those come from
  the CV at runtime.
* `cv_extraction.yml` — `hardware_tier`, the `model_registry` mapping
  tiers → Ollama tags, Ollama host/timeout, and generation options
  (temperature, num_predict, num_ctx).

## Project layout

```
src/job_universe/
  schemas.py                       # JobPosting + CVProfile + JobSearchTargeting + identity helpers
  scraping.py                      # fetch / normalize / dedupe / append helpers (provider-agnostic)
  hooks.py                         # CredentialsHook + OllamaClientHook
  datasets.py                      # SeededParquetDataset (empty frame on missing file)
  clients/
    adzuna.py                      # rate-limited httpx client + tenacity retry
    jobspy_wrapper.py              # per-site isolated JobSpy wrapper
  llm/
    client.py                      # OllamaClient: /api/chat w/ JSON-schema format
    prompts.py                     # SKILL_EXTRACTION_SYSTEM, TARGETING_SYSTEM
  pipelines/
    cv_extraction/                 # CV → CVProfile → JobSearchTargeting
    scraping/                      # thin wiring around src/job_universe/scraping.py
conf/base/
  catalog.yml                      # persisted datasets
  parameters/scraping.yml          # tech knobs only
  parameters/cv_extraction.yml     # hardware_tier, model registry, Ollama host
conf/local/
  credentials.yml                  # ADZUNA_APP_ID, ADZUNA_APP_KEY (gitignored)
data/
  01_raw/cv/cv.md                  # the user's CV (gitignored)
  02_intermediate/                 # postings_normalized.parquet, cv_profile.json,
                                   # cv_derived_scraping_params.json
  03_primary/                      # job_postings.parquet (cumulative)
tests/                             # pytest cases
```
