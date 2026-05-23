# job-universe

Stage 1 of the skill-graph project: scrape ML/AI job postings from Adzuna and
JobSpy (LinkedIn / Indeed / Glassdoor) into a deduplicated, append-only parquet
store with first-/last-seen tracking. Downstream stages (skill extraction,
embedding, canonical-skill assembly, CV mapping) read from this store but are
out of scope here.

Built on [Kedro 1.4](https://docs.kedro.org).

## Setup

Requirements: Python 3.12, [uv](https://github.com/astral-sh/uv), an Adzuna
developer account.

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

## Run the pipeline

```
kedro run --pipeline=scraping
```

First run produces `data/03_primary/job_postings.parquet`. Subsequent runs
update `last_seen_at` for ids already in the store and append rows for new ids
— the file never duplicates.

## Run the tests

```
pytest tests/
```

(45 tests covering schema validation, normalization, cross-source dedup,
cumulative-merge semantics, and the Adzuna 25 req/60s rate limit.)

## Tuning

All knobs live in `conf/base/parameters/scraping.yml`:

* `scraping.adzuna.queries` — list of role-keyword queries
* `scraping.adzuna.countries` — Adzuna country codes (default `["de"]`)
* `scraping.adzuna.requests_per_minute` — throttle ceiling, default 20 (Adzuna's
  hard limit is 25)
* `scraping.jobspy.sites` — any subset of `linkedin`, `indeed`, `glassdoor`
* `scraping.jobspy.hours_old` — JobSpy freshness window

## Project layout

```
src/job_universe/
  schemas.py                   # JobPosting + _content_hash + _normalize_text
  hooks.py                     # CredentialsHook: exposes creds as a node input
  datasets.py                  # SeededParquetDataset (empty frame on missing file)
  clients/
    adzuna.py                  # rate-limited httpx client + tenacity retry
    jobspy_wrapper.py          # per-site isolated JobSpy wrapper
  pipelines/scraping/
    nodes.py                   # fetch / normalize / dedupe / merge / update
    pipeline.py                # node wiring
conf/base/
  catalog.yml                  # persisted datasets
  parameters/scraping.yml      # queries, sites, rate limits
conf/local/
  credentials.yml              # ADZUNA_APP_ID, ADZUNA_APP_KEY (gitignored)
data/
  02_intermediate/             # postings_normalized.parquet
  03_primary/                  # job_postings.parquet (cumulative)
tests/                         # 45 pytest cases
```
