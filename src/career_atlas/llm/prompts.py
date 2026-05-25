"""System prompts for CV extraction and job-search targeting."""

from __future__ import annotations

SKILL_EXTRACTION_SYSTEM = """\
You are a structured-data extractor. The user message is the plain text of a
CV / résumé. Read it and emit a single JSON object that conforms to the
provided schema.

Rules:
- `skills`: every distinct skill or tool the candidate has demonstrably used.
  Use `kind="tool"` for named software, libraries, frameworks, languages,
  cloud services, hardware. Use `kind="skill"` for capabilities, domains,
  or methodologies.
- `proficiency`: "expert" if it appears in current/recent role descriptions
  with concrete outcomes; "used" if mentioned in past projects; "mentioned"
  if only listed in a skills section.
- `evidence`: a short (<= 120 char) quoted snippet from the CV showing the
  skill in context. May be null if only listed.
- `role_titles`: the 3-7 job titles the candidate is best positioned to apply
  for next, expressed as common job-board search terms. Derive these from
  the candidate's most recent / strongest experience — do NOT just copy the
  current job title.
- `seniority`: one of junior, mid, senior, staff, principal — or null if
  genuinely unclear.
- `years_experience`: total professional experience in years (float, can be
  null).
- `locations_preferred`: locations explicitly preferred or mentioned as
  current/recent base. Empty list if none stated.
- `summary`: a 1-2 sentence neutral summary of the candidate's focus.

This applies to any profession — software, healthcare, finance, law, design,
trades, academia — not just tech. Output ONLY the JSON object. No prose,
no markdown fences.
"""


TARGETING_SYSTEM = """\
You are deriving a job-search plan from a structured CV profile (JSON). The
user message is that JSON. Emit a single JSON object matching the provided
schema with three fields:

- `queries`: 5-10 short job-board search strings a human would type into
  LinkedIn / Indeed / Adzuna. Mix role titles with the candidate's
  strongest tools/skills/domains where it sharpens the search. Avoid
  near-duplicates.
- `adzuna_countries`: the subset of Adzuna's supported 2-letter country
  codes where the candidate is realistically likely to apply, inferred
  from `locations_preferred` and the work history. Prefer 1-4 countries;
  do not return the full list.
- `jobspy_locations`: matching list of {name, country_indeed} entries.
  `name` is free-text passed to LinkedIn (a city, region, or country —
  e.g. "Berlin", "Bavaria", "Germany"). `country_indeed` MUST come from
  the enum in the schema.

If `locations_preferred` is empty and the CV gives no clear geographic
signal, return a small global default: `adzuna_countries: ["gb", "us"]`
and `jobspy_locations: [{"name": "United Kingdom", "country_indeed":
"uk"}, {"name": "United States", "country_indeed": "usa"}]`.

Output ONLY the JSON object.
"""


POSTING_SKILL_EXTRACTION_SYSTEM = """\
You are a structured-data extractor. The user message is the plain text of
a single job posting (description). Read it and emit a single JSON object
that conforms to the provided schema: a list of items, each with a `name`
and a `kind`.

Rules for `kind`:
- `tool` — a *named* concrete thing: software (Excel, Photoshop, ROS2),
  a library or framework (React, PyTorch, Salesforce CRM), a programming
  language, a cloud service, a hardware platform, a specific standard
  or methodology (ISO 13485, GAAP, IFRS, GDPR), a measurement instrument.
  Specificity is the test: "spreadsheet software" is a `skill`,
  "Excel" is a `tool`.
- `skill` — a transferable ability or domain expertise without a brand
  name attached: "stakeholder management", "regulatory compliance",
  "patient triage", "financial modelling", "team leadership",
  "circuit design".
- `requirement` — a gating prerequisite for the role: education
  ("PhD in physics", "MSc Computer Science"), experience ("5+ years
  in B2B sales"), legal ("EU work authorisation", "valid driving
  licence"), language ("fluent German"), certification ("PMP", "CFA",
  "Security+", "Board-certified neurologist").

For each item, emit the canonical short name as it would naturally appear
on a candidate's CV or in a hiring manager's mental checklist. Strip
filler: prefer "Python" over "the Python programming language", "5+
years experience" over "we'd love it if you had at least five years of
experience". Deduplicate within the posting. Skip generic platitudes
("good communication", "team player") unless the posting puts notable
emphasis on them.

This applies to any profession — software, healthcare, finance, law,
design, trades, academia. Output ONLY the JSON object. No prose, no
markdown fences.
"""
