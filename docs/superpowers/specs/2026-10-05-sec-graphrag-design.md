# SEC 10-K GraphRAG Explorer — Design Spec

- **Date:** 2026-10-05
- **Status:** Approved in brainstorming, pending implementation plan
- **Location:** `projects/sec_graphrag/` (uv workspace member of `exp-python`)

## 1. Goal

A portfolio project that showcases **GraphRAG, LangChain/LangGraph and LangSmith** by answering
natural-language questions over the 10-K filings of ~22 US-listed semiconductor companies, and
**proving with LangSmith experiments** where a knowledge graph beats plain vector RAG.

The headline deliverable is a README results table: scores per question category for three
fixed retrieval pipelines (`vector`, `cypher`, `hybrid`) and the tool-calling agent (`auto`),
plus a cost/latency table.

### Decisions made

| Topic | Decision |
|---|---|
| Domain | US-listed semiconductor companies, latest 10-K only |
| 10-K sections | Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A) |
| LLM provider | OpenAI (chat + `text-embedding-3-small`) |
| Interface | CLI (Typer): one-shot `ask` and multi-turn `chat`; Jupyter notebooks; no web UI |
| Graph construction | Fixed schema + LangChain `LLMGraphTransformer` |
| Answering | Hand-built LangGraph tool-calling agent; each retriever is a tool. Fixed per-strategy pipelines are kept for a fair retrieval comparison (revised 2026-10-06, replacing the router agent) |
| Off-topic questions | Answered briefly from general knowledge, clearly labelled as not from the filings; company-specific questions must use the tools |
| Neo4j | Existing Neo4j Aura instance (single database), project data namespaced with `:SEC` label |
| Build approach | Incremental: vector baseline → chat agent → evals → graph → cypher tool → hybrid tool → agent hardening |
| Eval set | ~60 questions, LLM-drafted from filing text, hand-reviewed |
| Extras | CI eval regression, cost/latency report, online feedback (prompt hub deferred) |

### Non-goals (v1)

Multi-year filings, 10-Q / 20-F forms, financial-table extraction, web UI, LangSmith prompt hub,
Microsoft-GraphRAG-style community summaries. Listed as "Future work" in the README.

## 2. Architecture

```
EDGAR ──(edgartools)──> raw 10-K items (1, 1A, 7) ──> data/raw/*.json      (cached)
                                   │
                    chunk (RecursiveCharacterTextSplitter)
                     ┌─────────────┴──────────────┐
             embed chunks                 LLMGraphTransformer (fixed schema)
          (OpenAI embeddings)                     │  data/extracted/*.jsonl (cached)
                     └─────────────┬──────────────┘
                                 Neo4j Aura (:SEC namespace)
             (:Chunk {embedding}) ─[:MENTIONS]→ entities ─[rels]→ entities
                                   │
  LangGraph agent: LLM ⇄ tools (search_filings | query_graph | explore_graph) → cited answer
  fixed pipelines (evals): retrieve with one strategy → answer
                                   │
               LangSmith: traces · datasets · experiments · feedback
```

### Repo layout

```
projects/sec_graphrag/
├── pyproject.toml            # langchain, langchain-openai, langchain-neo4j, langgraph,
│                             # langsmith, edgartools, typer, pydantic-settings; dev: pytest
├── README.md
├── .env.example
├── src/sec_graphrag/
│   ├── config.py             # pydantic-settings: Neo4j, OpenAI, LangSmith, EDGAR, model names
│   ├── companies.py          # filer tickers
│   ├── ingest/               # fetch.py, chunk.py, embed.py, extract.py
│   ├── graph/                # schema.py, constraints.py, resolve.py, quality.py
│   ├── retrieval/            # base.py, vector.py, text2cypher.py, hybrid.py
│   ├── agent/                # answer.py, qa.py (fixed pipelines), tools.py, graph.py (agent), chat.py
│   ├── evals/                # draft.py, sync.py, evaluators.py, run.py, dataset.jsonl, baseline.json
│   ├── reports/              # cost_latency.py
│   └── cli.py
├── notebooks/                # 01_explore_filings, 02_graph_exploration, 03_eval_analysis
├── data/                     # gitignored: raw/, extracted/
└── tests/
```

CI workflow lives at the repo root: `.github/workflows/sec-graphrag-eval.yml`.

### Configuration (`.env`)

Existing: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `OPENAI_API_KEY`.
New: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=sec-graphrag`,
`EDGAR_IDENTITY="Name email"` (required by SEC). Model names (`extract_model`, `answer_model`,
`judge_model`, `embedding_model`) are settings with defaults in `config.py`.

### CLI

```
sec-graphrag doctor                 # check Neo4j, OpenAI, LangSmith connectivity
sec-graphrag fetch                  # download + cache 10-K items
sec-graphrag build --vector         # chunks + embeddings + vector index
sec-graphrag build --graph          # extraction + load + entity resolution + quality report
sec-graphrag ask "..." [--strategy vector|cypher|hybrid|auto] [--ticker T]   # one-shot
sec-graphrag chat                   # multi-turn conversation with the agent (memory per session)
sec-graphrag eval [--strategy ...] [--subset ci|full]
sec-graphrag eval sync              # upload dataset.jsonl to LangSmith
sec-graphrag eval set-baseline      # explicitly update evals/baseline.json
sec-graphrag report                 # cost/latency tables from LangSmith runs
sec-graphrag wipe                   # MATCH (n:SEC) DETACH DELETE n (with confirmation)
```

## 3. Graph schema

Every node carries the extra label `:SEC` so project data is isolated from other data in the
shared Aura database.

### Nodes

| Label | Key | Other properties |
|---|---|---|
| `Company` | `cik` for filers; normalized `name` otherwise | `ticker`, `name`, `is_filer` |
| `Filing` | `accession_no` | `fiscal_year`, `form`, `filed_at` |
| `Chunk` | `id` | `text`, `item` (1/1A/7), `ticker`, `fiscal_year`, `embedding` (1536-d) |
| `Person` | `name` | |
| `RiskFactor` | `name` | `category` ∈ {supply_chain, geopolitical, regulatory, competition, cyber, macro, other} |
| `Product` | `name` | |
| `Technology` | `name` | |
| `Location` | `name` | |

### Relationships

```
(Company)-[:FILED]->(Filing)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:MENTIONS]->(Company|Person|RiskFactor|Product|Technology|Location)
(Company)-[:SUPPLIES]->(Company)
(Company)-[:CUSTOMER_OF]->(Company)
(Company)-[:COMPETES_WITH]->(Company)
(Company)-[:FACES_RISK]->(RiskFactor)
(RiskFactor)-[:RELATED_TO]->(Location|Technology)
(Company)-[:MAKES]->(Product)-[:USES]->(Technology)
(Company)-[:OPERATES_IN]->(Location)
(Person)-[:EXECUTIVE_OF {title}]->(Company)
(Person)-[:DIRECTOR_OF]->(Company)
```

Every extracted (non-structural) relationship stores `chunk_id` of the passage it came from.

### Indexes and constraints

Uniqueness constraints on the keys above; vector index on `Chunk.embedding`; full-text index over
entity `name` properties (used for entity linking).

### Entity resolution

1. Filers are anchored by `cik`/`ticker`.
2. Alias table (`graph/resolve.py`), e.g. `TSMC`, `Taiwan Semiconductor Manufacturing Co.` → `TSMC`.
3. Post-load pass: normalize names (case, punctuation, legal suffixes like Inc./Corp./Ltd.) and
   merge near-duplicates within a label above a similarity threshold.
4. Quality report lists remaining duplicate candidates for manual alias additions.

## 4. Ingestion pipeline

Each stage caches to disk, is idempotent (`MERGE`), and can be rerun without repeating LLM spend.

1. **Fetch** — `edgartools` gets the latest 10-K per filer and extracts Items 1, 1A, 7 →
   `data/raw/{ticker}.json` (latest 10-K only) with CIK, accession no., filing date and `period_of_report` (fiscal years differ per company).
   Filers (to be verified as 10-K filers during implementation): NVDA, AMD, INTC, QCOM, AVGO,
   TXN, MU, ADI, MRVL, NXPI, ON, MCHP, AMAT, LRCX, KLAC, TER, ENTG, SWKS, QRVO, MPWR, WOLF, AAPL.
   TSMC, ASML, Samsung etc. are not 10-K filers; they appear as non-filer `Company` nodes.
2. **Chunk + embed** — ~1,000-token chunks, 150 overlap, metadata `ticker`, `item`,
   `fiscal_year`. Embedded with `text-embedding-3-small`, written as `(:SEC:Chunk)`.
   The vector baseline is usable after this stage.
3. **Extract** — `LLMGraphTransformer` constrained to the schema, with a domain-specific prompt
   (e.g. "foundry partner" → `SUPPLIES`). Mini-class extraction model, `.abatch` with
   `max_concurrency≈8`. Results written to `data/extracted/{ticker}.jsonl` before loading.
   Traced in a separate LangSmith project `sec-graphrag-ingest`.
   A 2-company trial run measures cost and quality before the full run (expected: single-digit USD).
4. **Load + resolve** — constraints first, `MERGE` nodes/relationships, entity resolution,
   then a quality report: counts per label/type, top-20 most-connected companies, duplicate
   candidates.

Size (measured after fetch): ~641k words ≈ 850k tokens ≈ 1k chunks; entity nodes estimated 5–15k — well within Aura Free limits (~200k nodes / 400k rels).

Keeping Neo4j in sync with the source (`build --vector` is safe to re-run):
- chunk settings/code changed → `build --vector` (stale chunks deleted, changed text re-embedded only)
- new 10-K published or cleaning changed → `fetch --force [TICKER]`, then `build --vector`
  (each filer keeps only its latest filing; older filings and their chunks are deleted)
- company removed from `FILERS` → `build --vector` deletes its filings/chunks and marks it
  `is_filer = false` (the node stays, as other filings may mention it)
- embedding model changed → deliberate manual step: clear embeddings (and recreate the vector
  index if the dimensions differ), then `build --vector`

Special case: Intel's 10-K uses a cross-reference index instead of Item sections, so its whole document is kept as one `Full` section (`Filer.full_text`). Every other filing is validated to have Items 1, 1A and 7 with at least 1,000 words each; a failing filing is reported, never cached.

## 5. Retrieval and agent

### Retriever interface

`retrieve(question, **filters) -> RetrievalResult(context: str, sources: list[Source], cypher: str | None)`
where `Source = (ticker, fiscal_year, item, chunk_id)`.

### Vector (`retrieval/vector.py`)

`Neo4jVector` similarity over `:SEC:Chunk`, top-k 6, optional `ticker` filter when the question
names a filer.

### Text2Cypher (`retrieval/text2cypher.py`) — custom, not `GraphCypherQAChain`

- Prompt = graph schema + ~15 few-shot examples chosen by `SemanticSimilarityExampleSelector`.
- Safety: read-only transaction (`execute_read`), regex rejection of write clauses
  (`CREATE|MERGE|DELETE|SET|REMOVE|DROP|LOAD CSV|CALL dbms`), default `LIMIT`, query timeout.
- Validation: `EXPLAIN` before execution; `CypherQueryCorrector` fixes relationship direction.
- Self-correction: on error or empty result, retry with the error fed back, max 2 retries
  (implemented as an agent loop so each attempt is traced).

### Hybrid (`retrieval/hybrid.py`)

1. Entity linking: structured-output LLM call extracts entity mentions → full-text index lookup.
2. Vector search for top chunks.
3. Expand linked entities + entities mentioned by retrieved chunks to 1–2 hop neighborhoods,
   rendered as triples.
4. Merge chunks + triples, de-duplicate, trim to a token budget.

### LangGraph agent (`agent/`) — revised 2026-10-06

A hand-built `StateGraph` tool-calling agent (not a prebuilt constructor):

```
START → agent (LLM with tools bound) ──tool calls?──▶ tools (ToolNode) ──▶ agent ...
                                     └──no──▶ END (final cited answer)
```

- State: `messages` (conversation, `add_messages` reducer).
- Tools, added as retrievers are built: `search_filings(query, ticker?)` (vector, M1.5),
  `query_graph(question)` (Text2Cypher, M4), `explore_graph(question)` (hybrid, M5). Each returns
  context with `[TICKER · Item]` headers so the agent can cite.
- System prompt (grounding rule): any question about specific companies, their business, risks,
  suppliers, customers or numbers MUST be answered from tool results with citations. General
  questions may be answered directly, prefixed as general knowledge, not from the filings.
- Memory: LangGraph checkpointer (in-memory `MemorySaver`) keyed by `thread_id`; one thread per
  `chat` session. LangSmith groups a session's turns as a thread via `thread_id` metadata.
- Limits: a recursion limit bounds tool-call loops.

Fixed pipelines (`agent/qa.py`) stay: `ask --strategy vector|cypher|hybrid` runs exactly one
retriever then the answer chain. They give a like-for-like comparison of retrieval methods;
`--strategy auto` (default from M1.5) runs the agent.

M6 hardens the agent: tool descriptions, Cypher failure fallback to vector search, guardrails,
and agent-vs-pipeline comparison.

CLI `ask` output: answer, tools used, latency, cost, Cypher (if any), sources, then a feedback
prompt. `chat` shows the same per turn.

## 6. Evaluation, observability, feedback

### Dataset

~60 questions: lookup 15, multi-hop 25, aggregation 10, thematic 10.
Fields: `question, reference_answer, expected_entities, category, source_chunks`.

- Source of truth: `evals/dataset.jsonl` in git; `eval sync` uploads to LangSmith dataset
  `sec-graphrag-qa` with splits `ci` (~12, stratified by category) and `full`.
- Drafting: `evals/draft.py` samples related chunks across filings, LLM drafts candidates,
  user accepts/edits/rejects in notebook `03`.
- **Reference answers are grounded in filing text, never in the extracted graph**, to avoid
  biasing the evaluation toward graph retrieval.

### Evaluators

| Name | Kind | Measures |
|---|---|---|
| `correctness` | LLM judge vs reference (0 / 0.5 / 1) | Answer correctness |
| `entity_recall` | Deterministic | Fraction of `expected_entities` in the answer |
| `groundedness` | LLM judge vs retrieved context | Unsupported claims / hallucination |
| `cypher_ok` | Deterministic, cypher runs only | Query succeeded; attempts used |
| `used_tools` | Deterministic, agent runs only | Company-specific question answered only after a tool call (grounding rule) |

Judge: stronger model than the answerer, temperature 0, prompt calibrated with a few
hand-checked examples. Latency, tokens and cost come from LangSmith run stats.

### Experiments

`langsmith.evaluate(target, data="sec-graphrag-qa", evaluators=..., experiment_prefix=<strategy>,
metadata={strategy, models, git_sha}, num_repetitions=2)` for each fixed pipeline (`vector`,
`cypher`, `hybrid`) and the agent (`auto`). A script exports per-category score tables to the
README.

### CI eval regression

`.github/workflows/sec-graphrag-eval.yml`, triggered on PRs touching `projects/sec_graphrag/**`:
runs `eval --subset ci --strategy auto`, compares with `evals/baseline.json`, fails if
`correctness` drops by more than 0.1 or `cypher_ok` falls below its threshold, and writes the
LangSmith experiment URL to the job summary. Secrets: OpenAI, LangSmith, Neo4j. Read-only
against Aura. Baseline changes only via `eval set-baseline`.

### Cost/latency report

`report` uses `client.runs.query()` (the replacement for the deprecated `list_runs()`) on root runs in the Q&A project, grouped by `strategy`
metadata: run count, p50/p95 latency, mean tokens, mean cost, error rate, mean Cypher retries;
plus a per-node breakdown. Outputs Markdown (`reports/`) and CSV.

### Online feedback

After `ask` and each `chat` turn, prompt `Helpful? [y/n/skip]` + optional comment →
`client.create_feedback(run_id, key="user_helpful", score=1|0, comment=...)`, with `run_id`
preassigned via run config. Notebook `03` lists negatively rated runs and promotes them into
`dataset.jsonl` as new eval candidates.

## 7. Testing

pytest unit tests for deterministic logic only, with no network calls: chunk metadata, Cypher
safety check, name normalization/alias resolution, agent loop/tool routing with a fake chat model,
deterministic evaluators, report aggregation. LLM quality is measured by LangSmith evals, not
unit tests.

## 8. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold: workspace member, config, CLI skeleton, `.env.example`, constraints, `doctor` | `doctor` passes for Neo4j, OpenAI, LangSmith |
| M1 | Fetch + chunk + embed + `ask --strategy vector` with tracing | Questions answered with citations; traces visible |
| M1.5 | Hand-built LangGraph agent with `search_filings` tool; `chat` command with memory; `ask` defaults to the agent | Multi-turn chat with follow-ups; tool calls and threads visible in LangSmith |
| M2 | Eval set drafted/reviewed, synced, evaluators (incl. `used_tools`), `eval` command | Baseline experiments recorded per category: vector pipeline and agent |
| M3 | Extraction (2-company trial, then full), resolution, quality report, notebook `02` | Quality report sane, few duplicates |
| M4 | Text2Cypher with safety, validation, retries; exposed as `query_graph` tool + fixed pipeline | Cypher experiment recorded vs vector |
| M5 | Hybrid retriever; exposed as `explore_graph` tool + fixed pipeline | Hybrid experiment recorded |
| M6 | Agent hardening: tool descriptions, Cypher→vector fallback, guardrails | `auto` (agent) experiment recorded vs all three pipelines |
| M7 | `report`, CLI feedback, failure promotion, CI workflow + baseline | A PR with a deliberately broken prompt fails CI |
| M8 | README: diagram, results, cost/latency, demo GIF, lessons learned | Understandable in 2 minutes |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Extraction quality poor / noisy graph | Fixed schema, domain prompt, 2-company trial, quality report, alias table |
| Entity fragmentation breaks multi-hop | Resolution pass + duplicate report |
| LLM judge unreliable | Calibration examples, 3-level scale, repetitions, deterministic `entity_recall` alongside |
| Aura Free limits / idle pause | Size estimate fits; `doctor` detects paused instance |
| Shared database with other experiments | `:SEC` label on all nodes; `wipe` scoped to it |
| `edgartools` item parsing fails for some filings | Cache raw output; log and skip with a per-ticker report |
| Cost overrun | Trial run, caching of extraction output, mini model for extraction |
| Agent answers company questions from training memory (ungrounded) | Grounding rule in system prompt, general-knowledge label, `used_tools` evaluator |
| Agent results don't isolate which retrieval method helped | Fixed per-strategy pipelines kept and evaluated alongside the agent |
