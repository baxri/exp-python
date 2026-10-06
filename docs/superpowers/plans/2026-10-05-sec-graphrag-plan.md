# SEC 10-K GraphRAG Explorer — Implementation Plan

- **Spec:** [`../specs/2026-10-05-sec-graphrag-design.md`](../specs/2026-10-05-sec-graphrag-design.md)
- **Code:** `projects/sec_graphrag/`
- **Started:** 2026-10-05

## How we work

This project is a learning vehicle, so every step follows the same loop:

1. **Concept** — explain the idea and why it is done this way.
2. **Code** — write one small piece (usually one file or function).
3. **Walkthrough** — explain the code line by line where it matters.
4. **Verify** — the user runs a command and sees it work.
5. **Pause** — questions, then continue only on the user's go-ahead.

Dependencies are added in the step that first needs them, so each one has a visible reason.
Nothing is committed without the user's explicit approval.

Detail level: M0–M2 are broken into steps now. M3–M8 are outlined and get detailed when we reach
them, because results from earlier milestones (e.g. extraction quality) will shape them.

---

## M0 — Scaffold ✅

Goal: a runnable, configured project that can talk to Neo4j, OpenAI and LangSmith.

- [x] **0.1 Workspace member**
  - Learn: uv workspaces, `src/` layout, console-script entry points, editable installs.
  - Files: `projects/sec_graphrag/pyproject.toml`, `src/sec_graphrag/__init__.py`, root `pyproject.toml`.
  - Verify: `uv run sec-graphrag` prints a greeting.

- [x] **0.2 Typed configuration**
  - Learn: `pydantic-settings`, reading `.env`, `SecretStr` to keep secrets out of logs, nested
    defaults for model names.
  - Add deps: `pydantic-settings`.
  - Files: `src/sec_graphrag/config.py`, `projects/sec_graphrag/.env.example`.
  - Verify: `uv run python -c "from sec_graphrag.config import get_settings; print(get_settings())"`
    shows values with secrets masked.

- [x] **0.3 CLI skeleton**
  - Learn: Typer commands/options/help, pointing the entry point at a Typer app.
  - Add deps: `typer`.
  - Files: `src/sec_graphrag/cli.py`, update `[project.scripts]`.
  - Verify: `uv run sec-graphrag --help` lists placeholder commands.

- [x] **0.4 Neo4j connection**
  - Learn: driver vs session vs transaction, connection pooling, `execute_read` / `execute_write`,
    why one driver per process.
  - Add deps: `neo4j`.
  - Files: `src/sec_graphrag/db.py`.
  - Verify: a one-line script runs `RETURN 1` against Aura.

- [x] **0.5 `doctor` command + LangSmith setup**
  - Learn: how LangSmith tracing is switched on by environment variables, a first traced LLM
    call, projects in the LangSmith UI.
  - Add deps: `langchain-openai`, `langsmith`.
  - Files: `src/sec_graphrag/doctor.py`, `cli.py`.
  - User action: create a LangSmith API key, add `LANGSMITH_*` and `EDGAR_IDENTITY` to `.env`.
  - Verify: `uv run sec-graphrag doctor` shows ✅ for Neo4j, OpenAI, LangSmith; the trace appears
    in LangSmith.

- [x] **0.6 Schema constraints + `init-db` + `wipe`**
  - Learn: Cypher constraints and indexes, `IF NOT EXISTS` for idempotency, the `:SEC` namespace
    label, safe destructive commands (confirmation prompt).
  - Files: `graph/constraints.py`, `db.py` (`run_auto_commit`), `cli.py`. (`graph/schema.py` deferred to M3, where extraction first needs it.)
  - Verify: `sec-graphrag init-db` twice (second run is a no-op); `SHOW CONSTRAINTS` in Aura.

- [x] **0.7 Testing setup**
  - Learn: pytest layout, fixtures, testing pure functions without network.
  - Add dev deps: `pytest`.
  - Files: `tests/conftest.py`, `tests/test_config.py`, `tests/test_tracing.py`, `tests/test_constraints.py`.
  - Verify: `uv run pytest` green.

## M1 — Fetch + vector baseline ✅

Goal: ask a question and get a cited answer from plain vector RAG, traced in LangSmith.

- [x] **1.1 Company list + EDGAR basics**
  - Learn: EDGAR, CIK, form types (10-K vs 20-F), SEC fair-access rules.
  - Add deps: `edgartools`.
  - Files: `src/sec_graphrag/companies.py`.
  - Verify: script confirms each ticker has a recent 10-K; drop any that don't.

- [x] **1.2 Fetch one filing**
  - Learn: extracting Items 1/1A/7, a `FilingDoc` pydantic model, JSON caching to `data/raw/`.
  - Files: `src/sec_graphrag/ingest/fetch.py`, `.gitignore` for `data/`.
  - Verify: NVDA's cached JSON exists; item lengths look sensible.

- [x] **1.3 Fetch all + `fetch` command**
  - Learn: skip-if-cached, per-ticker error handling and a summary report.
  - Files: `ingest/fetch.py`, `cli.py`, `companies.py` (`full_text` flag for Intel). Notebook moved to M3.
  - Verify: `sec-graphrag fetch` summary. Result: 22/22, ~641k words (~1k chunks); Intel kept as full text.

- [x] **1.4 Chunking**
  - Learn: chunk size/overlap trade-offs, token-based length with `tiktoken`, metadata on chunks.
  - Add deps: `langchain-text-splitters`, `tiktoken`.
  - Files: `ingest/chunk.py`, `tests/test_chunk.py`.
  - Verify: tests green; chunk counts per company printed.

- [x] **1.5 Structural graph load**
  - Learn: `MERGE` vs `CREATE`, batching with `UNWIND`, writing `Company`/`Filing`/`Chunk`.
  - Files: `graph/load.py`.
  - Verify: node counts in Aura match chunk counts.

- [x] **1.6 Embeddings + vector index**
  - Learn: what embeddings are, dimensions, cosine similarity, batching and cost, Neo4j vector
    indexes.
  - Files: `ingest/embed.py`, `graph/constraints.py`, `cli.py` (`build --vector`).
  - Verify: every `Chunk` has an embedding; raw vector query returns plausible chunks.

- [x] **1.7 Retriever interface + vector retriever**
  - Learn: LangChain `Neo4jVector`, retriever abstraction, our `RetrievalResult`/`Source` types,
    metadata filters.
  - Add deps: `langchain-neo4j`.
  - Files: `retrieval/base.py`, `retrieval/vector.py`.
  - Verify: retrieve for a sample question, print sources.

- [x] **1.8 Answer chain**
  - Learn: LCEL (`prompt | llm | parser`), prompt design for citations, formatting context.
  - Files: `agent/answer.py` (moves into the agent in M6).
  - Verify: cited answer for a sample question.

- [x] **1.9 `ask` command + tracing**
  - Learn: run names, tags and metadata in LangSmith, reading a trace tree.
  - Files: `cli.py`.
  - Verify: `sec-graphrag ask "..." --strategy vector`; inspect the trace in the UI.

## M1.5 — Chat agent (added 2026-10-06)

Goal: a hand-built LangGraph tool-calling agent behind `ask` (default) and a new multi-turn
`chat` command. Retrieval becomes a tool the model decides to call; general questions are
answered directly but labelled as not from the filings. Fixed pipelines stay for evals.

- [x] **A.1 Retrieval as a tool**
  - Learn: what a LangChain tool is (`@tool`, name + description + argument schema), how the
    model "sees" a tool (JSON schema), `bind_tools`, what a tool call looks like in an AIMessage.
  - Add deps: `langgraph`.
  - Files: `agent/tools.py` (`search_filings(query, ticker?)`).
  - Verify: invoke the tool directly; print its schema; a model with the tool bound returns a
    tool call for a company question and none for a general one.

- [ ] **A.1b Retrieval timing**
  - Learn: child spans with `@traceable` (`embed_question`, `neo4j_search`); separating
    one-time setup from per-question cost; avoiding a second Aura connection pool.
  - Files: `retrieval/vector.py`.
  - Verify: the `retrieve` span shows its parts; measure `Neo4jVector` setup options vs a query
    on our shared driver, keep the faster one that still showcases LangChain.

- [ ] **A.2 The agent graph, by hand**
  - Learn: `StateGraph`, state + `add_messages` reducer, nodes, edges, conditional edges,
    `ToolNode`, `tools_condition`, the agent loop, compiling, drawing the graph.
  - Files: `agent/graph.py`.
  - Verify: a company question loops agent → tools → agent → END; a general question goes
    agent → END; the Mermaid diagram of the graph.

- [ ] **A.3 System prompt: grounding rule + citations**
  - Learn: steering tool use with instructions; the "general knowledge" label; why ungrounded
    answers are the main risk of tool-calling RAG.
  - Files: `agent/graph.py` (prompt).
  - Verify: company question → tool + citations; general question → labelled answer;
    unanswerable question → "not in the filings".

- [ ] **A.4 `ask` runs the agent by default**
  - Learn: keeping two paths (agent vs fixed pipelines) behind one interface; reading tool calls
    from the result; agent traces in LangSmith; clean trace inputs/outputs (no object reprs,
    no tuple outputs).
  - Files: `agent/qa.py`, `cli.py`.
  - Verify: `ask "..."` (agent, shows tools used) and `ask "..." --strategy vector` (pipeline).

- [ ] **A.5 Conversation memory**
  - Learn: checkpointers (`MemorySaver`), `thread_id`, how state persists between turns, why
    follow-ups ("their risks") need history.
  - Files: `agent/graph.py`.
  - Verify: a two-turn conversation in a script where turn 2 depends on turn 1.

- [ ] **A.6 `chat` command**
  - Learn: a REPL loop, `/new` and `/exit` commands, streaming output, LangSmith threads
    (`thread_id` metadata groups a session); one-time setup at startup, outside the per-turn
    traces.
  - Files: `agent/chat.py`, `cli.py`.
  - Verify: an interactive session with follow-ups; the session appears as one thread in
    LangSmith.

- [ ] **A.7 Tests for the agent loop**
  - Learn: faking a tool-calling model so the graph's routing is tested without OpenAI.
  - Files: `tests/test_agent.py`.
  - Verify: tool-call path and direct-answer path both covered; `uv run pytest` green.

## M2 — Eval set + first experiment

Goal: a reviewed ~60-question dataset in LangSmith and recorded baselines per category for the
vector pipeline and the agent.

- [ ] **2.1 Dataset model + hand-written seed**
  - Learn: what makes a good eval question; the four categories; `EvalExample` pydantic model.
  - Files: `evals/models.py`, `evals/dataset.jsonl` (~5 hand-written).
  - Verify: loader parses the file.

- [ ] **2.2 LLM drafting of candidates**
  - Learn: `with_structured_output`, grounding questions in source chunks, avoiding graph bias.
  - Files: `evals/draft.py`.
  - Verify: `evals/candidates.jsonl` with drafted questions.

- [ ] **2.3 Review in notebook**
  - Learn: human-in-the-loop dataset curation.
  - Files: `notebooks/03_eval_analysis.ipynb`.
  - Verify: ~60 accepted examples, balanced by category.

- [ ] **2.4 Sync to LangSmith**
  - Learn: datasets, examples, splits (`ci`, `full`), dataset versions.
  - Files: `evals/sync.py`, `cli.py` (`eval sync`).
  - Verify: dataset visible in LangSmith with both splits.

- [ ] **2.5 Deterministic evaluators**
  - Learn: evaluator function signature (run outputs vs example reference), `entity_recall`,
    `used_tools` (agent answered a company question only after calling a tool).
  - Files: `evals/evaluators.py`, `tests/test_evaluators.py`.
  - Verify: tests green.

- [ ] **2.6 LLM-as-judge evaluators**
  - Learn: judge prompts, scoring scales, calibration, judge-model choice.
  - Files: `evals/evaluators.py` (`correctness`, `groundedness`).
  - Verify: judge scores on 3 hand-checked examples match expectations.

- [ ] **2.7 Run the first experiment**
  - Learn: `langsmith.evaluate`, target functions, experiment metadata, repetitions, the
    comparison view.
  - Files: `evals/run.py`, `cli.py` (`eval`).
  - Verify: vector-pipeline and agent experiments in LangSmith; per-category scores printed.

- [ ] **2.8 Record the baseline**
  - Learn: exporting experiment results; starting the README results table.
  - Files: `projects/sec_graphrag/README.md`.

## M3 — Knowledge graph (detail when reached)

Extraction prompt + schema, 2-company trial (cost/quality), full run with caching,
`MENTIONS` links, entity resolution (alias table + similarity merge), quality report,
`notebooks/02_graph_exploration.ipynb`.

## M4 — Text2Cypher (detail when reached)

Few-shot example bank + semantic selector, generation prompt, safety check, `EXPLAIN`
validation, `CypherQueryCorrector`, retry loop; exposed as the `query_graph` agent tool and as
the `cypher` fixed pipeline; experiment vs vector.

## M5 — Hybrid retriever (detail when reached)

Entity linking with full-text index, neighborhood expansion to triples, context merge and
trimming; exposed as the `explore_graph` agent tool and as the `hybrid` fixed pipeline;
experiment.

## M6 — Agent hardening (detail when reached; revised 2026-10-06, was "router agent")

Tool descriptions tuned from eval failures, Cypher failure → vector fallback, guardrails
(recursion limit, max tool calls), span metadata, `auto` (agent) experiment vs all three fixed
pipelines.

## M7 — Production extras (detail when reached)

`report` command (cost/latency from `client.runs.query()`; `list_runs` is deprecated), CLI feedback via `create_feedback`,
promoting thumbs-down runs to the dataset, CI workflow + `baseline.json` + `eval set-baseline`.

## M8 — README (detail when reached)

Architecture diagram, results tables, cost/latency table, demo GIF, lessons learned,
future work.
