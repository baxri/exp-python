"""Command-line interface: `sec-graphrag <command>`."""

from enum import StrEnum
from typing import Annotated

import typer

app = typer.Typer(
    help="GraphRAG Q&A over semiconductor 10-K filings.",
    no_args_is_help=True,
)


class Strategy(StrEnum):
    """How a question is answered. `auto` lets the router agent decide."""

    vector = "vector"
    cypher = "cypher"
    hybrid = "hybrid"
    auto = "auto"


class Subset(StrEnum):
    ci = "ci"
    full = "full"


def _todo(step: str) -> None:
    typer.secho(f"Not implemented yet — see plan step {step}.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


@app.command()
def doctor() -> None:
    """Check connectivity to Neo4j, OpenAI and LangSmith."""
    # Imported here, not at the top: LangChain is slow to import, and other
    # commands (and --help) shouldn't pay for it.
    from sec_graphrag.config import get_settings
    from sec_graphrag.doctor import Status, run_checks
    from sec_graphrag.tracing import configure_langsmith

    configure_langsmith(get_settings())
    icons = {Status.ok: "✅", Status.warn: "⚠️ ", Status.fail: "❌"}
    results = run_checks()
    for r in results:
        typer.echo(f"{icons[r.status]} {r.name:<18} {r.detail}")
    if any(r.status is Status.fail for r in results):
        raise typer.Exit(code=1)


@app.command(name="init-db")
def init_db() -> None:
    """Create Neo4j constraints and indexes (safe to run repeatedly)."""
    from sec_graphrag.graph.constraints import create_constraints, list_constraints

    create_constraints()
    for c in list_constraints():
        typer.echo(f"  {c['name']:<22} {c['label']}.{c['property']} IS UNIQUE")
    typer.secho("Schema ready.", fg=typer.colors.GREEN)


@app.command()
def fetch(
    tickers: Annotated[
        list[str] | None,
        typer.Argument(help="Tickers to fetch. Default: all companies."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-download even if cached.")] = False,
) -> None:
    """Download 10-K items 1, 1A and 7 from SEC EDGAR into the local cache."""
    from sec_graphrag.companies import FILERS, get_filer
    from sec_graphrag.ingest.fetch import FetchStatus, fetch_many

    # Validate every ticker before downloading anything.
    try:
        selected = [get_filer(t).ticker for t in tickers] if tickers else None
    except KeyError as e:
        raise typer.BadParameter(str(e)) from e
    selected = selected or [f.ticker for f in FILERS]

    icons = {
        FetchStatus.downloaded: "⬇️ ",
        FetchStatus.cached: "💾",
        FetchStatus.failed: "❌",
    }
    counts = dict.fromkeys(FetchStatus, 0)
    for result in fetch_many(selected, force=force):
        counts[result.status] += 1
        if result.doc:
            words = sum(len(text.split()) for text in result.doc.items.values())
            detail = f"{result.doc.period_of_report}  {words:>7,} words"
        else:
            detail = result.error
        typer.echo(f"{icons[result.status]} {result.ticker:<5} {detail}")

    typer.echo(
        f"\n{counts[FetchStatus.downloaded]} downloaded, "
        f"{counts[FetchStatus.cached]} cached, {counts[FetchStatus.failed]} failed"
    )
    if counts[FetchStatus.failed]:
        raise typer.Exit(code=1)


@app.command()
def build(
    vector: Annotated[
        bool, typer.Option("--vector", help="Chunk, embed and index filings.")
    ] = False,
    graph: Annotated[
        bool, typer.Option("--graph", help="Extract entities and relationships.")
    ] = False,
) -> None:
    """Load cached filings into Neo4j."""
    if not (vector or graph):
        raise typer.BadParameter("Pass --vector, --graph or both.")
    if graph:
        _todo("M3")

    from sec_graphrag.companies import FILERS
    from sec_graphrag.graph.constraints import create_constraints
    from sec_graphrag.graph.load import load_filing, retire_removed_filers
    from sec_graphrag.ingest.chunk import chunk_filing
    from sec_graphrag.ingest.fetch import fetch_filing

    create_constraints()  # cheap and idempotent: never load without them
    total = 0
    ciks = []
    for filer in FILERS:
        doc = fetch_filing(filer.ticker)  # from the cache (downloads if missing)
        chunks = chunk_filing(doc)
        load_filing(doc, chunks)
        total += len(chunks)
        ciks.append(doc.cik)
        typer.echo(f"  {filer.ticker:<5} {len(chunks):>4} chunks")
    typer.secho(f"Loaded {len(FILERS)} filings, {total} chunks.", fg=typer.colors.GREEN)

    # Companies removed from FILERS lose their filings and chunks.
    if retired := retire_removed_filers(ciks):
        typer.echo(f"No longer filers (filings removed): {', '.join(retired)}")

    from sec_graphrag.ingest.embed import count_missing, embed_missing_chunks

    missing = count_missing()
    typer.echo(f"Embedding {missing} chunks without an embedding...")
    done = 0
    for batch_size in embed_missing_chunks():
        done += batch_size
        typer.echo(f"  {done}/{missing}")
    typer.secho("Embeddings up to date.", fg=typer.colors.GREEN)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    strategy: Annotated[
        Strategy, typer.Option(help="Retrieval strategy.")
    ] = Strategy.auto,
) -> None:
    """Answer a question about the filings, with citations."""
    _todo("1.9")


@app.command(name="eval")
def eval_(
    strategy: Annotated[
        Strategy, typer.Option(help="Strategy to evaluate.")
    ] = Strategy.auto,
    subset: Annotated[Subset, typer.Option(help="Dataset split to run.")] = Subset.full,
) -> None:
    """Run a LangSmith evaluation experiment."""
    _todo("2.7")


@app.command()
def report() -> None:
    """Summarize cost and latency per strategy from LangSmith runs."""
    _todo("M7")


@app.command()
def wipe(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Delete all project data (:SEC nodes) from Neo4j. Constraints are kept."""
    from sec_graphrag.graph.constraints import count_project_nodes, wipe_project_data

    count = count_project_nodes()
    if count == 0:
        typer.echo("Nothing to delete: no :SEC nodes.")
        return
    if not yes:
        # abort=True exits with code 1 when the answer is no.
        typer.confirm(f"Delete {count} :SEC nodes and their relationships?", abort=True)
    deleted = wipe_project_data()
    typer.secho(f"Deleted {deleted} nodes.", fg=typer.colors.GREEN)
