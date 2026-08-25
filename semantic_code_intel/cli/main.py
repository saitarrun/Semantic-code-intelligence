"""
Command Line Interface for Semantic Code Intelligence Platform.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
import typer
import uvicorn
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from semantic_code_intel.benchmark.bench_suite import BenchmarkRunner
from semantic_code_intel.cli.formatters import (
    console,
    render_banner,
    render_latency_table,
    render_query_results,
    render_stats_table,
)
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.generation.synthesizer import CodeSynthesizer
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline

app = typer.Typer(
    name="code-intel",
    help="Semantic Code Intelligence Platform: Local-first Code RAG with FAISS, BM25, and Cross-Encoder Reranking",
    add_completion=False
)

logging.basicConfig(level=logging.WARNING)


@app.command(name="index")
def index_cmd(
    target_dir: Path = typer.Argument(
        default=Path("."),
        help="Path to codebase directory to index"
    ),
    index_dir: Optional[Path] = typer.Option(
        None,
        "--index-dir", "-i",
        help="Custom directory to store index files"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force full reindex, wiping previous index"
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size", "-b",
        help="Embedding inference batch size"
    )
):
    """Scan and index repository into FAISS vector index, BM25 index, and metadata store."""
    render_banner()
    cfg = CodeIntelConfig(project_root=target_dir)
    if index_dir:
        cfg.index_dir = index_dir
    cfg.embedding.batch_size = batch_size

    indexer = HybridIndexer(cfg)
    console.print(f"[bold cyan]Indexing repository:[/] [yellow]{target_dir.resolve()}[/]")
    console.print(f"[bold cyan]Storage directory:[/]  [yellow]{cfg.get_index_dir()}[/]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Scanning & parsing codebase...", total=None)

        def progress_cb(stage: str, current: int, total: int, message: str = "", percentage: float = 0.0):
            desc = message or f"{stage.capitalize()} {current}/{total}"
            progress.update(task, description=f"[green]{desc}[/]", total=total, completed=current)

        metrics = indexer.index_codebase(
            target_dir=target_dir,
            force_reindex=force,
            progress_callback=progress_cb
        )

    console.print("\n[bold green]✓ Indexing successfully completed![/]\n")
    render_stats_table(metrics)


@app.command(name="query")
def query_cmd(
    query_text: str = typer.Argument(..., help="Search query or question"),
    target_dir: Path = typer.Option(
        Path("."),
        "--dir", "-d",
        help="Root path of indexed repository"
    ),
    index_dir: Optional[Path] = typer.Option(
        None,
        "--index-dir", "-i",
        help="Path where index is stored"
    ),
    top_k: int = typer.Option(
        5,
        "--top-k", "-k",
        help="Number of final results to return"
    ),
    mode: str = typer.Option(
        "hybrid",
        "--mode", "-m",
        help="Search mode: 'hybrid', 'dense', or 'sparse'"
    ),
    rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Enable/disable Cross-Encoder reranking"
    ),
    show_code: bool = typer.Option(
        True,
        "--show-code/--citations-only",
        help="Display full code snippet content"
    )
):
    """Execute sub-second hybrid code search with exact file and line-level citations."""
    cfg = CodeIntelConfig(project_root=target_dir)
    if index_dir:
        cfg.index_dir = index_dir

    pipeline = HybridRetrievalPipeline(cfg)
    if not pipeline.is_indexed():
        console.print(f"[bold red]Error:[/] Index not found in [yellow]{cfg.get_index_dir()}[/]. Run `code-intel index` first.")
        raise typer.Exit(code=1)

    response = pipeline.query(
        query_text=query_text,
        top_k=top_k,
        use_reranker=rerank,
        mode=mode
    )

    render_query_results(response, show_code=show_code)


@app.command(name="ask")
def ask_cmd(
    query_text: str = typer.Argument(..., help="Question to ask about the codebase"),
    target_dir: Path = typer.Option(Path("."), "--dir", "-d"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    provider: str = typer.Option("extractive", "--provider", "-p", help="LLM backend: 'extractive' or 'ollama'")
):
    """Query codebase and synthesize a cited answer with code context."""
    cfg = CodeIntelConfig(project_root=target_dir)
    if index_dir:
        cfg.index_dir = index_dir

    pipeline = HybridRetrievalPipeline(cfg)
    if not pipeline.is_indexed():
        console.print(f"[bold red]Error:[/] Index not found in [yellow]{cfg.get_index_dir()}[/]. Run `code-intel index` first.")
        raise typer.Exit(code=1)

    res = pipeline.query(query_text=query_text, top_k=top_k, use_reranker=True)
    synthesizer = CodeSynthesizer(provider=provider)
    answer = synthesizer.synthesize(query_text, res.results)

    console.print(f"\n[bold green]Answer for:[/] [bold]{query_text}[/]\n")
    console.print(answer.answer)
    console.print(render_latency_table(res.latency))


@app.command(name="stats")
def stats_cmd(
    target_dir: Path = typer.Option(Path("."), "--dir", "-d"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i")
):
    """View repository index stats, chunk count, and metadata."""
    cfg = CodeIntelConfig(project_root=target_dir)
    if index_dir:
        cfg.index_dir = index_dir

    pipeline = HybridRetrievalPipeline(cfg)
    if not pipeline.is_indexed():
        console.print(f"[bold red]Error:[/] Index not found in [yellow]{cfg.get_index_dir()}[/].")
        raise typer.Exit(code=1)

    stats = pipeline.metadata_store.get_stats()
    manifest = pipeline.metadata_store.get_manifest_val("index_manifest", {})
    combined = {**stats, **manifest}
    render_stats_table(combined)


@app.command(name="interactive")
def interactive_cmd(
    target_dir: Path = typer.Option(Path("."), "--dir", "-d"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i")
):
    """Start an interactive REPL search shell with sub-second retrieval."""
    render_banner()
    cfg = CodeIntelConfig(project_root=target_dir)
    if index_dir:
        cfg.index_dir = index_dir

    pipeline = HybridRetrievalPipeline(cfg)
    if not pipeline.is_indexed():
        console.print(f"[bold red]Error:[/] Index not found in [yellow]{cfg.get_index_dir()}[/]. Run `code-intel index` first.")
        raise typer.Exit(code=1)

    console.print("[bold green]Interactive Code Intelligence REPL[/] (Type 'exit' or 'q' to quit)\n")

    while True:
        try:
            query = console.input("[bold cyan]code-intel> [/]").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Exiting REPL.[/]")
                break

            response = pipeline.query(query_text=query, top_k=3, use_reranker=True)
            render_query_results(response, show_code=True)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting REPL.[/]")
            break


@app.command(name="serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind server to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload server on changes")
):
    """Launch the FastAPI REST API server and interactive Web UI."""
    render_banner()
    console.print(f"[bold green]Starting Web UI and API server at:[/] [bold cyan]http://{host}:{port}[/]\n")
    uvicorn.run("semantic_code_intel.api.app:app", host=host, port=port, reload=reload)


@app.command(name="benchmark")
def benchmark_cmd(
    workspace_dir: Path = typer.Option(
        Path("./benchmark_workspace"),
        "--workspace", "-w",
        help="Directory to place benchmark repo and index"
    ),
    target_loc: int = typer.Option(
        35000,
        "--loc", "-l",
        help="Target Lines of Code to generate and index (default: 35,000)"
    ),
    queries: int = typer.Option(
        30,
        "--queries", "-q",
        help="Number of test queries to run for evaluation"
    )
):
    """Run full automated benchmark on a 30,000+ LOC codebase."""
@app.command(name="mcp")
def mcp_cmd():
    """Start the Model Context Protocol (MCP) JSON-RPC 2.0 stdio server."""
    from semantic_code_intel.mcp.server import run_mcp_server
    run_mcp_server()


@app.command(name="watch")
def watch_cmd(
    target_dir: Path = typer.Option(
        Path("."),
        "--dir", "-d",
        help="Root path of repository to watch"
    )
):
    """Start the real-time background filesystem watcher for sub-50ms incremental indexing."""
    import time
    from semantic_code_intel.indexing.watcher import CodebaseWatcher

    render_banner()
    cfg = CodeIntelConfig(project_root=target_dir)
    console.print(f"[bold green]Starting Incremental Watcher on:[/] [bold cyan]{target_dir.resolve()}[/]")
    console.print("[dim]Press Ctrl+C to stop.[/]\n")

    def on_change(event_type: str, file_path: str):
        console.print(f"[cyan]⚡ Event:[/] [{ 'green' if event_type=='updated' else 'red' }]{event_type}[/] [yellow]{file_path}[/]")

    watcher = CodebaseWatcher(config=cfg, on_change_callback=on_change)
    watcher.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/]")
        watcher.stop()


if __name__ == "__main__":
    app()

