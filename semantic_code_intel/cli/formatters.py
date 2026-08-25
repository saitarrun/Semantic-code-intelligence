"""
Rich terminal output formatters, syntax highlighting, panels, and tables.
"""

from __future__ import annotations

from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from semantic_code_intel.retrieval.pipeline import LatencyBreakdown, QueryResponse

console = Console()


def render_banner() -> None:
    """Render the ASCII art welcome banner for the platform."""
    banner_text = Text(
        r"""
  ____                          _   _        ____          _        ___       _       _ _ 
 / ___|  ___ _ __ ___   __ _ _ | |_(_) ___  / ___|___   __| | ___  |_ _|_ __ | |_ ___| | |
 \___ \ / _ \ '_ ` _ \ / _` | || __| |/ __|| |   / _ \ / _` |/ _ \  | || '_ \| __/ _ \ | |
  ___) |  __/ | | | | | (_| | || |_| | (__ | |__| (_) | (_| |  __/  | || | | | ||  __/ | |
 |____/ \___|_| |_| |_|\__,_|_| \__|_|\___| \____\___/ \__,_|\___| |___|_| |_|\__\___|_|_|
        Semantic Code Intelligence & Hybrid RAG Engine [FAISS + BM25 + CrossEncoder]
        """,
        style="bold cyan"
    )
    console.print(banner_text)


def render_latency_table(latency: LatencyBreakdown) -> Table:
    """Render a colored latency performance breakdown table."""
    table = Table(title="Retrieval Latency Breakdown", show_header=True, header_style="bold magenta")
    table.add_column("Pipeline Stage", style="cyan")
    table.add_column("Latency (ms)", justify="right", style="green")
    table.add_column("Status", justify="center")

    table.add_row("FAISS Dense Vector Search", f"{latency.dense_ms:.2f} ms", "⚡ Fast")
    table.add_row("BM25 Sparse Lexical Search", f"{latency.sparse_ms:.2f} ms", "⚡ Fast")
    table.add_row("Reciprocal Rank Fusion (RRF)", f"{latency.fusion_ms:.2f} ms", "⚡ Fast")
    table.add_row("SQLite Metadata & Code Fetch", f"{latency.metadata_fetch_ms:.2f} ms", "⚡ Fast")
    table.add_row("Cross-Encoder Reranking", f"{latency.rerank_ms:.2f} ms", "🎯 High Precision")
    
    sub_second = latency.total_ms < 1000.0
    status_icon = "[bold green]SUB-SECOND (PASS)[/]" if sub_second else "[yellow]SLOW[/]"
    table.add_row(
        "[bold]Total End-to-End Latency[/]",
        f"[bold]{latency.total_ms:.2f} ms[/]",
        status_icon
    )
    return table


def render_query_results(response: QueryResponse, show_code: bool = True) -> None:
    """Render search results with syntax highlighting, citations, and scores."""
    if not response.results:
        console.print("[yellow]No relevant code chunks found for this query.[/]")
        return

    console.print(f"\n[bold green]Top {len(response.results)} Citations[/] (from {response.total_candidates_considered} candidates):\n")

    for idx, r in enumerate(response.results, start=1):
        c = r.chunk
        
        # Header info
        title_text = f"#{idx} Citation: [bold yellow]{r.citation}[/] | Lang: [cyan]{c.language}[/]"
        if c.symbol_name:
            title_text += f" | Symbol: [bold]{c.symbol_type.value} {c.symbol_name}[/]"

        # Subtitle scores
        scores_line = f"Score: [bold green]{r.score:.4f}[/]"
        if r.rerank_score is not None:
            scores_line += f" (Rerank: {r.rerank_score:.3f})"
        if r.dense_score is not None:
            scores_line += f" | Dense: {r.dense_score:.3f}"
        if r.sparse_score is not None:
            scores_line += f" | BM25: {r.sparse_score:.3f}"

        panel_content = []
        panel_content.append(scores_line)
        
        if c.context_header:
            panel_content.append(f"[dim]Scope: {c.context_header}[/dim]")
        if c.docstring:
            panel_content.append(f"[italic blue]Docstring: {c.docstring.strip()}[/italic blue]")

        console.print(Panel(
            "\n".join(panel_content),
            title=title_text,
            title_align="left",
            border_style="bright_blue"
        ))

        if show_code:
            lexer = c.language if c.language in ["python", "javascript", "typescript", "go", "rust", "cpp", "c", "java", "sql", "html", "css", "markdown", "yaml", "json"] else "text"
            syntax = Syntax(
                c.content,
                lexer,
                theme="monokai",
                line_numbers=True,
                start_line=c.start_line,
                word_wrap=True
            )
            console.print(syntax)
            console.print("")

    # Display timing table
    console.print(render_latency_table(response.latency))


def render_stats_table(stats: Dict[str, Any]) -> None:
    """Render repository indexing statistics."""
    table = Table(title="Semantic Code Index Statistics", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green", justify="right")

    for k, v in stats.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            val_str = f"{v:.2f}"
        elif isinstance(v, int):
            val_str = f"{v:,}"
        else:
            val_str = str(v)
        table.add_row(label, val_str)

    console.print(table)
