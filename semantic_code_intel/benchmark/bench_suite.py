"""
Automated benchmark runner executing full end-to-end performance benchmarks on 30,000+ LOC codebases.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from semantic_code_intel.benchmark.generator import CodebaseGenerator
from semantic_code_intel.benchmark.metrics import compute_latency_stats, evaluate_retrieval
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline

console = Console()
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs end-to-end indexing and retrieval benchmark on 30,000+ LOC codebases."""

    def __init__(self, workspace_dir: Path, target_loc: int = 35000):
        self.workspace_dir = workspace_dir.resolve()
        self.target_loc = target_loc
        self.codebase_dir = self.workspace_dir / "benchmark_codebase_30k"
        self.index_dir = self.workspace_dir / "benchmark_index_30k"
        self.config = CodeIntelConfig(
            project_root=self.codebase_dir,
            index_dir=self.index_dir
        )

    def run_full_benchmark(self, num_test_queries: int = 50) -> Dict[str, Any]:
        """Execute complete benchmark: code generation -> indexing -> query latency & quality evaluation."""
        console.print(Panel(
            f"[bold cyan]Starting Benchmark Suite: Target ~{self.target_loc:,} Lines of Code[/]\n"
            f"Codebase Path: [yellow]{self.codebase_dir}[/]\n"
            f"Index Storage: [yellow]{self.index_dir}[/]",
            title="Semantic Code Intelligence Benchmark",
            border_style="bright_blue"
        ))

        # 1. Generate Synthetic Codebase
        console.print("\n[bold cyan]1. Generating realistic multi-module codebase...[/]")
        gen_start = time.time()
        generator = CodebaseGenerator()
        total_files, total_lines, generated_queries = generator.generate_codebase(
            self.codebase_dir, target_loc=self.target_loc
        )
        gen_elapsed = time.time() - gen_start
        console.print(f"[green]✓ Generated {total_files} files, {total_lines:,} lines of code in {gen_elapsed:.2f}s[/]")

        # 2. Benchmark Hybrid Indexing
        console.print("\n[bold cyan]2. Indexing codebase (FAISS + BM25 + SQLite)...[/]")
        indexer = HybridIndexer(self.config)
        idx_metrics = indexer.index_codebase(target_dir=self.codebase_dir, force_reindex=True)
        console.print(
            f"[green]✓ Indexed {int(idx_metrics['total_chunks']):,} chunks across "
            f"{int(idx_metrics['total_lines']):,} lines in {idx_metrics['elapsed_seconds']:.2f}s "
            f"({int(idx_metrics['total_lines'] / max(0.01, idx_metrics['elapsed_seconds'])):,} LOC/sec)[/]"
        )

        # 3. Benchmark Query Retrieval Latency & IR Metrics
        console.print(f"\n[bold cyan]3. Executing {num_test_queries} retrieval queries with Cross-Encoder reranking...[/]")
        pipeline = HybridRetrievalPipeline(self.config)
        
        # Warmup query
        pipeline.query("validate configuration parameters", top_k=5, use_reranker=True)

        test_queries = generated_queries[:num_test_queries]
        total_latencies: List[float] = []
        dense_latencies: List[float] = []
        sparse_latencies: List[float] = []
        fusion_latencies: List[float] = []
        rerank_latencies: List[float] = []
        eval_records: List[Dict[str, Any]] = []

        for q in test_queries:
            query_str = q["query"]
            res = pipeline.query(query_str, top_k=5, use_reranker=True)
            
            total_latencies.append(res.latency.total_ms)
            dense_latencies.append(res.latency.dense_ms)
            sparse_latencies.append(res.latency.sparse_ms)
            fusion_latencies.append(res.latency.fusion_ms)
            rerank_latencies.append(res.latency.rerank_ms)

            retrieved_files = [r.chunk.file_path for r in res.results]
            retrieved_symbols = [r.chunk.symbol_name or "" for r in res.results]

            eval_records.append({
                "query": query_str,
                "expected_file": q.get("expected_file", ""),
                "target_symbol": q.get("target_symbol", ""),
                "retrieved_files": retrieved_files,
                "retrieved_symbols": retrieved_symbols
            })

        latency_stats = compute_latency_stats(total_latencies)
        dense_stats = compute_latency_stats(dense_latencies)
        sparse_stats = compute_latency_stats(sparse_latencies)
        rerank_stats = compute_latency_stats(rerank_latencies)
        ir_metrics = evaluate_retrieval(eval_records)

        # Render Benchmark Tables
        self._render_results_tables(
            total_files=total_files,
            total_lines=total_lines,
            idx_metrics=idx_metrics,
            latency_stats=latency_stats,
            dense_stats=dense_stats,
            sparse_stats=sparse_stats,
            rerank_stats=rerank_stats,
            ir_metrics=ir_metrics
        )

        results_payload = {
            "dataset": {
                "total_files": total_files,
                "total_lines": total_lines,
                "total_chunks": int(idx_metrics["total_chunks"]),
            },
            "indexing": idx_metrics,
            "latency": {
                "total_e2e": latency_stats.model_dump(),
                "dense_faiss": dense_stats.model_dump(),
                "sparse_bm25": sparse_stats.model_dump(),
                "cross_encoder_rerank": rerank_stats.model_dump(),
            },
            "retrieval_quality": ir_metrics.model_dump()
        }

        # Save JSON Report
        report_path = self.workspace_dir / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)

        console.print(f"\n[bold green]✓ Benchmark report saved to:[/] [yellow]{report_path}[/]\n")
        return results_payload

    def _render_results_tables(
        self,
        total_files: int,
        total_lines: int,
        idx_metrics: Dict[str, float],
        latency_stats: Any,
        dense_stats: Any,
        sparse_stats: Any,
        rerank_stats: Any,
        ir_metrics: Any
    ) -> None:
        # Table 1: Codebase & Indexing
        t1 = Table(title="1. Codebase & Indexing Performance", show_header=True, header_style="bold magenta")
        t1.add_column("Metric", style="cyan")
        t1.add_column("Value", style="bold green", justify="right")
        t1.add_row("Total Source Files", f"{total_files:,}")
        t1.add_row("Total Lines of Code (LOC)", f"{total_lines:,}")
        t1.add_row("Total Semantic Chunks", f"{int(idx_metrics['total_chunks']):,}")
        t1.add_row("Total Indexing Time", f"{idx_metrics['elapsed_seconds']:.2f} s")
        t1.add_row("Parsing Time", f"{idx_metrics['parse_time_seconds']:.2f} s")
        t1.add_row("FAISS Dense Embed Time", f"{idx_metrics['embed_time_seconds']:.2f} s")
        t1.add_row("BM25 Tokenize & Index Time", f"{idx_metrics['bm25_time_seconds']:.2f} s")
        t1.add_row(
            "Indexing Throughput",
            f"{int(total_lines / max(0.01, idx_metrics['elapsed_seconds'])):,} LOC/s"
        )
        console.print(t1)

        # Table 2: Latency Percentiles
        t2 = Table(title="2. Sub-Second Retrieval Latency Percentiles (N=50)", show_header=True, header_style="bold blue")
        t2.add_column("Component", style="cyan")
        t2.add_column("p50 (ms)", justify="right")
        t2.add_column("p90 (ms)", justify="right")
        t2.add_column("p95 (ms)", justify="right")
        t2.add_column("p99 (ms)", justify="right")
        t2.add_column("Mean (ms)", justify="right", style="bold green")
        t2.add_column("Sub-Second Pass", justify="center", style="bold green")

        t2.add_row(
            "FAISS Dense Vector",
            f"{dense_stats.p50_ms:.1f}", f"{dense_stats.p90_ms:.1f}",
            f"{dense_stats.p95_ms:.1f}", f"{dense_stats.p99_ms:.1f}",
            f"{dense_stats.mean_ms:.1f}", "✓"
        )
        t2.add_row(
            "BM25 Sparse Lexical",
            f"{sparse_stats.p50_ms:.1f}", f"{sparse_stats.p90_ms:.1f}",
            f"{sparse_stats.p95_ms:.1f}", f"{sparse_stats.p99_ms:.1f}",
            f"{sparse_stats.mean_ms:.1f}", "✓"
        )
        t2.add_row(
            "Cross-Encoder Reranker",
            f"{rerank_stats.p50_ms:.1f}", f"{rerank_stats.p90_ms:.1f}",
            f"{rerank_stats.p95_ms:.1f}", f"{rerank_stats.p99_ms:.1f}",
            f"{rerank_stats.mean_ms:.1f}", "✓"
        )
        t2.add_row(
            "[bold]End-to-End Hybrid Search[/]",
            f"[bold]{latency_stats.p50_ms:.1f}[/]", f"[bold]{latency_stats.p90_ms:.1f}[/]",
            f"[bold]{latency_stats.p95_ms:.1f}[/]", f"[bold]{latency_stats.p99_ms:.1f}[/]",
            f"[bold]{latency_stats.mean_ms:.1f}[/]",
            f"[bold green]{latency_stats.sub_second_percentage}%[/]"
        )
        console.print(t2)

        # Table 3: Retrieval Quality
        t3 = Table(title="3. Information Retrieval (IR) Accuracy & Ranking Quality", show_header=True, header_style="bold green")
        t3.add_column("Evaluation Metric", style="cyan")
        t3.add_column("Score", style="bold green", justify="right")
        t3.add_row("Hit Rate @ 1 (Top-1 Match)", f"{ir_metrics.hit_rate_at_1 * 100:.1f}%")
        t3.add_row("Hit Rate @ 3 (Top-3 Match)", f"{ir_metrics.hit_rate_at_3 * 100:.1f}%")
        t3.add_row("Hit Rate @ 5 (Top-5 Match)", f"{ir_metrics.hit_rate_at_5 * 100:.1f}%")
        t3.add_row("Mean Reciprocal Rank (MRR)", f"{ir_metrics.mrr:.4f}")
        console.print(t3)
