"""Test generating and benchmarking 30k+ LOC synthetic codebase."""

import tempfile
from pathlib import Path
from semantic_code_intel.benchmark.bench_suite import BenchmarkRunner
from semantic_code_intel.benchmark.generator import CodebaseGenerator


def test_generator_and_benchmark_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Test with a scaled subset (e.g. 5,000 LOC in quick test)
        runner = BenchmarkRunner(workspace_dir=workspace, target_loc=5000)
        report = runner.run_full_benchmark(num_test_queries=5)

        assert report["dataset"]["total_lines"] >= 4000
        assert report["dataset"]["total_files"] >= 10
        assert report["latency"]["total_e2e"]["mean_ms"] < 1000.0
        assert report["latency"]["total_e2e"]["sub_second_percentage"] == 100.0
        assert report["retrieval_quality"]["mrr"] > 0.0
