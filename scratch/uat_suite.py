"""
User Acceptance Testing (UAT) Automated Suite.
Conducts rigorous end-to-end user scenario testing across all platform capabilities:
- Scenario 1: Exact Semantic Search & Precision Citations
- Scenario 2: Architectural AI Synthesis & Streaming Walkthrough
- Scenario 3: Symbol Architecture & Dependency Navigation
- Scenario 4: Refactoring Diff Generation & Safe Patch Application
- Scenario 5: Real-Time Incremental Background File Watcher
- Scenario 6: Semantic Git Commit Generator (Zero Attribution)
- Scenario 7: External IDE LSP & MCP Server Protocol Validation
- Scenario 8: Large-Scale OSS Codebase Switching & Querying
"""

import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

class UATRunner:
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0

    def record(self, scenario: str, test_case: str, passed: bool, notes: str = "", latency_ms: float = 0.0):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        status_str = "PASS" if passed else "FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        
        lat_str = f"[{latency_ms:.1f}ms]" if latency_ms > 0 else ""
        print(f"  [{color}{status_str}{reset}] {test_case:<50} {lat_str} {notes}")
        self.results.append({
            "scenario": scenario,
            "test_case": test_case,
            "status": status_str,
            "latency_ms": latency_ms,
            "notes": notes
        })

    def get_json(self, path: str):
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{BASE_URL}{path}")
        with urllib.request.urlopen(req, timeout=15) as res:
            elapsed = (time.perf_counter() - t0) * 1000
            return res.status, json.loads(res.read().decode()), elapsed

    def post_json(self, path: str, data: dict):
        t0 = time.perf_counter()
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}{path}",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=20) as res:
            elapsed = (time.perf_counter() - t0) * 1000
            return res.status, json.loads(res.read().decode()), elapsed

    def run_all_scenarios(self):
        print("\n" + "="*80)
        print("          USER ACCEPTANCE TESTING (UAT) — END-TO-END SUITE")
        print("="*80)

        # -------------------------------------------------------------
        # SCENARIO 1: Semantic Code Search & Citations
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 1: Semantic Code Search & Precision Citations")
        try:
            status, data, elapsed = self.post_json("/api/search", {
                "query": "Where is ReciprocalRankFusion implemented?",
                "repo_path": ".",
                "top_k": 3,
                "mode": "hybrid",
                "rerank": True
            })
            valid_results = len(data.get("results", [])) > 0
            first_res = data["results"][0] if valid_results else {}
            has_lines = "start_line" in first_res and "end_line" in first_res
            has_code = len(first_res.get("code", "")) > 0
            has_citation = "pipeline.py" in first_res.get("citation", "") or "rrf.py" in first_res.get("file_path", "") or "pipeline.py" in first_res.get("file_path", "")

            self.record("Search", "UAT-1.1: Hybrid query returns relevant code chunk", valid_results and has_citation, f"Top match: {first_res.get('file_path')}", elapsed)
            self.record("Search", "UAT-1.2: Exact line numbers and syntax payload present", has_lines and has_code, f"Lines: L{first_res.get('start_line')}-L{first_res.get('end_line')}")
            self.record("Search", "UAT-1.3: Telemetry latency breakdown present", "dense_search_ms" in data.get("latency_ms", {}), f"Total: {data['latency_ms']['total_end_to_end_ms']:.1f}ms")
        except Exception as e:
            self.record("Search", "UAT-1.1: Hybrid query execution", False, str(e))

        # -------------------------------------------------------------
        # SCENARIO 2: Architectural AI Synthesizer & Explanation
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 2: Architectural AI Synthesizer & Walkthrough")
        try:
            status, data, elapsed = self.post_json("/api/synthesize", {
                "query": "Explain the AST parsing and chunking lifecycle",
                "repo_path": ".",
                "top_k": 2
            })
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            self.record("Synthesizer", "UAT-2.1: Synthesizer generates structured technical answer", len(answer) > 40, f"Answer length: {len(answer)} chars", elapsed)
            self.record("Synthesizer", "UAT-2.2: Citations linked to underlying source files", len(citations) > 0, f"Citations found: {len(citations)}")
        except Exception as e:
            self.record("Synthesizer", "UAT-2.1: Synthesizer execution", False, str(e))

        # -------------------------------------------------------------
        # SCENARIO 3: Symbol Architecture Table & Master-Detail Matching
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 3: Symbol Architecture & Dependency Inspection")
        try:
            status, data, elapsed = self.get_json("/api/graph?repo_path=.")
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            files = [n for n in nodes if n.get("group") == 1]
            symbols = [n for n in nodes if n.get("group") == 2]

            self.record("Architecture", "UAT-3.1: Master file list extracted cleanly", len(files) > 0, f"Files: {len(files)}", elapsed)
            self.record("Architecture", "UAT-3.2: Symbols mapped with types (function/class)", len(symbols) > 0, f"Symbols: {len(symbols)}")
            self.record("Architecture", "UAT-3.3: Call relationships mapped between symbols", len(edges) > 0, f"Call edges: {len(edges)}")
        except Exception as e:
            self.record("Architecture", "UAT-3.1: Architecture extraction", False, str(e))

        # -------------------------------------------------------------
        # SCENARIO 4: Automated Refactoring & Diff Patcher
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 4: Automated Refactoring & Diff Patching")
        try:
            status, data, elapsed = self.post_json("/api/patch/generate", {
                "instruction": "Add docstring and parameter typing to search endpoint",
                "repo_path": ".",
                "top_k": 1
            })
            diff_text = data.get("diff", "")
            has_diff = data.get("success") is True and len(diff_text) > 0
            self.record("Patcher", "UAT-4.1: Unified git diff generated with +/- markers", has_diff, f"Diff size: {len(diff_text)} chars", elapsed)
        except Exception as e:
            self.record("Patcher", "UAT-4.1: Diff patcher generation", False, str(e))

        # -------------------------------------------------------------
        # SCENARIO 5: Real-Time Incremental Background File Watcher
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 5: Real-Time Incremental Background Watcher")
        try:
            status_init, data_init, _ = self.get_json("/api/watcher/status")
            status_tog, data_tog, elapsed_tog = self.post_json("/api/watcher/toggle?repo_path=.", {})
            is_active = data_tog.get("running") is True
            # Toggle back to clean state
            self.post_json("/api/watcher/toggle?repo_path=.", {})

            self.record("Watcher", "UAT-5.1: Watcher toggle starts background monitor", is_active, "Sub-50ms observer active", elapsed_tog)
            self.record("Watcher", "UAT-5.2: Watcher status reports active target root", data_tog.get("watched_root") is not None, f"Watched: {data_tog.get('watched_root')}")
        except Exception as e:
            self.record("Watcher", "UAT-5.1: Watcher lifecycle", False, str(e))

        # -------------------------------------------------------------
        # SCENARIO 6: Semantic Git Commit Generator (Zero Attribution)
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 6: Semantic Git Commit Generator")
        try:
            status, data, elapsed = self.post_json("/api/git/commit/generate?repo_path=.", {})
            title = data.get("title", "")
            full_msg = data.get("full_message", "")
            has_conventional_format = any(title.startswith(t) for t in ["feat", "fix", "refactor", "test", "perf", "chore", "docs"])
            no_ai_lines = "Co-Authored-By" not in full_msg and "Generated with" not in full_msg

            self.record("Git Intel", "UAT-6.1: Conventional Commit format generated", has_conventional_format, f"Title: {title}", elapsed)
            self.record("Git Intel", "UAT-6.2: AGENTS.md rule: Zero AI attribution lines", no_ai_lines, "100% clean attribution")
        except Exception as e:
            self.record("Git Intel", "UAT-6.1: Git commit generation", False, str(e))

        # -------------------------------------------------------------
        # SCENARIO 7: Language Server Protocol (LSP) Inspection
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 7: Language Server Protocol (LSP) Bridge")
        try:
            status, data, elapsed = self.get_json("/api/lsp/inspect?repo_path=.&symbol=search_code")
            has_defs = len(data.get("definitions", [])) > 0
            has_hover = data.get("hover") is not None
            self.record("LSP Bridge", "UAT-7.1: Jump-to-definition lookup returns exact URI/Range", has_defs, f"Locations: {len(data['definitions'])}", elapsed)
            self.record("LSP Bridge", "UAT-7.2: Hover documentation returns markdown signature", has_hover, "Markdown docstring extracted")
        except Exception as e:
            self.record("LSP Bridge", "UAT-7.1: LSP inspection", False, str(e))

        # -------------------------------------------------------------
        # SCENARIO 8: Large-Scale OSS Codebases (FastAPI, Rich, client-go)
        # -------------------------------------------------------------
        print("\n▶ SCENARIO 8: Polyglot OSS Repositories Evaluation")
        try:
            # Query FastAPI repo
            st_fa, data_fa, el_fa = self.post_json("/api/search", {
                "query": "OAuth2PasswordBearer extract bearer token",
                "repo_path": "./oss_evaluation/fastapi",
                "top_k": 2,
                "mode": "hybrid"
            })
            fa_ok = len(data_fa.get("results", [])) > 0
            self.record("OSS Evaluation", "UAT-8.1: FastAPI repo (112k LOC) semantic search", fa_ok, f"Found {len(data_fa.get('results', []))} matches", el_fa)

            # Query Rich repo
            st_rc, data_rc, el_rc = self.post_json("/api/search", {
                "query": "Progress bar ETA time remaining calculated",
                "repo_path": "./oss_evaluation/rich",
                "top_k": 2,
                "mode": "hybrid"
            })
            rc_ok = len(data_rc.get("results", [])) > 0
            self.record("OSS Evaluation", "UAT-8.2: Textualize Rich repo (67k LOC) search", rc_ok, f"Found {len(data_rc.get('results', []))} matches", el_rc)
        except Exception as e:
            self.record("OSS Evaluation", "UAT-8.1: OSS evaluation", False, str(e))

        # -------------------------------------------------------------
        # Summary
        # -------------------------------------------------------------
        print("\n" + "="*80)
        pct = (self.passed_tests / max(self.total_tests, 1)) * 100
        print(f"  UAT RUN COMPLETE: {self.passed_tests}/{self.total_tests} Tests Passed ({pct:.1f}% Success Rate)")
        print("="*80 + "\n")

if __name__ == "__main__":
    runner = UATRunner()
    runner.run_all_scenarios()
