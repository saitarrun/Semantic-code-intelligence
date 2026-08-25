"""
Realistic multi-language synthetic codebase generator for 30,000+ LOC benchmarking.
Generates interconnected modules: Auth, Payments, Inventory, Search, ML Pipeline, Analytics, Database.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple


class CodebaseGenerator:
    """Generates synthetic multi-language codebases for stress-testing and benchmarking."""

    MODULES = [
        "authentication", "authorization", "billing_engine", "payment_gateway",
        "inventory_manager", "order_processing", "notification_service", "user_profile",
        "search_indexer", "recommendation_ai", "analytics_pipeline", "rate_limiter",
        "cache_manager", "database_pool", "event_stream", "metrics_collector",
        "logging_infra", "distributed_lock", "tenant_manager", "audit_logger",
        "crypto_vault", "webhook_dispatcher", "file_storage", "job_scheduler",
        "api_gateway", "session_store", "feature_flags", "circuit_breaker",
        "export_service", "report_generator"
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_python_module(self, module_name: str, target_lines: int) -> Tuple[str, List[Dict[str, str]]]:
        """Generate a realistic Python module with classes, docstrings, typing, and methods."""
        lines: List[str] = []
        queries: List[Dict[str, str]] = []

        lines.append(f'"""')
        lines.append(f'Module: {module_name}')
        lines.append(f'Provides high-performance distributed logic for {module_name.replace("_", " ")}.')
        lines.append(f'"""\n')
        lines.append('from __future__ import annotations')
        lines.append('import time')
        lines.append('import hashlib')
        lines.append('import logging')
        lines.append('from typing import Any, Dict, List, Optional, Union, Tuple')
        lines.append('from dataclasses import dataclass, field\n')
        lines.append(f'logger = logging.getLogger(__name__)\n\n')

        class_count = max(2, target_lines // 150)
        for c_idx in range(class_count):
            class_name = f"{''.join(p.capitalize() for p in module_name.split('_'))}Handler{c_idx}"
            lines.append(f'@dataclass')
            lines.append(f'class {class_name}:')
            lines.append(f'    """Service handler for managing {module_name} domain operations."""')
            lines.append(f'    service_id: str = "{module_name}_{c_idx}"')
            lines.append(f'    timeout_seconds: float = 30.0')
            lines.append(f'    is_active: bool = True')
            lines.append(f'    metrics: Dict[str, float] = field(default_factory=dict)\n')

            method_count = max(4, target_lines // (class_count * 20))
            for m_idx in range(method_count):
                method_name = f"execute_{module_name}_step_{m_idx}"
                
                lines.append(f'    def {method_name}(self, payload: Dict[str, Any], retry_count: int = 3) -> Dict[str, Any]:')
                lines.append(f'        """Process {module_name} step {m_idx} with retry logic and idempotency tracking."""')
                lines.append(f'        if not self.is_active:')
                lines.append(f'            raise RuntimeError("Service {class_name} is currently inactive")')
                lines.append(f'        ')
                lines.append(f'        transaction_id = hashlib.sha256(f"{{payload}}_{{time.time()}}".encode()).hexdigest()[:16]')
                lines.append(f'        logger.debug(f"Executing step {m_idx} for transaction {{transaction_id}}")')
                lines.append(f'        ')
                lines.append(f'        # Compute simulated cryptographic checkpoint')
                lines.append(f'        checkpoint = sum(ord(c) for c in transaction_id) % 1000')
                lines.append(f'        self.metrics["last_checkpoint_{m_idx}"] = float(checkpoint)')
                lines.append(f'        ')
                lines.append(f'        results = {{')
                lines.append(f'            "status": "SUCCESS",')
                lines.append(f'            "step": {m_idx},')
                lines.append(f'            "module": "{module_name}",')
                lines.append(f'            "transaction_id": transaction_id,')
                lines.append(f'            "checkpoint_value": checkpoint,')
                lines.append(f'            "execution_timestamp": time.time()')
                lines.append(f'        }}')
                lines.append(f'        return results\n')

                queries.append({
                    "query": f"How does {class_name} execute {module_name} step {m_idx}?",
                    "target_symbol": method_name,
                    "target_class": class_name,
                    "module": module_name
                })

        # Add standalone helper functions
        lines.append(f'def validate_{module_name}_configuration(config_dict: Dict[str, Any]) -> bool:')
        lines.append(f'    """Validate configuration parameters for {module_name} subsystem."""')
        lines.append(f'    required_keys = ["host", "port", "timeout", "cluster_mode"]')
        lines.append(f'    for key in required_keys:')
        lines.append(f'        if key not in config_dict:')
        lines.append(f'            logger.error(f"Missing configuration parameter: {{key}}")')
        lines.append(f'            return False')
        lines.append(f'    return True\n')

        queries.append({
            "query": f"Validate {module_name} configuration parameters and required keys",
            "target_symbol": f"validate_{module_name}_configuration",
            "target_class": "",
            "module": module_name
        })

        content = "\n".join(lines)
        return content, queries

    def generate_typescript_module(self, module_name: str, target_lines: int) -> str:
        """Generate a TypeScript service file."""
        lines: List[str] = []
        name_camel = ''.join(p.capitalize() for p in module_name.split('_'))
        
        lines.append(f'/**')
        lines.append(f' * TypeScript Client for {module_name}')
        lines.append(f' */\n')
        lines.append(f'export interface {name_camel}Config {{')
        lines.append(f'  endpoint: string;')
        lines.append(f'  apiKey: string;')
        lines.append(f'  maxRetries: number;')
        lines.append(f'  timeoutMs: number;')
        lines.append(f'}}\n')

        lines.append(f'export interface {name_camel}Response<T> {{')
        lines.append(f'  success: boolean;')
        lines.append(f'  data?: T;')
        lines.append(f'  errorCode?: string;')
        lines.append(f'  latencyMs: number;')
        lines.append(f'}}\n')

        lines.append(f'export class {name_camel}Service {{')
        lines.append(f'  private config: {name_camel}Config;')
        lines.append(f'  private cache: Map<string, any> = new Map();\n')
        lines.append(f'  constructor(config: {name_camel}Config) {{')
        lines.append(f'    this.config = config;')
        lines.append(f'  }}\n')

        method_count = max(4, target_lines // 25)
        for m in range(method_count):
            lines.append(f'  /**')
            lines.append(f'   * Dispatches async operation for {module_name} phase {m}')
            lines.append(f'   */')
            lines.append(f'  public async dispatch{name_camel}Phase{m}(reqId: string, payload: Record<string, any>): Promise<{name_camel}Response<any>> {{')
            lines.append(f'    const startTime = Date.now();')
            lines.append(f'    if (this.cache.has(reqId)) {{')
            lines.append(f'      return {{ success: true, data: this.cache.get(reqId), latencyMs: Date.now() - startTime }};')
            lines.append(f'    }}')
            lines.append(f'    const result = {{ id: reqId, phase: {m}, processed: true, timestamp: Date.now() }};')
            lines.append(f'    this.cache.set(reqId, result);')
            lines.append(f'    return {{ success: true, data: result, latencyMs: Date.now() - startTime }};')
            lines.append(f'  }}\n')

        lines.append('}')
        return "\n".join(lines)

    def generate_go_module(self, module_name: str, target_lines: int) -> str:
        """Generate a Go service file."""
        lines: List[str] = []
        name_camel = ''.join(p.capitalize() for p in module_name.split('_'))
        pkg_name = module_name.replace("_", "")

        lines.append(f'package {pkg_name}\n')
        lines.append('import (')
        lines.append('\t"context"')
        lines.append('\t"fmt"')
        lines.append('\t"sync"')
        lines.append('\t"time"')
        lines.append(')\n')

        lines.append(f'type {name_camel}Manager struct {{')
        lines.append('\tmu sync.RWMutex')
        lines.append('\tisRunning bool')
        lines.append(f'\tmetrics map[string]int64')
        lines.append('}\n')

        lines.append(f'func New{name_camel}Manager() *{name_camel}Manager {{')
        lines.append(f'\treturn &{name_camel}Manager{{')
        lines.append('\t\tisRunning: true,')
        lines.append('\t\tmetrics: make(map[string]int64),')
        lines.append('\t}')
        lines.append('}\n')

        method_count = max(4, target_lines // 20)
        for m in range(method_count):
            lines.append(f'// Process{name_camel}Task{m} executes concurrent Go pipeline step {m}')
            lines.append(f'func (m *{name_camel}Manager) Process{name_camel}Task{m}(ctx context.Context, taskId string) (string, error) {{')
            lines.append('\tm.mu.Lock()')
            lines.append('\tdefer m.mu.Unlock()')
            lines.append('\tif !m.isRunning {')
            lines.append(f'\t\treturn "", fmt.Errorf("{name_camel}Manager is stopped")')
            lines.append('\t}')
            lines.append(f'\tm.metrics["task_{m}"]++')
            lines.append(f'\treturn fmt.Sprintf("OK_%s_{m}", taskId), nil')
            lines.append('}\n')

        return "\n".join(lines)

    def generate_codebase(
        self,
        output_dir: Path,
        target_loc: int = 35000
    ) -> Tuple[int, int, List[Dict[str, str]]]:
        """
        Generate a codebase with target lines of code (default >35,000 LOC).
        Returns (total_files, total_lines, benchmark_queries).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        total_lines = 0
        total_files = 0
        all_queries: List[Dict[str, str]] = []

        lines_per_module = max(100, target_loc // len(self.MODULES))

        for module_name in self.MODULES:
            mod_dir = output_dir / "src" / module_name
            mod_dir.mkdir(parents=True, exist_ok=True)

            # 1. Python implementation (60% of lines)
            py_loc = int(lines_per_module * 0.60)
            py_content, py_queries = self.generate_python_module(module_name, py_loc)
            py_file = mod_dir / f"{module_name}_service.py"
            py_file.write_text(py_content, encoding="utf-8")
            
            py_line_count = len(py_content.splitlines())
            total_lines += py_line_count
            total_files += 1

            for q in py_queries:
                q["expected_file"] = str(py_file.relative_to(output_dir))
                all_queries.append(q)

            # 2. TypeScript client (25% of lines)
            ts_loc = int(lines_per_module * 0.25)
            ts_content = self.generate_typescript_module(module_name, ts_loc)
            ts_file = mod_dir / f"{module_name}Client.ts"
            ts_file.write_text(ts_content, encoding="utf-8")
            total_lines += len(ts_content.splitlines())
            total_files += 1

            # 3. Go backend component (15% of lines)
            go_loc = int(lines_per_module * 0.15)
            go_content = self.generate_go_module(module_name, go_loc)
            go_file = mod_dir / f"{module_name}_handler.go"
            go_file.write_text(go_content, encoding="utf-8")
            total_lines += len(go_content.splitlines())
            total_files += 1

        # Add documentation & configs
        doc_file = output_dir / "ARCHITECTURE.md"
        doc_file.write_text(
            f"# Enterprise Platform Architecture\n\nGenerated benchmark repository containing {total_lines} lines of code.\n"
            f"Modules: {', '.join(self.MODULES)}.\n",
            encoding="utf-8"
        )
        total_files += 1
        total_lines += len(doc_file.read_text().splitlines())

        return total_files, total_lines, all_queries
