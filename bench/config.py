"""
Benchmark configuration: constants, RunConfig, BenchmarkRun, ResultsStore.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Script directory for absolute paths
_SCRIPT_DIR = Path(__file__).parent.parent.resolve()

# Embedding models (OpenAI only - Voyage removed)
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_EMBED_DIMS = 1536

# Token budgets to evaluate
TOKEN_BUDGETS = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000]

# Naive chunking config
NAIVE_SEPARATORS = ["\n\n", "\n", " ", "", "#", "##", "###"]
NAIVE_CHUNK_SIZE = 500
NAIVE_CHUNK_OVERLAP = 100

# Context formatting
NAIVE_CHUNK_SEPARATOR = "\n\n---\n\n"
CHEATSHEET_SEPARATOR = "\n\n"
CHUNK_HEADER_TEMPLATE = "[Source {source_id}]\n"

# Budget enforcement
MAX_BUDGET_OVERAGE = 100

# Retrieval config (pure semantic - no fulltext/BM25/RRF)
EXTENDED_K = 30000  # Max chunksets to rank

# Debug flags
SKIP_CHEATSHEET_DEDUP = False

# Paths
VECTORS_DIR = _SCRIPT_DIR / "vectors"
RESULTS_DIR = _SCRIPT_DIR / "results" / "bench_results"

# Data paths
TRANSFORMED_DIR = _SCRIPT_DIR / "data" / "databricks"
UNSTRUCTURED_DIR = _SCRIPT_DIR / "data" / "unstructured"
EVIDENCE_FILE = _SCRIPT_DIR / "data" / "evidence_requirements.json"


@dataclass
class RunConfig:
    """Immutable configuration that defines a benchmark run."""
    extended_k: int
    eval_mode: str
    budgets: tuple[int, ...]
    weight: float = 0.5  # Balanced scoring weight
    agg: str = "max_adj"  # Aggregation method

    def to_dict(self) -> dict:
        return {
            "extended_k": self.extended_k,
            "eval_mode": self.eval_mode,
            "budgets": list(self.budgets),
            "weight": self.weight,
            "agg": self.agg,
        }

    def config_hash(self) -> str:
        s = f"{self.extended_k}_{self.eval_mode}_{self.weight}_{self.agg}"
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def short_name(self) -> str:
        return f"w{self.weight}_{self.agg}"


@dataclass 
class BenchmarkRun:
    """Complete benchmark run with config, corpus info, and results."""
    run_id: str
    timestamp: str
    config: RunConfig
    corpus: dict
    results: dict[str, dict[int, dict]]
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "corpus": self.corpus,
            "results": {
                method: {str(b): r for b, r in budgets.items()}
                for method, budgets in self.results.items()
            }
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkRun":
        config = RunConfig(
            extended_k=d["config"].get("extended_k", 30000),
            eval_mode=d["config"].get("eval_mode", "index_based"),
            budgets=tuple(d["config"].get("budgets", [])),
            weight=d["config"].get("weight", 0.5),
            agg=d["config"].get("agg", "max_adj"),
        )
        results = {
            method: {int(b): r for b, r in budgets.items()}
            for method, budgets in d["results"].items()
        }
        return cls(
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            config=config,
            corpus=d["corpus"],
            results=results,
        )


class ResultsStore:
    """Simple file-based results store for benchmark runs."""
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, run: BenchmarkRun) -> Path:
        filename = f"{run.run_id}.json"
        filepath = self.results_dir / filename
        with open(filepath, "w") as f:
            json.dump(run.to_dict(), f, indent=2)
        return filepath
    
    def load(self, run_id: str) -> Optional[BenchmarkRun]:
        filepath = self.results_dir / f"{run_id}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            return BenchmarkRun.from_dict(json.load(f))
    
    def list_runs(self) -> list[BenchmarkRun]:
        runs = []
        for filepath in self.results_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    runs.append(BenchmarkRun.from_dict(json.load(f)))
            except Exception:
                continue
        return sorted(runs, key=lambda r: r.timestamp, reverse=True)
    
    def find_by_config(self, config: RunConfig) -> list[BenchmarkRun]:
        target_hash = config.config_hash()
        return [r for r in self.list_runs() if r.config.config_hash() == target_hash]


def create_run_id() -> str:
    """Create unique run ID with timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"{ts}_{rand}"


def print_results_comparison(runs: list[BenchmarkRun], budget: int = 20000):
    """Print comparison table for multiple runs at a specific budget."""
    if not runs:
        print("No runs to compare.")
        return
    
    all_methods = set()
    for run in runs:
        all_methods.update(run.results.keys())
    methods = sorted(all_methods)
    
    budget_key = str(budget)
    
    print(f"\n{'='*100}")
    print(f"RESULTS COMPARISON @ {budget} tokens")
    print(f"{'='*100}")
    header = f"{'Config':<25}"
    for method in methods:
        header += f" {method[:12]:>12}"
    print(header)
    print(f"{'-'*100}")
    
    for run in runs:
        row = f"{run.config.short_name():<25}"
        for method in methods:
            result = run.results.get(method, {})
            data = result.get(budget_key) or result.get(budget)
            if data:
                acc = data.get("accuracy", 0) * 100
                row += f" {acc:>11.0f}%"
            else:
                row += f" {'N/A':>12}"
        print(row)
    
    print(f"{'='*100}")
