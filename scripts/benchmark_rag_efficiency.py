#!/usr/bin/env python3
"""Benchmark token efficiency of RAG vs traditional file reading.

This script compares the token cost of:
1. Traditional approach: Reading full files to get context
2. RAG approach: Using semantic search to get targeted chunks

Usage:
    python scripts/benchmark_rag_efficiency.py
    python scripts/benchmark_rag_efficiency.py --project-dir /path/to/project
    python scripts/benchmark_rag_efficiency.py --run-rag  # Include live RAG search

Requirements:
    pip install tiktoken

Example output:
    ## Token Efficiency Report
    | Method | Tokens |
    |--------|--------|
    | Traditional (file read) | 8,500 |
    | RAG-based (search) | 3,200 |
    | **Savings** | **62%** |
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    sys.exit(1)


@dataclass
class TokenReport:
    """Report of token usage for a method."""

    method: str
    total_tokens: int
    details: list[dict[str, Any]]
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "details": self.details,
        }


@dataclass
class ComparisonReport:
    """Comparison between traditional and RAG approaches."""

    test_name: str
    description: str
    traditional: TokenReport
    rag: TokenReport

    @property
    def savings_percent(self) -> float:
        if self.traditional.total_tokens == 0:
            return 0.0
        return (1 - self.rag.total_tokens / self.traditional.total_tokens) * 100

    @property
    def tokens_saved(self) -> int:
        return self.traditional.total_tokens - self.rag.total_tokens


class TokenCounter:
    """Token counter using tiktoken (OpenAI tokenizer)."""

    # GPT-4o pricing (as of Jan 2026)
    PRICING = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    }

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def estimate_cost(self, tokens: int, token_type: str = "input") -> float:
        """Estimate cost in USD."""
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4o"])
        return tokens * pricing.get(token_type, pricing["input"])


def measure_file_read_cost(
    file_paths: list[Path],
    counter: TokenCounter,
) -> TokenReport:
    """Calculate tokens needed to read full files.

    Args:
        file_paths: List of file paths to measure.
        counter: Token counter instance.

    Returns:
        TokenReport with total tokens and per-file breakdown.
    """
    total_tokens = 0
    details = []

    for path in file_paths:
        if not path.exists():
            details.append({"file": str(path), "error": "File not found"})
            continue

        try:
            content = path.read_text(encoding="utf-8")
            tokens = counter.count(content)
            total_tokens += tokens
            details.append({
                "file": str(path.name),
                "path": str(path),
                "lines": len(content.splitlines()),
                "chars": len(content),
                "tokens": tokens,
            })
        except Exception as e:
            details.append({"file": str(path), "error": str(e)})

    return TokenReport(
        method="file_read",
        total_tokens=total_tokens,
        details=details,
        estimated_cost_usd=counter.estimate_cost(total_tokens),
    )


def measure_rag_search_cost(
    search_results: list[dict[str, Any]],
    counter: TokenCounter,
) -> TokenReport:
    """Calculate tokens in RAG search results.

    Args:
        search_results: List of search result dictionaries with 'text' field.
        counter: Token counter instance.

    Returns:
        TokenReport with total tokens and per-result breakdown.
    """
    total_tokens = 0
    details = []

    for result in search_results:
        text = result.get("text", "")
        tokens = counter.count(text)
        total_tokens += tokens
        details.append({
            "name": result.get("name", "unknown"),
            "file": result.get("file_path", ""),
            "chars": len(text),
            "tokens": tokens,
        })

    return TokenReport(
        method="rag_search",
        total_tokens=total_tokens,
        details=details,
        estimated_cost_usd=counter.estimate_cost(total_tokens),
    )


def simulate_rag_results(query: str, files: list[Path]) -> list[dict[str, Any]]:
    """Simulate RAG search results by extracting function signatures.

    This is a simplified simulation. In real usage, actual nexus-dev
    search results would be used.

    Args:
        query: Search query (unused in simulation).
        files: Files to extract "chunks" from.

    Returns:
        Simulated search results.
    """
    results = []
    max_chunk_size = 2000  # Nexus-dev default max chars per result

    for path in files:
        if not path.exists():
            continue

        try:
            content = path.read_text(encoding="utf-8")
            # Simulate chunking: take first max_chunk_size chars
            # Real RAG would return semantic chunks
            chunk = content[:max_chunk_size]
            results.append({
                "name": path.stem,
                "file_path": str(path),
                "text": chunk,
                "chunk_type": "simulated",
            })
        except Exception:
            continue

    # Limit to 5 results (nexus-dev default)
    return results[:5]


def run_benchmark(
    test_cases: list[dict[str, Any]],
    project_dir: Path,
    counter: TokenCounter,
    use_real_rag: bool = False,
) -> list[ComparisonReport]:
    """Run benchmark for all test cases.

    Args:
        test_cases: List of test case definitions.
        project_dir: Project directory for resolving file paths.
        counter: Token counter instance.
        use_real_rag: If True, use actual nexus-dev search.

    Returns:
        List of comparison reports.
    """
    if use_real_rag:
        return asyncio.run(_run_benchmark_async(
            test_cases, project_dir, counter, use_real_rag=True
        ))
    else:
        return _run_benchmark_sync(test_cases, project_dir, counter)


def _run_benchmark_sync(
    test_cases: list[dict[str, Any]],
    project_dir: Path,
    counter: TokenCounter,
) -> list[ComparisonReport]:
    """Run benchmark with simulated RAG (synchronous)."""
    reports = []

    for case in test_cases:
        name = case.get("name", "unknown")
        description = case.get("description", "")
        traditional_files = [
            project_dir / f for f in case.get("traditional_files", [])
        ]
        rag_query = case.get("rag_query", "")

        # Measure traditional approach
        traditional = measure_file_read_cost(traditional_files, counter)

        # Simulate RAG results
        rag_results = simulate_rag_results(rag_query, traditional_files)
        rag = measure_rag_search_cost(rag_results, counter)

        reports.append(ComparisonReport(
            test_name=name,
            description=description,
            traditional=traditional,
            rag=rag,
        ))

    return reports


async def _run_benchmark_async(
    test_cases: list[dict[str, Any]],
    project_dir: Path,
    counter: TokenCounter,
    use_real_rag: bool = True,
) -> list[ComparisonReport]:
    """Run benchmark with real nexus-dev RAG search (async)."""
    # Import nexus-dev modules
    try:
        from nexus_dev.config import NexusConfig
        from nexus_dev.database import DocumentType, NexusDatabase
        from nexus_dev.embeddings import create_embedder
    except ImportError as e:
        print(f"Error: nexus-dev not installed or not in path: {e}")
        print("Install with: pip install -e .")
        sys.exit(1)

    # Load config from project directory
    config_path = project_dir / "nexus_config.json"
    if not config_path.exists():
        print(f"Error: No nexus_config.json found in {project_dir}")
        print("Run 'nexus-init' first to initialize the project.")
        sys.exit(1)

    config = NexusConfig.load(config_path)
    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    reports = []

    for case in test_cases:
        name = case.get("name", "unknown")
        description = case.get("description", "")
        traditional_files = [
            project_dir / f for f in case.get("traditional_files", [])
        ]
        rag_query = case.get("rag_query", "")

        # Measure traditional approach
        traditional = measure_file_read_cost(traditional_files, counter)

        # Real RAG search
        try:
            search_results = await database.search(
                query=rag_query,
                doc_type=DocumentType.CODE,
                limit=5,
            )

            # Convert SearchResult objects to dicts
            rag_results = [
                {
                    "name": r.name,
                    "file_path": r.file_path,
                    "text": r.text,
                    "chunk_type": r.chunk_type,
                    "score": r.score,
                }
                for r in search_results
            ]
        except Exception as e:
            print(f"Warning: RAG search failed for '{name}': {e}")
            rag_results = []

        rag = measure_rag_search_cost(rag_results, counter)

        reports.append(ComparisonReport(
            test_name=name,
            description=description,
            traditional=traditional,
            rag=rag,
        ))

    return reports



def generate_markdown_report(reports: list[ComparisonReport]) -> str:
    """Generate a markdown report from comparison results.

    Args:
        reports: List of comparison reports.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Token Efficiency Benchmark Report",
        "",
        "## Summary",
        "",
        "| Test Case | Traditional | RAG | Savings |",
        "|-----------|-------------|-----|---------|",
    ]

    total_traditional = 0
    total_rag = 0

    for report in reports:
        total_traditional += report.traditional.total_tokens
        total_rag += report.rag.total_tokens
        lines.append(
            f"| {report.test_name} | {report.traditional.total_tokens:,} | "
            f"{report.rag.total_tokens:,} | {report.savings_percent:.1f}% |"
        )

    # Overall savings
    if total_traditional > 0:
        overall_savings = (1 - total_rag / total_traditional) * 100
    else:
        overall_savings = 0

    lines.extend([
        f"| **Total** | **{total_traditional:,}** | **{total_rag:,}** | **{overall_savings:.1f}%** |",
        "",
        "## Cost Estimation (GPT-4o pricing)",
        "",
        "| Metric | Traditional | RAG | Savings |",
        "|--------|-------------|-----|---------|",
        f"| Input tokens | {total_traditional:,} | {total_rag:,} | {total_traditional - total_rag:,} |",
        f"| Cost (USD) | ${total_traditional * 2.5 / 1_000_000:.6f} | "
        f"${total_rag * 2.5 / 1_000_000:.6f} | "
        f"${(total_traditional - total_rag) * 2.5 / 1_000_000:.6f} |",
        "",
    ])

    # Detailed breakdown per test
    lines.extend([
        "## Detailed Breakdown",
        "",
    ])

    for report in reports:
        lines.extend([
            f"### {report.test_name}",
            f"*{report.description}*",
            "",
            "**Traditional (file read):**",
        ])
        for detail in report.traditional.details:
            if "error" in detail:
                lines.append(f"- ❌ {detail['file']}: {detail['error']}")
            else:
                lines.append(
                    f"- 📄 `{detail['file']}`: {detail['lines']} lines, "
                    f"{detail['tokens']:,} tokens"
                )

        lines.extend([
            "",
            "**RAG (semantic search):**",
        ])
        for detail in report.rag.details:
            lines.append(
                f"- 🔍 `{detail['name']}`: {detail['tokens']:,} tokens"
            )

        lines.extend([
            "",
            f"**Savings: {report.savings_percent:.1f}% ({report.tokens_saved:,} tokens)**",
            "",
        ])

    return "\n".join(lines)


def get_default_test_cases() -> list[dict[str, Any]]:
    """Get default test cases for nexus-dev project."""
    return [
        {
            "name": "find_embedding_function",
            "description": "Find the function that generates embeddings",
            "traditional_files": ["src/nexus_dev/embeddings.py"],
            "rag_query": "function that generates embeddings",
        },
        {
            "name": "understand_search_flow",
            "description": "Understand how search works end-to-end",
            "traditional_files": [
                "src/nexus_dev/server.py",
                "src/nexus_dev/database.py",
            ],
            "rag_query": "search semantic similarity database",
        },
        {
            "name": "chunking_strategy",
            "description": "How does code chunking work?",
            "traditional_files": [
                "src/nexus_dev/chunkers/base.py",
                "src/nexus_dev/chunkers/__init__.py",
            ],
            "rag_query": "code chunking extract functions classes",
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark token efficiency of RAG vs traditional file reading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default test cases
    python scripts/benchmark_rag_efficiency.py

    # Specify project directory
    python scripts/benchmark_rag_efficiency.py --project-dir /path/to/nexus-dev

    # Use custom test cases
    python scripts/benchmark_rag_efficiency.py --test-cases tests/benchmark_cases.json

    # Output as JSON
    python scripts/benchmark_rag_efficiency.py --format json
        """,
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        help="Path to JSON file with test cases",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model for token counting (default: gpt-4o)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--run-rag",
        action="store_true",
        help="Use actual nexus-dev search (requires running server)",
    )

    args = parser.parse_args()

    # Load test cases
    if args.test_cases and args.test_cases.exists():
        with open(args.test_cases) as f:
            data = json.load(f)
            test_cases = data.get("test_cases", [])
    else:
        test_cases = get_default_test_cases()

    # Initialize counter
    counter = TokenCounter(model=args.model)

    # Run benchmark
    reports = run_benchmark(
        test_cases=test_cases,
        project_dir=args.project_dir,
        counter=counter,
        use_real_rag=args.run_rag,
    )

    # Generate output
    if args.format == "json":
        output = json.dumps(
            [
                {
                    "test_name": r.test_name,
                    "description": r.description,
                    "traditional": r.traditional.to_dict(),
                    "rag": r.rag.to_dict(),
                    "savings_percent": r.savings_percent,
                    "tokens_saved": r.tokens_saved,
                }
                for r in reports
            ],
            indent=2,
        )
    else:
        output = generate_markdown_report(reports)

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Report written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
