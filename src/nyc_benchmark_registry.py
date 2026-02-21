"""Registry for NYC benchmark model families."""

from __future__ import annotations

from pathlib import Path

from src.benchmark_runner import BenchmarkFamily, BenchmarkRunner


FAMILY_REGISTRY: dict[str, BenchmarkFamily] = {
    "e0_e8_best_model": BenchmarkFamily(
        key="e0_e8_best_model",
        module="src.benchmark_families.e0_e8_best_model_benchmark",
        output_dir=Path("results/prediction_market_benchmark/e0_e8_best_model_base"),
        description="Best-model-derived E0-E22 benchmark family.",
    ),
    "wga_v2": BenchmarkFamily(
        key="wga_v2",
        module="src.benchmark_families.wga_v2_benchmark",
        output_dir=Path("results/prediction_market_benchmark/wga_v2_model"),
        description="WGA v2 benchmark family.",
    ),
    "unified_outperformance": BenchmarkFamily(
        key="unified_outperformance",
        module="src.benchmark_families.unified_outperformance_benchmark",
        output_dir=Path("results/prediction_market_benchmark/unified_outperformance"),
        description="Unified outperformance benchmark family.",
    ),
}


def run_family(family_key: str) -> None:
    if family_key not in FAMILY_REGISTRY:
        known = ", ".join(sorted(FAMILY_REGISTRY))
        raise KeyError(f"Unknown benchmark family '{family_key}'. Available: {known}")
    BenchmarkRunner(FAMILY_REGISTRY[family_key]).run()
