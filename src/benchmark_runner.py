"""Reusable benchmark execution engine for NYC benchmark families."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class BenchmarkFamily:
    """Config for a benchmark family entrypoint."""

    key: str
    module: str
    main_callable: str = "main"
    output_dir: Path | None = None
    description: str = ""


class BenchmarkRunner:
    """Shared benchmark runner with module loading and lifecycle hooks."""

    def __init__(self, family: BenchmarkFamily) -> None:
        self.family = family

    def load_dataset(self) -> None:
        """Dataset loading hook owned by the family module."""
        return None

    def build_model_registry(self) -> None:
        """Model registry hook owned by the family module."""
        return None

    def train_eval(self) -> None:
        """Train/eval loop hook owned by the family module."""
        return None

    def write_artifacts(self) -> None:
        """Artifact writing hook owned by the family module."""
        return None

    def report(self) -> None:
        """Reporting hook owned by the family module."""
        return None

    def _resolve_main(self) -> Callable[[], None]:
        module = import_module(self.family.module)
        entrypoint = getattr(module, self.family.main_callable, None)
        if entrypoint is None:
            raise AttributeError(
                f"Benchmark family '{self.family.key}' missing callable "
                f"'{self.family.main_callable}' in module '{self.family.module}'."
            )
        return entrypoint

    def run(self) -> None:
        self.load_dataset()
        self.build_model_registry()
        self.train_eval()
        self.write_artifacts()
        self.report()
        self._resolve_main()()
