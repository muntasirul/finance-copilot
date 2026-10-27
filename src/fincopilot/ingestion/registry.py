from __future__ import annotations

import importlib.util
import logging
from importlib.metadata import entry_points
from pathlib import Path

from fincopilot.ingestion.base import StatementParser

log = logging.getLogger(__name__)
ENTRY_POINT_GROUP = "fincopilot.parsers"
DETECT_THRESHOLD = 0.5


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, StatementParser] = {}

    def register(self, parser: StatementParser) -> None:
        self._parsers[parser.name] = parser

    def names(self) -> list[str]:
        return sorted(self._parsers)

    def get(self, name: str) -> StatementParser:
        return self._parsers[name]

    def load_entry_points(self) -> None:
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                self.register(ep.load()())
            except Exception:  # noqa: BLE001
                log.exception("failed to load parser plugin %s", ep.name)

    def load_directory(self, directory: Path) -> None:
        """Load ``*.py`` files exposing a ``PARSER`` instance (user-local plugins)."""
        if not directory.exists():
            return
        for file in sorted(directory.glob("*.py")):
            spec = importlib.util.spec_from_file_location(f"user_parser_{file.stem}", file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                self.register(module.PARSER)
            except Exception:  # noqa: BLE001
                log.exception("failed to load user parser %s", file)

    def detect(self, path: Path) -> tuple[StatementParser, float] | None:
        """Pick the parser with the highest confidence above the threshold."""
        best: tuple[StatementParser, float] | None = None
        for p in self._parsers.values():
            try:
                score = p.detect(path)
            except Exception:  # noqa: BLE001
                log.debug("detect() raised in %s", p.name, exc_info=True)
                continue
            if score >= DETECT_THRESHOLD and (best is None or score > best[1]):
                best = (p, score)
        return best


def default_registry(user_dir: Path | None = None) -> ParserRegistry:
    reg = ParserRegistry()
    reg.load_entry_points()
    if not reg.names():  # running from source without an installed dist
        from fincopilot.ingestion.parsers.generic_csv import GenericCSVParser
        from fincopilot.ingestion.parsers.generic_pdf import GenericPDFParser

        reg.register(GenericCSVParser())
        reg.register(GenericPDFParser())
    if user_dir:
        reg.load_directory(user_dir)
    return reg
