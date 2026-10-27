# ADR-003: Statement parsers as plugins

**Status:** Accepted  **Date:** 2026-10-28

## Context
Every bank emits a different PDF/CSV layout. Hard-coding them in core would make the project
unmaintainable and block community contributions.

## Decision
A `StatementParser` abstract base class with `detect(path) -> float` (confidence 0–1) and
`parse(path) -> list[ParsedTransaction]`. Parsers are discovered via the Python entry-point group
`fincopilot.parsers` plus a local `~/.fincopilot/parsers/` directory. The registry picks the
highest-confidence parser above a threshold, falling back to generic CSV/PDF parsers.

## Consequences
- New banks = one file + one entry-point line, covered by golden-file tests.
- Core never imports bank-specific code.
- Generic PDF parser keeps an OCR fallback flag for scanned statements.
