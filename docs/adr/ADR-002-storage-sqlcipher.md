# ADR-002: Storage — SQLite with SQLCipher, key derived via Argon2id

**Status:** Accepted  **Date:** 2026-10-28

## Context
All financial data is stored locally. It must be encrypted at rest, zero-config, portable, and
fast enough for analytics over ~1M rows.

## Decision
SQLite via SQLAlchemy 2. In production builds the driver is `sqlcipher3`; in dev/test the
plain `sqlite3` driver is used with the same schema (switchable via `FINCOPILOT_DB_DRIVER`).
The encryption key is derived from the user passphrase with Argon2id (salt stored next to the DB).
Analytics use materialised SQL views refreshed after each import.

## Consequences
- Single-file, encrypted database; backups are just file copies.
- No server process; forecasting/analytics run in-process with pandas over SQL extracts.
- The LLM assistant receives a **read-only** connection and may only query whitelisted views.
