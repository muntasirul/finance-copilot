# ADR-001: Desktop shell — Tauri with a Python sidecar

**Status:** Accepted  **Date:** 2026-10-28

## Context
The app must run fully offline on macOS/Windows/Linux, bundle ML models, and keep the ML stack in
Python. Options considered: Electron + Python, Tauri + Python sidecar, pure Python GUI (PyQt),
Streamlit.

## Decision
Tauri 2 for the shell and React/TypeScript UI. The Python backend (FastAPI) is bundled with
PyInstaller and spawned by Tauri as a child process bound to `127.0.0.1:<random port>`. Tauri
generates a per-launch bearer token and passes it via environment; every request must carry it.

## Consequences
- Small binaries, native webview, Rust-side sandboxing.
- Two toolchains (Rust/Node + Python) — mitigated by `make` targets and CI matrix.
- Packaging the sidecar must be prototyped early (Phase 0/4) to avoid late surprises.
