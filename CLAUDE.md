# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenClaw plugin providing LanceDB-backed long-term memory for AI agents. Features hybrid retrieval (Vector + BM25), cross-encoder reranking, multi-scope isolation, LLM-powered smart extraction, Weibull decay model, and a management CLI.

## Commands

### Tests
```bash
npm test                      # Run full suite (27+ test files, ~2 min)
npm run test:openclaw-host    # Functional test against live OpenClaw host
```

There is no single-test runner configured. Run individual tests directly:
```bash
node test/functional-e2e.mjs                        # Self-contained test script
node --test test/scope-access-undefined.test.mjs     # Uses node:test runner
```

Tests use two patterns:
- `node --test <file>` — files using `node:test` (describe/it). These have `*.test.mjs` names.
- `node <file>` — self-contained scripts with manual assertions. These have `*.mjs` names without `.test`.

All `.ts` imports in tests go through `jiti` for on-the-fly transpilation.

### Version Sync
```bash
npm version <semver>   # Syncs version between package.json and openclaw.plugin.json
```
CI enforces version parity between these two files.

### No Build Step
Source is TypeScript but there is no compilation step. The plugin runs `.ts` files directly via the OpenClaw runtime. Tests use `jiti` to import `.ts` sources.

## Architecture

**Entry point:** `index.ts` — Plugin registration, lifecycle hooks, tool registration, auto-capture/recall pipelines.

### Core Components (all in `src/`)

| Module | Responsibility |
|--------|---------------|
| `store.ts` | LanceDB persistence layer — vector search, BM25, scope filtering, metadata patching |
| `retriever.ts` | Hybrid retrieval with RRF fusion, composite scoring (time decay, recency boost, length normalization, importance), cross-encoder reranking |
| `tools.ts` | Agent tool definitions (recall, store, forget, update, list, stats) registered via OpenClaw plugin API |
| `smart-extractor.ts` | LLM-powered memory extraction pipeline — candidate generation, dedup, merge, noise filtering |
| `embedder.ts` | Embedding abstraction with LRU cache, long-context chunking, task-aware embeddings |
| `scopes.ts` | Multi-tenant scope isolation (global, agent:{id}, user:{id}, project:{id}, reflection, custom) |
| `smart-metadata.ts` | Unified metadata layer (L0/L1/L2 format), lifecycle tracking, relation links |
| `decay-engine.ts` | Weibull stretched-exponential decay with per-tier parameters |
| `tier-manager.ts` | Memory tier evolution: Core → Working → Peripheral |
| `llm-client.ts` / `llm-oauth.ts` | LLM API client with API key and OAuth2 auth |
| `noise-filter.ts` / `noise-prototypes.ts` | Noise detection before storage |
| `reflection-store.ts` | Session reflection persistence |
| `cli.ts` (root) | Commander-based management CLI |

### Data Flow
```
Agent conversation → Auto-capture → Smart Extractor (LLM) → Noise Filter
  → Embedder → LanceDB Store (scoped)

Recall query → Embedder → Hybrid search (Vector + BM25) → RRF Fusion
  → Composite scoring → Reranker → Ranked results
```

### Key Design Patterns

- **Scope-aware everything:** All store/retrieve operations filter by scope. The `MemoryScopeManager` enforces access boundaries. System/admin agents bypass scope checks.
- **Smart metadata unification:** `buildSmartMetadata()` and `unifyMemoryEntry()` bridge legacy and current schemas. Always use the smart metadata layer, not raw JSON.
- **Error redaction:** `redactSecrets()` strips API keys, paths, and emails from error messages before logging. Never log unredacted errors.
- **Graceful degradation:** If BM25 index is unavailable, retriever falls back to vector-only. If reranker fails, results return without reranking.

## Configuration

Plugin configuration is defined in `openclaw.plugin.json` with TypeBox schema validation. Key sections: `embedding` (required), `retrieval`, `decay`, `tier`, `smartExtraction`, `scopes`.
