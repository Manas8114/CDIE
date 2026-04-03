# Deployment Readiness

This project now supports a safer split between canonical project data and runtime state.

## Recommended runtime layout

- Keep `data/` for durable exports such as `safety_map.json`
- Keep `CDIE_RUNTIME_DIR` on a local writable disk for:
  - drift snapshot history
  - runtime SQLite mirrors
  - temporary pipeline state

Avoid OneDrive-synced, network-mounted, or antivirus-heavy paths for `CDIE_RUNTIME_DIR`.

## Required environment variables

- `NEXT_PUBLIC_API_URL` - frontend API base, for example `http://localhost:8000`
- `CDIE_RUNTIME_DIR` - local writable runtime directory
- `CDIE_DATA_DIR` - optional override for canonical project data

Optional GenAI services:

- `OPEA_LLM_ENDPOINT`
- `OPEA_EMBEDDING_ENDPOINT`
- `OPEA_RERANKING_ENDPOINT`
- `OPENAI_API_KEY`

## Pre-deploy checklist

1. Run the offline pipeline once:
   - `python -m cdie.pipeline.run_pipeline`
2. Confirm API health:
   - `GET /health`
3. Confirm variable catalog:
   - `GET /variables`
4. Confirm drift history:
   - `GET /api/drift/timeline`
5. Confirm frontend wiring:
   - `npm run lint`
   - `npx tsc --noEmit`
6. Confirm trust metadata on a real query:
   - `POST /query`
   - inspect `match_type`, `evidence_tier`, `trust_message`

## Docker notes

The compose file mounts a shared runtime volume at `/app/runtime` for the API and pipeline.
That keeps drift history and runtime SQLite mirrors out of the bind-mounted project data path.
