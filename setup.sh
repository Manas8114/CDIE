#!/bin/bash
# CDIE v5 — One-Click Setup Script
# Telecom SIM Box Fraud Detection with Causal AI + OPEA GenAIComps
# For ITU AI4Good OPEA Innovation Challenge

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CDIE v5 — Causal Decision Intelligence Engine              ║"
echo "║  Telecom SIM Box Fraud Detection                            ║"
echo "║  OPEA GenAIComps + Intel AMX/AVX-512                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check for .env file
if [ ! -f .env ]; then
    echo "[SETUP] No .env file found. Creating from template..."
    cp .env.example .env
    echo "[SETUP] ⚠️  Please edit .env and set your HF_TOKEN before running Docker."
    echo "[SETUP]    Example: HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx"
fi

# Step 2: Check for Docker
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "[ERROR] Docker Compose v2 is not installed. Please update Docker."
    exit 1
fi

echo "[SETUP] ✅ Docker and Docker Compose v2 detected."

# Step 3: Start OPEA services
echo ""
echo "[STEP 1/5] Starting OPEA TGI + TextGen + TEI Embed/Rerank services..."
docker compose up -d tgi-service opea-llm-textgen tei-embedding tei-reranking
echo "[SETUP] ⏳ Waiting 30s for models to load..."
sleep 30

# Step 4: Run causal discovery pipeline
echo "[STEP 2/5] Running offline causal discovery pipeline..."
docker compose run --rm pipeline

# Step 5: Start API server
echo "[STEP 3/5] Starting CDIE API server..."
docker compose up -d api

# Step 6: Start Streamlit UI (legacy)
echo "[STEP 4/5] Starting Streamlit UI (legacy)..."
docker compose up -d ui

# Step 7: Start Next.js UI (primary)
echo "[STEP 5/5] Starting Next.js Dashboard (production build)..."
docker compose up -d nextjs-ui

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ CDIE v5 is running!                                     ║"
echo "║                                                              ║"
echo "║  🌐 Next.js Dashboard: http://localhost:3000  [PRIMARY]      ║"
echo "║  🔌 FastAPI Docs:      http://localhost:8000/docs            ║"
echo "║  📊 Streamlit UI:      http://localhost:8501  [LEGACY]       ║"
echo "║  🧠 OPEA TextGen:      http://localhost:9000                 ║"
echo "║  📈 Latency Test:      http://localhost:8000/benchmark       ║"
echo "║  ℹ️  System Info:       http://localhost:8000/info            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "API Endpoints:"
echo "  Drift:       GET  /api/drift/timeline"
echo "  Backtest:    POST /api/backtest"
echo "  Federation:  GET  /api/federation/export"
echo "  Knowledge:   GET  /api/knowledge/priors"
