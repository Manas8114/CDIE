@echo off
echo ============================================
echo   CDIE v5 - Docker Compose Startup
echo   Causal Decision Intelligence Engine
echo ============================================
echo.

:: Check for .env file
if not exist .env (
    echo [WARNING] .env file not found. Copying from .env.example...
    copy .env.example .env
    echo Please edit .env and set your HF_TOKEN for OPEA model downloads.
    echo.
)

:: Set Intel CPU Optimization Flags
set OMP_NUM_THREADS=12
set DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
set KMP_AFFINITY=granularity=fine,compact,1,0

echo [1/3] Intel optimizations configured (OMP_NUM_THREADS=12)
echo.

echo [2/3] Building and starting all Docker containers...
echo       This includes: Pipeline, API, Next.js UI, Streamlit UI, OPEA services
docker compose up -d --build

echo.
echo [3/3] All containers deployed.
echo.
echo ============================================
echo   CDIE v5 Stack is starting up!
echo.
echo   Next.js Dashboard:         http://localhost:3000   [PRIMARY]
echo   API Swagger Docs:          http://localhost:8000/docs
echo   Streamlit UI (Legacy):     http://localhost:8501
echo   OPEA TextGen API:          http://localhost:9000
echo   OPEA Embeddings API:       http://localhost:6006
echo   OPEA Reranking API:        http://localhost:8808
echo ============================================
echo.
echo   Features:
echo     - Causal Drift Dashboard    /api/drift/*
echo     - Backtesting Engine        /api/backtest
echo     - Federated Learning        /api/federation/*
echo     - Knowledge Brain           /api/knowledge/*
echo.
echo Note: The Pipeline container runs once to generate safety_map.db.
echo       Follow logs: docker compose logs -f
echo.
pause
