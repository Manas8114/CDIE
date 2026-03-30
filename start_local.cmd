@echo off
echo ============================================
echo   CDIE v5 - Local Startup (No Docker)
echo ============================================
echo.

:: Set Intel CPU Optimization
set OMP_NUM_THREADS=12
set DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
set KMP_AFFINITY=granularity=fine,compact,1,0

echo [1/5] Intel optimizations set (OMP_NUM_THREADS=%OMP_NUM_THREADS%)
echo.

:: Install dependencies
echo [2/5] Installing dependencies...
call pip install -r requirements.txt --quiet
cd frontend && call npm install --silent && cd ..
echo       Done.
echo.

:: Run pipeline first to generate safety_map.db
echo [3/5] Running causal discovery pipeline...
python -m cdie.pipeline.run_pipeline
echo       Pipeline complete.
echo.

:: Start FastAPI backend
echo [4/5] Starting FastAPI API on http://localhost:8000 ...
start "CDIE-API" cmd /k "cd /d %~dp0 && set OMP_NUM_THREADS=12 && set DNNL_MAX_CPU_ISA=AVX512_CORE_AMX && python -m uvicorn cdie.api.main:app --host 0.0.0.0 --port 8000"

:: Start Next.js Modern UI
echo [5/5] Starting Next.js Dashboard on http://localhost:3000 ...
start "CDIE-NextJS-UI" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo ============================================
echo   All services started in new windows!
echo.
echo   API:           http://localhost:8000       (Swagger: /docs)
echo   Next.js UI:    http://localhost:3000       [PRIMARY]
echo.
echo   API Endpoints:
echo     Drift:       /api/drift/timeline
echo     Backtest:    /api/backtest
echo     Federation:  /api/federation/export
echo     Knowledge:   /api/knowledge/priors
echo ============================================
echo.
echo   Close this window when ready.
pause
