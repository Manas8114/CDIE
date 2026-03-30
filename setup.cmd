@echo off
echo ==============================================================
echo   CDIE v5 -- Causal Decision Intelligence Engine
echo   Telecom SIM Box Fraud Detection
echo   OPEA GenAIComps + Intel AMX/AVX-512
echo ==============================================================
echo.

:: Step 0: Environment file
IF NOT EXIST .env (
    echo [SETUP] No .env file found. Creating from template...
    copy .env.example .env
    echo [SETUP] WARNING: Please edit .env and set your HF_TOKEN before running Docker.
    echo [SETUP]    Example: HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
)

:: Step 1: Check Docker
docker --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed. Please install Docker first.
    exit /b 1
)

docker compose version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Compose v2 is not installed. Please update Docker Desktop.
    exit /b 1
)

echo [SETUP] OK: Docker and Docker Compose v2 detected.
echo.

:: Step 2: Start OPEA services
echo [STEP 1/5] Starting OPEA TGI + TextGen + TEI Embed/Rerank services...
docker compose up -d tgi-service opea-llm-textgen tei-embedding tei-reranking
echo [SETUP] Waiting 30s for models to load...
timeout /t 30 /nobreak >nul

:: Step 3: Run causal discovery pipeline
echo [STEP 2/5] Running offline causal discovery pipeline...
docker compose up --build --force-recreate --no-deps pipeline
docker compose run --rm pipeline

:: Step 4: Start API server
echo [STEP 3/5] Starting CDIE API server...
docker compose up -d --build api

:: Step 5: Start Streamlit UI
echo [STEP 4/5] Starting Streamlit UI (legacy)...
docker compose up -d --build ui

:: Step 6: Start Next.js UI
echo [STEP 5/5] Starting Next.js Dashboard (production build)...
docker compose up -d --build nextjs-ui

echo.
echo ==============================================================
echo   CDIE v5 is running!
echo.
echo   Next.js Dashboard:  http://localhost:3000   [PRIMARY]
echo   FastAPI Docs:        http://localhost:8000/docs
echo   Streamlit UI:        http://localhost:8501   [LEGACY]
echo   OPEA TextGen:        http://localhost:9000
echo   Latency Test:        http://localhost:8000/benchmark/latency
echo   System Info:         http://localhost:8000/info
echo ==============================================================
echo.
pause
