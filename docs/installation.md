# 🚀 Installation & Setup: Get CDIE v4 Running

CDIE v4 (Causal Decision Intelligence Engine) is designed for single-node deployment on **Intel Xeon** server hardware using Docker Compose. This document provides a step-by-step setup guide.

---

## 🛠️ Prerequisites

### Hardware Requirements:
- **CPU**: Intel Xeon with AVX-512 and AMX instructions (Sapphire Rapids or newer recommended).
- **RAM**: Minimum 64GB (The full OPEA stack with TGI requires ~57GB at peak).
- **Storage**: 100GB of available disk space (Docker images + Model weights).

### Software Requirements:
- **Operating System**: Linux (Ubuntu 22.04+ recommended) or Windows with Docker Desktop (WSL2).
- **Docker**: Docker Engine 24.0+ and Docker Compose v2.20+.
- **Hugging Face**: An `HF_TOKEN` for model downloads.

---

## 1. 📂 Clone & Configure

```bash
git clone https://github.com/Manas8114/CDIE-v4.git
cd CDIE-v4
```

### Environment Variables:
Copy the example environment file and add your Hugging Face token:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=your_token_here
```

---

## 2. ⚡ One-Click Deployment

### Linux/macOS:
Run the `setup.sh` script to build and start all 7 containers in the correct sequence (TGI → OPEA Services → Pipeline → API → UI).

```bash
chmod +x setup.sh
./setup.sh
```

### Windows:
Run the `setup.cmd` script from a PowerShell or Command Prompt.

```bash
.\setup.cmd
```

---

## 3. 🏗️ Manual Docker Compose (Optional)

If you prefer to start services manually:

```bash
# Start the OPEA microservice stack
docker-compose up -d tgi-service embedding-service reranking-service textgen-service

# Run the Offline Pipeline once to generate the Safety Map
docker-compose up pipeline

# Start the API and UI
docker-compose up -d api ui
```

---

## 4. 🎛️ Verifying Hardware Optimizations

Once the containers are running, you can verify that the Intel-optimized DNNL flags are active:

```bash
docker exec -it cdie-api curl http://localhost:8000/benchmark/hardware
```

**Expected JSON Response:**
```json
{
  "optimization_active": true,
  "hardware_detection": {
    "avx512_available": true,
    "amx_available": true
  }
}
```

---

## 🛑 Troubleshooting

### 1. Out of Memory (OOM):
If the TGI container (32GB) fails to start, ensure your Docker Desktop or Server has at least 64GB of allocated RAM. You can disable the LLM and Reranking services in `docker-compose.yml` to run a "lite" version (4GB RAM) with rule-based templates.

### 2. Model Download Failure:
Check your `HF_TOKEN` in the `.env` file and ensure you have internet access. Some OPEA images are large (~5GB-10GB).
