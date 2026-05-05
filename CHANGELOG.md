# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- GitHub Actions CI workflow for Ruff, MyPy, and Pytest.
- Health-check endpoints and hardware benchmark endpoints (`/benchmark/hardware`).
- Support for Docker Secrets for secure credential handling (`HF_TOKEN_FILE`).
- Advanced documentation for causal method contribution and hardware benchmarking.

### Changed
- Shifted from single large models to efficient Intel-optimized OPEA microservices.
- Transitioned default UI to Next.js (previously Streamlit).
- Improved Safety Map versioning (added `version` and `sha256_hash`).

### Security
- Excluded tokens from `.env` tracking and migrated to Docker Secrets configuration.
