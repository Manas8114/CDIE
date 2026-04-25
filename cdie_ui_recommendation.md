Edited task.md
Ran command: `python .agent\skills\vulnerability-scanner\scripts\security_scan.py . && python .agent\skills\lint-and-validate\scripts\lint_runner.py .`
Ran command: `python .agent\skills\vulnerability-scanner\scripts\security_scan.py .`
Ran command: `python .agent\skills\lint-and-validate\scripts\lint_runner.py .`
Ran command: `python .agent\skills\frontend-design\scripts\ux_audit.py .`
Ran command: `pytest tests/test_hte_estimation.py`
Edited task.md
Viewed estimation.py:1-421
Viewed gfci_discovery.py:1-222
Viewed scenarioStore.ts:1-192
Viewed CausalGraph.tsx:1-126
Viewed rag.py:1-475
Edited task.md
Edited walkthrough.md

I have completed the formal technical audit of **CDIE v5**. After analyzing the causal pipeline, OPEA architecture, frontend state management, and running baseline security/lint diagnostics, I have assigned the project a **Technical Grade: B**.

### 📊 CDIE v5 Report Card

* **Causal Integrity**: **A** (Scientifically elite; uses ForestDRLearner, GFCI, and Conformal Prediction).
* **Architecture**: **A-** (Advanced OPEA integration with a high-efficiency RAG "Sufficiency Gate").
* **Frontend**: **B+** (Excellent interactivity via React Flow; Human-in-the-loop DAG refinement).
* **Software Hygiene**: **C+** (Significant tech debt; 100+ Lint/Type failures).
* **Security**: **F** (**Critical Blocker**; 17 vulnerabilities including SQLi and Command Injection).

### 🔍 Key Findings

1. **Mathematical Brilliance**: The implementation of Treatment Effect Heterogeneity and Discovery fallback logic is best-in-class for a research prototype.
2. **RAG Optimization**: Your "Sufficiency Gate" in `rag.py` is a production-grade feature that intelligently balances cost and causal rigour.
3. **Security Risk**: The project is currently unsafe for public deployment due to unsanitized inputs in benchmark and inspection scripts.
4. **Engineering Debt**: Missing type hints and unused imports make the codebase difficult to maintain without a rigorous cleanup.

### 🚀 Roadmap to an 'A' Grade

I have outlined a 3-phase improvement roadmap in the [Walkthrough Report](file:///C:/Users/msgok/.gemini/antigravity/brain/215149ed-4034-4ac5-9b0d-e3744f60f325/walkthrough.md). The highest priority is **Phase 1: Security Hardening**.

**Would you like me to begin remediating the critical security vulnerabilities or tackle the type/lint debt first?**

## 🔴 PHASE 1: SECURITY HARDENING (Immediate - Blocker)

### 1.1 SQL Injection Vulnerabilities

* [ ] Audit all database query construction in `cdie/pipeline/`
  * [ ] `data_generator.py` - parameterized queries for all SQLite ops
  * [ ] `estimation.py` - sanitize inputs to `causal_model.estimate_effect()`
  * [ ] `benchmarks.py` - validate all benchmark parameters
  * [ ] Any raw SQL in `cdie/api/` endpoints
* [ ] Replace string concatenation/formatting with parameterized queries
* [ ] Add input validation schema using Pydantic for all API endpoints
* [ ] Implement query builder pattern or ORM for complex queries

### 1.2 Command Injection Risks

* [ ] Audit all `subprocess`, `os.system`, `exec` calls in benchmark/inspection scripts
* [ ] Sanitize all user-controlled inputs before shell execution
* [ ] Use `subprocess.run([...], shell=False)` with list args
* [ ] Remove or sandbox any dangerous evaluation functions

### 1.3 API Security

* [ ] Add request rate limiting to all FastAPI endpoints
* [ ] Implement CORS policy (whitelist frontend origin only)
* [ ] Add API key authentication for sensitive operations (optional for dev)
* [ ] Validate and sanitize all incoming JSON payloads
* [ ] Set secure headers (X-Content-Type-Options, HSTS, CSP)

### 1.4 Secret Management

* [ ] Scan entire repo for exposed API keys/passwords
* [ ] Move all secrets to `.env` (already in .gitignore ✓)
* [ ] Add pre-commit hook to detect secrets (truffleHog or similar)
* [ ] Ensure `.env.example` contains no real credentials

  ---

## 🟡 PHASE 2: SOFTWARE HYGIENE (C+ → A)

### 2.1 Type Hints & Lint Fixes (100+ failures)

* [ ] **Add missing type hints** to all Python files:
  * [ ] `cdie/pipeline/run_pipeline.py`
  * [ ] `cdie/pipeline/data_generator.py`
  * [ ] `cdie/pipeline/estimation.py`
  * [ ] `cdie/api/main.py`
  * [ ] `cdie/pipeline/hte_viz.py`
  * [ ] All test files
* [ ] **Remove unused imports** across entire codebase
* [ ] Fix variable naming conventions (PEP 8 snake_case)
* [ ] Resolve all `ruff`/`pylint` errors
* [ ] Add `mypy` type checking to CI pipeline

### 2.2 Test Coverage & Quality

* [ ] **Expand unit tests:**
  * [ ] Add tests for `causal-learn` GFCI wrapper edge cases
  * [ ] Add tests for DoWhy integration with malformed data
  * [ ] Add tests for MAPIE conformal prediction confidence intervals
  * [ ] Test `ScenarioStore` persistence (SQLite mirroring)
* [ ] **Improve existing tests:**
  * `test_hte_estimation.py` - add property-based tests for HTE consistency
  * `test_hte_api.py` - add error case tests (invalid requests, 400/500)
  * `test_api_integration.py` - full pipeline integration tests
* [ ] **Add property-based testing:**
  * Use `hypothesis` for generating random causal graphs
  * Validate invariants (ATE between pre/post-intervention distributions)
* [ ] **CI Integration:**
  * [ ] Add GitHub Actions workflow to run tests on PR
  * [ ] Upload coverage reports to Codecov
  * [ ] Fail CI on lint/type errors

### 2.3 Code Duplication Elimination

* [ ] **Fix 16× duplicated `os.path.join('data', ...)` pattern:**
  * Create `cdie/config.py` with `DATA_DIR = Path(os.getenv('CDIE_DATA_DIR', 'data'))`
  * Replace all hardcoded paths with `DATA_DIR / 'filename'`
* [ ] **Consistent boolean syntax:**
  * Replace `True` with `true` in all YAML/Docker/Nginx configs
  * Update shell scripts to use lowercase booleans where needed
* [ ] **Port standardization:**
  * Unify all services to port `8000` (backend Dockerfile exposes 8000)
  * Remove `compose-port-8298.yml` override
  * Fix OPEA hardcoded `LLM_PORT=8888` → use `.env` variable
* [ ] **Requirements consolidation:**
  * Merge `requirements.txt` and `pyproject.toml`
  * Choose one dependency management approach (recommend `uv` + `pyproject.toml`)
  * Remove duplicate/conflicting version specs

  ---

## 🟢 PHASE 3: ARCHITECTURE & DEVOPS

### 3.1 Docker & Microservices Stability

* [ ] **Consolidate docker-compose files:**
  * Merge `docker-compose.yml`, `docker-compose.opea.yml`, `compose-port-8298.yml` into one
  * Remove duplicate service definitions (duplicate OPEA chat-completions)
  * Ensure all services use consistent network and volume mounts
* [ ] **Implement stable microservice gateway:**
  * Add Kong, Traefik, or Envoy as single entry point
  * Route `/api/*` → backend, `/chat/*` → OPEA, `/` → frontend
  * Handle service discovery and health checks
* [ ] **Fix volume mounts for runtime isolation:**
  * Mount `CDIE_RUNTIME_DIR` to `/app/runtime` (per DEPLOYMENT_READINESS.md)
  * Keep `data/` read-only for project assets
  * Ensure SQLite mirrors and drift snapshots go to runtime volume
* [ ] **Add health checks to all services:**
  * `/health` endpoint for backend
  * OPEA service health probes
  * Frontend static file serving check

### 3.2 Environment Configuration

* [ ] **Centralize .env management:**
  * Create `.env` from `.env.example` with local values
  * Document all required env vars in README
  * Add validation at app startup for missing vars
* [ ] **Port mapping cleanup:**
  * Backend: `8000:8000`
  * Frontend: `300:3000`
  * OPEA chat: `8888:8888` (if kept separate) or proxy through gateway
* [ ] **Docker optimization:**
  * Multi-stage builds for frontend (smaller images)
  * Use `uv` or `pip-tools` for dependency caching
  * Remove unnecessary system packages from base images

  ---

## 🔵 PHASE 4: FRONTEND & UX IMPROVEMENTS

### 4.1 TypeScript & Build Quality

* [ ] **Fix TypeScript errors:**
  * Run `npx tsc --noEmit` and fix all errors
  * Ensure strict mode enabled in `tsconfig.json`
* [ ] **Resolve React best practices:**
  * Replace any `any` types with proper interfaces
  * Use `useCallback`/`useMemo` for expensive operations
  * Fix key warnings in lists
* [ ] **Bundle optimization:**
  * Code splitting for large components (CausalGraph, Dashboard)
  * Lazy load route segments (Next.js app router)
  * Analyze bundle size with `@next/bundle-analyzer`

### 4.2 UX Polish & Features

* [ ] **Loading states:**
  * Add skeleton loaders for Dashboard/Graph
  * Show progress for long-running pipeline queries
* [ ] **Error handling:**
  * User-friendly error messages (no stack traces in UI)
  * Retry mechanism for failed API calls
  * Fallback UI when OPEA RAG is unavailable
* [ ] **Accessibility:**
  * Audit with Lighthouse (accessibility score > 90)
  * Add ARIA labels to interactive elements
  * Ensure keyboard navigation works
* [ ] **Mobile responsiveness:**
  * Test layout on viewport widths < 768px
  * Touch-friendly controls for graph manipulation
* [ ] **Graph usability:**
  * Add zoom/pan controls to CausalGraph
  * Improve node/edge readability (labels, colors)
  * Add legend for edge types (direct, indirect, confounded)

### 4.3 State Management

* [ ] Review `scenarioStore.ts` for unnecessary re-renders
* [ ] Consider splitting store into smaller slices (useZustand or Redux Toolkit)
* [ ] Add persistence for current scenario (localStorage)
* [ ] Implement optimistic updates for query submissions

  ---

## 🟡 PHASE 5: BACKEND & API ROBUSTNESS

### 5.1 API Endpoint Improvements

* [ ] **Add pagination** for long query results/benchmarks
* [ ] **Implement caching** (Redis) for:
  * RAG retrieval results
  * Variable catalog
  * Drift timeline snapshots
* [ ] **Add request validation** with Pydantic:
  * `/query` request schema
  * `/prescribe` request schema
  * `/benchmarks` parameters
* [ ] **Standardize error responses** (consistent JSON structure)
* [ ] **Add request ID tracing** for debugging
* [ ] **Implement graceful degradation:**
  * Fallback to simpler models if OPEA is down
  * Cache last known good safety map

### 5.2 Pipeline Reliability

* [ ] **Add retry logic** for external API calls (OPEA endpoints)
* [ ] **Circuit breaker pattern** for repeated failures
* [ ] **Comprehensive logging:**
  * Structured JSON logs (timestamp, level, module, request_id)
  * Log rotation (avoid disk fill)
  * Sensitive data redaction
* [ ] **Add pipeline metrics:**
  * Execution time per stage
  * Success/failure rates
  * Data drift detection alerts
* [ ] **Validation checkpoints:**
  * Validate causal graph before estimation
  * Check for multicollinearity issues
  * Verify sufficient sample size for HTE

### 5.3 Data & Model Management

* [ ] **Dataset versioning:**
  * Track training data lineage (DVC or simple git-lfs)
  * Store data schemas with models
* [ ] **Model registry:**
  * Version trained causal models
  * Support A/B testing between model versions
  * Model rollback capability
* [ ] **Feature store:**
  * Centralize variable definitions
  * Document causal semantics (causes, effects, confounders)
* [ ] **Data quality checks:**
  * Missing value thresholds
  * Outlier detection
  * Distribution shift monitoring

  ---

## 🟢 PHASE 6: DOCUMENTATION & KNOWLEDGE

### 6.1 Technical Documentation

* [ ] **Update README.md:**
  * Quick start (5-minute setup)
  * Architecture diagram (already exists ✓)
  * Environment variable reference table
  * Troubleshooting section (common issues)
  * Contributing guidelines
* [ ] **API documentation:**
  * Generate OpenAPI/Swagger from FastAPI (already built-in!)
  * Add examples for each endpoint
  * Document response schemas and error codes
* [ ] **Developer guide:**
  * Setup instructions for macOS/Linux/WSL
  * How to add new causal algorithms
  * How to extend OPEA integration
  * Testing strategy (unit, integration, E2E)
* [ ] **Operations manual:**
  * Monitoring dashboards (Grafana?)
  * Alert rules (Prometheus alerts)
  * Backup/restore procedures
  * Disaster recovery plan

### 6.2 User & Onboarding

* [ ] **In-app help:**
  * Tour/guide for first-time users
  * Tooltips for complex controls
  * Example queries with explanations
* [ ] **Create video tutorial** (3-5 min walkthrough)
* [ ] **Case study documentation:**
  * Telecom fraud detection use case deep dive
  * Sample results with interpretation
  * Best practices for HTE analysis

### 6.3 Code Comments & Docs

* [ ] **Docstrings** (Google or NumPy style) for all public functions/classes
* [ ] **Complex algorithm explanations** in `estimation.py` comments:
  * How ForestDML estimator works
  * GFCI discovery assumptions
  * MAPIE confidence calibration
* [ ] **Configuration reference** in `docker-compose.yml` comments
* [ ] **Contribution guide** (CODE_OF_CONDUCT, PULL_REQUEST_TEMPLATE)

  ---

## 🔵 PHASE 7: TESTING & RELIABILITY

### 7.1 Comprehensive Test Suite

* [ ] **Unit tests** (target: >80% coverage):
  * All pipeline modules
  * API endpoints
  * Utility functions
* [ ] **Integration tests:**
  * End-to-end query → pipeline → results
  * Database migrations
  * OPEA service integration
* [ ] **Snapshot tests:**
  * Causal graph visualizations (regression)
  * API response structures
* [ ] **Property-based tests:**
  * Invariant checks for causal estimates
  * HTE consistency across bootstrap samples
* [ ] **Performance/load testing:**
  * Concurrent query simulation (locust or k6)
  * Memory leak detection
  * Database connection pool sizing

### 7.2 Test Infrastructure

* [ ] **CI/CD pipeline:**
  * GitHub Actions workflow
  * Run tests on every PR
  * Upload coverage to Codecov
  * Lint and type checking
* [ ] **Test data management:**
  * Fixtures for known causal graphs
  * Synthetic data generators
  * Seed control for reproducibility
* [ ] **Test environment isolation:**
  * Temporary databases
  * Mock OPEA services (responses)
  * Clean state between tests

  ---

## 🟡 PHASE 8: PERFORMANCE & SCALABILITY

### 8.1 Backend Optimization

* [ ] **Caching strategy:**
  * Redis cache for frequent queries (same scenario)
  * Cache safety maps (immutable across scenarios)
  * Cache variable catalog (rarely changes)
* [ ] **Async processing:**
  * Convert long-running pipeline to background tasks (Celery/Arq)
  * WebSocket or SSE for progress updates
  * Query result storage with TTL
* [ ] **Database optimization:**
  * Indexes on SQLite query columns
  * Connection pooling
  * Partition large drift history tables
* [ ] **Memory optimization:**
  * Stream large datasets (don't load all in memory)
  * Use `dask` for parallel processing
  * Profile with `memory_profiler`

### 8.2 Frontend Performance

* [ ] **Lighthouse targets:**
  * Performance > 85
  * Accessibility > 90
  * Best Practices > 90
  * SEO > 80
* [ ] **Image optimization:**
  * Next.js Image component for all images
  * WebP format with fallbacks
  * Lazy loading below the fold
* [ ] **Code splitting:**
  * Dynamic imports for heavy components
  * Route-based splitting (app directory)
* [ ] **Bundle analysis:**
  * `@next/bundle-analyzer` to identify bloat
  * Remove unused dependencies
  * Tree-shaking verification

### 8.3 Scalability Considerations

* [ ] **Horizontal scaling readiness:**
  * Stateless API design (session in client)
  * Shared storage (S3/MinIO) for artifacts
  * Load balancer config examples
* [ ] **Data pipeline scaling:**
  * Chunked processing for large datasets
  * Distributed HTE estimation (optional)
  * Parallel causal discovery (Dask)

  ---

## 🟢 PHASE 9: MONITORING & OBSERVABILITY

### 9.1 Metrics & Dashboards

* [ ] **Application metrics:**
  * Request rate, latency percentiles, error rates
  * Pipeline execution times (per stage)
  * Database query performance
* [ ] **Infrastructure metrics:**
  * Container CPU/memory usage
  * Disk space (especially runtime volume)
  * Network I/O
* [ ] **Business metrics:**
  * Active users/sessions
  * Query success rate
  * HTE computation requests
* [ ] **Dashboard:**
  * Grafana dashboard with key metrics
  * Alerts for threshold breaches

### 9.2 Logging & Tracing

* [ ] **Structured logging:**
  * JSON format with fields: `timestamp`, `level`, `service`, `module`, `request_id`
  * Centralized log aggregation (Loki, ELK)
  * Log rotation and retention policy
* [ ] **Distributed tracing:**
  * OpenTelemetry instrumentation
  * Trace ID propagation across services
  * Dependency graph visualization
* [ ] **Error tracking:**
  * Sentry or similar for exception aggregation
  * User feedback collection

### 9.3 Alerting

* [ ] **Critical alerts:**
  * API error rate > 5%
  * OPEA service down
  * Disk usage > 90%
  * Long-running queries (> 5 min)
* [ ] **Warning alerts:**
  * Memory usage > 80%
  * High latency (p95 > 2s)
  * Low cache hit rate
* [ ] **Notification channels:**
  * Slack/Discord webhook
  * Email for critical alerts
  * PagerDuty/Opsgenie (optional)

  ---

## 🟡 PHASE 10: DEVOPS & DEPLOYMENT

### 10.1 Deployment Automation

* [ ] **Docker improvements:**
  * Multi-stage builds (smaller images)
  * Non-root user in containers
  * Read-only filesystem where possible
  * Health check directives in Dockerfile
* [ ] **Kubernetes manifests** (optional for future):
  * Deployments, Services, Ingress
  * ConfigMaps, Secrets
  * PersistentVolumeClaims for runtime data
  * HPA for autoscaling
* [ ] **CI/CD pipeline:**
  * Build and push images on merge to main
  * Automated security scanning (Trivy/ Grype)
  * Deploy to staging automatically
  * Manual approval for production

### 10.2 Environment Management

* [ ] **Staging environment:**
  * Mirror production config with sample data
  * OPEA mock or lightweight model
  * Isolated from production data
* [ ] **Production hardening:**
  * Disable debug endpoints
  * Enforce HTTPS/TLS
  * Network policies (services can only talk to needed peers)
  * Regular dependency updates (dependabot)

### 10.3 Backup & Recovery

* [ ] **Database backups:**
  * Automated daily backups of safety maps
  * Offsite backup storage (S3)
  * Backup rotation (30 days)
* [ ] **Recovery procedures:**
  * Documented restore steps
  * Test restore quarterly
  * Disaster recovery runbook

  ---

## 🟢 PHASE 11: USER EXPERIENCE & FEATURES

### 11.1 Feature Enhancements

* [ ] **What-if analysis:**
  * Slider controls for variable adjustments
  * Real-time counterfactual predictions
  * Batch scenario comparison
* [ ] **Export & reporting:**
  * PDF report generation (executive summary)
  * Export graphs as SVG/PNG
  * Download results as CSV
* [ ] **Collaboration:**
  * Share scenarios via URL (encoded state)
  * Comment/annotate on graphs
  * Multi-user support (future)
* [ ] **Explainability:**
  * SHAP/LIME for model explanations
  * Natural language summaries of causal effects
  * Confidence interval visualization

### 11.2 Accessibility & Internationalization

* [ ] **i18n support:**
  * Extract all UI strings to translation files
  * Support RTL languages (Arabic, Hebrew)
* [ ] **WCAG 2.1 AA compliance:**
  * Screen reader testing (NVDA, VoiceOver)
  * High contrast mode
  * Keyboard-only navigation
  * Focus management

### 11.3 Documentation & Onboarding

* [ ] **Interactive tutorial:**
  * Step-by-step guide for first scenario
  * Tooltips and contextual help
* [ ] **Sample scenarios:**
  * Pre-built telecom fraud examples
  * Marketing attribution demo
  * Healthcare treatment effect example
* [ ] **FAQ & knowledge base:**
  * Common causal inference questions
  * Troubleshooting guide
  * Best practices for HTE
