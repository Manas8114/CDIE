# CDIE v4 Codebase Audit: Identified Issues & Remediations

This document tracks functional bugs, logic errors, and technical debt identified during the system audit.

## 🔴 Critical (Blockers)

### 1. `ImportError` in `cdie/api/main.py`

- **Issue**: The API fails to start because it attempts to import `PrescribeRequest` from `cdie.api.models`, but the class is missing from that file. It is currently defined locally in `main.py`.
- **Impact**: API crash on startup.
- **Status**: [x] Fixed
- **Remediated**: Moved `PrescribeRequest` and `PrescribeResponse` to `cdie/api/models.py` and updated imports in `main.py`.

### 2. Startup Crash / Database Format Mismatch

- **Issue**: `main.py` (line 67) attempts to load `safety_map.json`, but `SafetyMapLookup` (used globally in `main.py`) internally expects a `.db` (SQLite) file. If the JSON exists but the DB doesn't, the API will fail to serve scenarios.
- **Impact**: Silent failure or crash during query processing if the DB is missing.
- **Status**: [x] Fixed
- **Remediated**: Standardized on the SQLite format for the Safety Map. Updated `main.py` to check for `safety_map.db` and fallback to `.json` with clear logs.

---

## 🟠 Major (Logic & Stability)

### 3. Causal Impact Calculation (0.0 Returns)

- **Issue**: In `cdie/pipeline/estimation.py` (line 41), the discrete treatment detection `len(np.unique) < 10` was too aggressive. Variables with 10-15 unique values (like Likert scales or integer scores) were being misclassified, leading to model non-convergence or zeroed-out impacts.
- **Impact**: Incorrect 0.0 causal reporting in the dashboard (CATE section).
- **Status**: [x] Fixed
- **Remediated**: Implemented a more robust discrete/continuous classifier checking both cardinality and data types. Added better error logging and fixed variable scoping bugs in the estimation pipeline.

### 4. Heuristic Fallback Bias

- **Issue**: In `estimation.py` (line 67, 76), the `OLS_fallback` used a hardcoded `0.15` multiplier for standard errors. This was a heuristic and provided non-scientific confidence intervals.
- **Impact**: Misleading uncertainty quantification for users when EconML fails.
- **Status**: [x] Fixed
- **Remediated**: Replaced the heuristic with a scientifically valid Standard Error calculation using `statsmodels.api.OLS`. Added `sm.add_constant` to ensure intercept is handled correctly in the fallback model.

### 5. Magnitude Scaling Over-Simplification

- **Issue**: In `cdie/pipeline/safety_map.py` (line 78), the effect is scaled linearly: `effect = ate * intervention_amount`.
- **Impact**: Fails to capture non-linear causal dynamics (diminishing returns, tipping points).
- **Status**: [ ] Noted (Architectural Limitation)
- **Remediation**: Future enhancement: Use the SCM to re-simulate intervention outcomes at different magnitudes instead of linear scaling.

---

## 🟡 Minor (Technical Debt & Cleanup)

### 6. Linting & Code Quality (1300+ Errors)

- **Issue**: `ruff` identified massive technical debt (unused imports, bare `except` blocks, f-string errors).
- **Impact**: Increased maintenance cost and potential for hidden bugs (especially in error handling).
- **Status**: [x] Fixed
- **Remediation**: Ran `ruff format` and `ruff check --fix` and manually resolved remaining bare `except` clauses and module imports in `app.py` and `safety_map.py`.

### 7. Inconsistent Endpoint Models

- **Issue**: Several endpoints in `main.py` (e.g., `/graph`, `/benchmark`) do not explicitly define a `response_model` despite the models existing in `models.py`.
- **Impact**: Weaker API contract and lack of automated validation/documentation in Swagger.
- **Status**: [x] Fixed
- **Remediated**: Added `response_model` to all endpoints in `main.py`.
