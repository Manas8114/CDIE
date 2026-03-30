# 📖 Introduction: Causal Intelligence for Telecom

CDIE v4 (Causal Decision Intelligence Engine) is a specialized platform designed to transition the telecommunications industry from **correlational** analytics to **causal** decision support.

---

## 🛑 The Problem: SIM Box Fraud

SIM box fraud (Interconnect Fraud) costs mobile operators an estimated **$3.8 billion annually**. 

Traditional machine learning models (XGBoost, Random Forest + SHAP) identify what *correlates* with fraud. For example, they might say "High Transaction Volume" is the top feature for fraud. 
However, **correlation is not causation**. 

### The Risk of Correlational AI
- **Simpson's Paradox**: An intervention could work for individual segments but hurt the overall network.
- **Confounding Variables**: Features that influence both the "cause" and "effect," leading to biased recommendations.
- **Incorrect Policy Changes**: Operators might tighten policies that don't actually reduce fraud, instead hurting legitimate customer experience (NPS).

---

## ✅ The Solution: Causal AI with CDIE v4

CDIE v4 uses **Structural Causal Models (SCM)** to discover the true underlying relationships between network KPIs, policies, and fraud.

### Key Advantages
1. **Provably Correct Interventions**: We estimate the **Average Treatment Effect (ATE)** using doubly-robust methods (LinearDML), ensuring impact estimates remain valid even if data models are slightly misspecified.
2. **Validated Claims**: Every causal link is rigorously tested through **DoWhy Refutation** suites (Placebo treatments, random common causes, etc.).
3. **Human-in-the-Loop (HITL)**: Domain experts can explicitly "correct" the AI's discovered graph, injecting expert priors for refined future discovery.
4. **OPEA-Integrated Intelligence**: Complex causal statistics are translated into executive-level **Causal Intelligence Reports** using Intel's Open Platform for Enterprise AI.

---

## 🎯 Use Case Alignment
CDIE directly aligns with the **Telecommunications Networks** vertical of the ITU AI4Good OPEA Innovation Challenge. It provides operators with **causal evidence** to justify policy changes at the CNO level, moving beyond black-box ML alerts.
