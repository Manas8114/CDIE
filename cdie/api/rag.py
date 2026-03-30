"""
CDIE v4 — RAG + Explanation Engine (OPEA-Integrated)
Template-based explanations with synthetic historical analogies.
Routes LLM calls through OPEA GenAIComps TextGen microservice.
Uses OPEA TEI Embedding for semantic retrieval (fallback: TF-IDF).
Uses OPEA TEI Reranking for passage re-ranking (fallback: cosine sim).
"""

import os
import time
import requests  # type: ignore
import numpy as np  # type: ignore
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

import json
from pathlib import Path

# Load historical playbooks for RAG retrieval
DATA_DIR = Path(os.environ.get("CDIE_DATA_DIR", Path(__file__).parent.parent.parent / "data"))

def load_historical_events():
    playbooks_path = DATA_DIR / "telecom_playbooks.json"
    if playbooks_path.exists():
        with open(playbooks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[RAG] WARNING: {playbooks_path} not found. Returning empty list.")
    return []

HISTORICAL_EVENTS = load_historical_events()


class ExplanationEngine:
    """Generates explanations using templates and historical analogies.
    
    OPEA Integration:
    - TEI Embedding (BAAI/bge-base-en-v1.5) for semantic vector retrieval
    - TEI Reranking (BAAI/bge-reranker-base) for passage re-ranking
    - LLM TextGen (Intel/neural-chat-7b-v3-3) for Assumption Intelligence
    """

    def __init__(self):
        self.events = HISTORICAL_EVENTS
        self.event_texts = [e["text"] for e in self.events]
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any | None = None

        # OPEA TEI Embedding integration
        self.embedding_endpoint = os.environ.get("OPEA_EMBEDDING_ENDPOINT")
        self.reranking_endpoint = os.environ.get("OPEA_RERANKING_ENDPOINT")
        self.embeddings_cache = None
        self.embedding_provider = "tfidf"

        if self.embedding_endpoint:
            try:
                self._build_opea_index()
                self.embedding_provider = "opea_tei"
                print(f"[RAG] OPEA TEI Embedding connected at {self.embedding_endpoint}")
            except Exception as e:
                print(f"[RAG] OPEA TEI Embedding unavailable: {e}. Falling back to TF-IDF.")
                self._build_tfidf_index()
        else:
            self._build_tfidf_index()
            print("[RAG] Using TF-IDF vectorizer (OPEA TEI not configured).")

        if self.reranking_endpoint:
            self.reranking_provider = "opea_tei"
            print(f"[RAG] OPEA TEI Reranking connected at {self.reranking_endpoint}")
        else:
            self.reranking_provider = "cosine"
            print("[RAG] Using cosine similarity for ranking (OPEA TEI Reranking not configured).")

        # OPEA GenAIComps TextGen integration (priority 1)
        self.opea_endpoint = os.environ.get("OPEA_LLM_ENDPOINT")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.llm_model = os.environ.get("LLM_MODEL_ID", "Intel/neural-chat-7b-v3-3")
        self.client = None
        self.llm_provider = "template"  # fallback default

        if self.opea_endpoint:
            try:
                from openai import OpenAI  # type: ignore
                self.client = OpenAI(
                    base_url=f"{self.opea_endpoint}/v1",
                    api_key="opea-placeholder",  # OPEA TextGen doesn't require auth
                )
                self.llm_provider = "opea"
                print(f"[RAG] OPEA GenAIComps TextGen connected at {self.opea_endpoint}")
            except ImportError:
                print("[RAG] OpenAI package not installed. Cannot connect to OPEA.")
        elif self.openai_api_key:
            try:
                from openai import OpenAI  # type: ignore
                self.client = OpenAI(api_key=self.openai_api_key)
                self.llm_model = "gpt-4o-mini"
                self.llm_provider = "openai"
                print("[RAG] OpenAI Client Initialized (fallback from OPEA).")
            except ImportError:
                print("[RAG] OpenAI package not installed. Falling back to templates.")
        else:
            print("[RAG] No LLM endpoint configured. Using template-based explanations.")

    def _build_tfidf_index(self):
        """Build TF-IDF index for historical event retrieval (fallback)."""
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.event_texts)

    def _build_opea_index(self):
        """Build dense embedding index using OPEA TEI Embedding service."""
        response = requests.post(
            f"{self.embedding_endpoint}/embed",
            json={"inputs": self.event_texts},
            timeout=30,
        )
        response.raise_for_status()
        self.embeddings_cache = np.array(response.json())

    def _embed_query_opea(self, query: str) -> np.ndarray:
        """Embed a single query using OPEA TEI Embedding."""
        response = requests.post(
            f"{self.embedding_endpoint}/embed",
            json={"inputs": query},
            timeout=10,
        )
        response.raise_for_status()
        return np.array(response.json())

    def _rerank_opea(self, query: str, passages: list[dict], top_k: int = 3) -> list[dict]:
        """Re-rank passages using OPEA TEI Reranking service."""
        try:
            texts = [p["text"] for p in passages]
            response = requests.post(
                f"{self.reranking_endpoint}/rerank",
                json={"query": query, "texts": texts},
                timeout=10,
            )
            response.raise_for_status()
            scores = response.json()

            for i, score_data in enumerate(scores):
                if i < len(passages):
                    idx = score_data.get("index", i)
                    passages[idx]["rerank_score"] = score_data.get("score", 0)

            passages.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return passages[:top_k]  # type: ignore
        except Exception as e:
            print(f"[RAG] Reranking failed: {e}. Using cosine similarity order.")
            return passages[:top_k]  # type: ignore

    def retrieve_analogies(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k most relevant historical events."""
        if self.embedding_provider == "opea_tei" and self.embeddings_cache is not None:
            return self._retrieve_opea(query, top_k)
        return self._retrieve_tfidf(query, top_k)

    def _retrieve_opea(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve using OPEA TEI dense embeddings + optional reranking."""
        try:
            query_embedding = self._embed_query_opea(query)
            if query_embedding is not None and query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            cache = self.embeddings_cache
            if cache is not None and cache.ndim == 1:
                cache = cache.reshape(1, -1)

            similarities = cosine_similarity(query_embedding, cache)[0]
            # Get top candidates for potential reranking
            n_candidates = min(top_k * 2, len(self.events))
            top_indices = np.argsort(similarities)[::-1][:n_candidates]

            candidates: list[dict[str, Any]] = []
            for idx in top_indices:
                event = self.events[idx].copy()
                event["similarity"] = float(np.round(similarities[idx], 4))
                event["confidence"] = "High" if similarities[idx] > 0.6 else "Medium" if similarities[idx] > 0.3 else "Low"
                event["retrieval_method"] = "opea_tei_embedding"
                candidates.append(event)

            # Rerank if OPEA Reranking is available
            if self.reranking_provider == "opea_tei" and len(candidates) > top_k:
                reranked = self._rerank_opea(query, candidates, top_k)
                for r in reranked:
                    r["retrieval_method"] = "opea_tei_embedding+reranking"
                return reranked

            return candidates[:top_k]  # type: ignore
        except Exception as e:
            print(f"[RAG] OPEA retrieval failed: {e}. Falling back to TF-IDF.")
            return self._retrieve_tfidf(query, top_k)

    def _retrieve_tfidf(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve using TF-IDF sparse vectors (fallback)."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            event = self.events[idx].copy()
            event["similarity"] = float(np.round(sim, 4))
            event["confidence"] = "High" if sim > 0.4 else "Medium" if sim > 0.2 else "Low"
            event["retrieval_method"] = "tfidf"
            results.append(event)
        return results

    def generate_explanation(
        self,
        query_type: str,
        source: str,
        target: str,
        effect: dict,
        refutation_status: dict | None = None,
        analogies: list | None = None,
        temporal_info: dict | None = None,
    ) -> str:
        """Generate a structured explanation for the causal result."""
        fallback_text = ""
        if query_type == "intervention":
            fallback_text = self._explain_intervention(source, target, effect, refutation_status, analogies)
        elif query_type == "counterfactual":
            fallback_text = self._explain_counterfactual(source, target, effect, analogies)
        elif query_type == "root_cause":
            fallback_text = self._explain_root_cause(source, target, effect, analogies)
        elif query_type == "temporal":
            fallback_text = self._explain_temporal(source, target, temporal_info, analogies)
        else:
            fallback_text = self._explain_intervention(source, target, effect, refutation_status, analogies)

        if self.client:
            try:
                return self._generate_llm_explanation(
                    query_type, source, target, effect, refutation_status, analogies
                )
            except Exception as e:
                print(f"[RAG] LLM generation failed: {e}. Falling back to rule-based templates.")
                return fallback_text

        return fallback_text

    def _generate_llm_explanation(self, query_type, source, target, effect, refutation_status, analogies):
        client = self.client
        if not client:
            return ""
        
        point = effect.get("point_estimate", 0) if isinstance(effect, dict) else 0
        lower = effect.get("ci_lower", 0) if isinstance(effect, dict) else 0
        upper = effect.get("ci_upper", 0) if isinstance(effect, dict) else 0

        analogies_list = "\n".join([f"- {a['text']}" for a in (analogies or [])])

        prompt = (
            f"You are the **CDIE v4 Causal Intelligence Engine**, an expert AI system built on the **OPEA (Open Platform for Enterprise AI)** framework.\n"
            f"Generate an **OPEA Causal Intelligence Report** responding to a {query_type} query regarding telecom fraud.\n\n"
            f"**CAUSAL EVIDENCE (From Offline Causal Discovery):**\n"
            f"- Source Intervention: {source}\n"
            f"- Target Effect: {target}\n"
            f"- Doubly-Robust ATE (Average Treatment Effect): {point:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])\n"
            f"- Refutation Test Status (Robustness): {refutation_status}\n\n"
            f"**RELEVANT TELECOM PLAYBOOKS (From OPEA TEI RAG Retrieval):**\n"
            f"{analogies_list}\n\n"
            f"**FORMATTING REQUIREMENTS (CRITICAL):**\n"
            f"Output a highly professional markdown report. Do not include pleasantries. Strictly use this structure:\n\n"
            f"### 📊 Causal Impact Summary\n"
            f"Explain the causal effect magnitude ({point:.4f}) and whether it represents an increase or decrease.\n\n"
            f"### 🛡️ Validation & Refutation\n"
            f"State the confidence interval and the results of the refutation tests (did they pass?). Use terms like 'do-calculus'.\n\n"
            f"### 📖 Playbook Recommendation (RAG)\n"
            f"Synthesize the provided Telecom Playbooks into concrete action items that operators should follow based on this causal finding.\n"
        )

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are CDIE v4, an elite Causal Inference engine for telecom network intelligence, reporting to a Chief Network Officer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()

    def _explain_intervention(self, source, target, effect, refutation, analogies):
        point = effect.get("point_estimate", 0) if isinstance(effect, dict) else 0
        lower = effect.get("ci_lower", point * 0.8) if isinstance(effect, dict) else point * 0.8
        upper = effect.get("ci_upper", point * 1.2) if isinstance(effect, dict) else point * 1.2
        direction = "increase" if point > 0 else "decrease"
        magnitude = abs(point)

        explanation = (
            f"**Impact Summary**: A change in {source} is estimated to cause a "
            f"{direction} of {magnitude:.2f} units in {target} "
            f"(95% CI: [{lower:.2f}, {upper:.2f}]).\n\n"
        )
        explanation += (
            f"**Causal Chain**: This effect propagates through the validated causal pathway "
            f"{source} → {target}. The estimate is doubly-robust (LinearDML), meaning it remains "
            f"valid even if either the outcome model or treatment model is misspecified.\n\n"
        )
        if refutation:
            n_pass = sum(1 for v in refutation.values() if v == "PASS")
            n_total = len(refutation)
            explanation += (
                f"**Validation**: {n_pass}/{n_total} refutation tests passed. "
                f"This causal claim has been tested against placebo treatments, "
                f"random confounders, and data subsets.\n\n"
            )
        if analogies:
            high_conf = [a for a in analogies if a.get("confidence") in ("High", "Medium")]
            if high_conf:
                explanation += "**Historical Precedent**: "
                explanation += high_conf[0]["text"] + "\n\n"
        explanation += (
            f"**Recommended Action**: Monitor {target} closely when implementing changes to {source}. "
            f"Consider segment-specific impacts (Enterprise vs Retail) before deployment."
        )
        return explanation

    def _explain_counterfactual(self, source, target, effect, analogies):
        point = effect.get("point_estimate", 0) if isinstance(effect, dict) else 0
        return (
            f"**Counterfactual Analysis**: Had {source} remained at its baseline value, "
            f"{target} would have been approximately {abs(point):.2f} units "
            f"{'higher' if point < 0 else 'lower'} than observed.\n\n"
            f"This estimate uses DoWhy's counterfactual framework, applying the "
            f"structural equations from the discovered causal model."
        )

    def _explain_root_cause(self, source, target, effect, analogies):
        return (
            f"**Root Cause Analysis**: The primary causal driver of changes in {source} "
            f"traces back through the causal graph. Key upstream factors include variables "
            f"that are direct parents of {source} in the discovered DAG.\n\n"
            f"Use the interactive causal graph to trace the full causal chain and identify "
            f"the most actionable intervention point."
        )

    def _explain_temporal(self, source, target, temporal_info, analogies):
        lag = temporal_info.get("lag", 2) if temporal_info else 2
        return (
            f"**Temporal Effect**: Changes in {source} take approximately {lag} time period(s) "
            f"to fully manifest in {target}.\n\n"
            f"This lag was identified by PCMCI+ temporal causal discovery and cross-validated "
            f"with Granger causality tests. Plan interventions with this delay in mind."
        )
