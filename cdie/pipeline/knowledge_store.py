"""
CDIE v5 — Knowledge Store
SQLite-backed versioned store for extracted causal priors.
Replaces flat JSON files with an auditable, versioned knowledge base.
Supports: prior ingestion, conflict detection, HITL adjudication, cold-start seeding.
"""
import json
import sqlite3
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from cdie.pipeline.data_generator import VARIABLE_NAMES, DATA_DIR

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Versioned causal prior store with conflict detection and adjudication."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DATA_DIR / "knowledge.db")
        self._init_db()
        self._ensure_cold_start_seed()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS priors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    origin TEXT DEFAULT 'extracted',
                    source_document TEXT,
                    created_at TEXT NOT NULL,
                    active INTEGER DEFAULT 1,
                    UNIQUE(source, target, origin)
                );

                CREATE TABLE IF NOT EXISTS conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prior_source TEXT NOT NULL,
                    prior_target TEXT NOT NULL,
                    prior_confidence REAL NOT NULL,
                    dag_source TEXT,
                    dag_target TEXT,
                    conflict_type TEXT NOT NULL,
                    description TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolution TEXT,
                    resolved_by TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );

                CREATE TABLE IF NOT EXISTS adjudication_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conflict_id INTEGER,
                    action TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conflict_id) REFERENCES conflicts(id)
                );
            """)

    def _ensure_cold_start_seed(self):
        """Load public playbook seed if no custom priors exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM priors").fetchone()[0]
            if count > 0:
                return

        seed_path = DATA_DIR / "public_playbook_seed.json"
        if not seed_path.exists():
            logger.info("[KnowledgeStore] No seed file found — starting empty.")
            return

        try:
            with open(seed_path) as f:
                seed_priors = json.load(f)
            self.add_priors(seed_priors, origin="public_seed", source_document="GSMA/ITU Public Fraud Intelligence")
            logger.info(f"[KnowledgeStore] Cold-start: loaded {len(seed_priors)} public seed priors.")
        except Exception as e:
            logger.error(f"[KnowledgeStore] Failed to load seed: {e}")

    def add_priors(
        self,
        priors: List[Dict[str, Any]],
        origin: str = "extracted",
        source_document: str = "",
    ) -> Dict[str, Any]:
        """
        Add extracted priors to the store.
        Returns summary with counts and any conflicts detected.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        added = 0
        updated = 0
        skipped = 0

        with sqlite3.connect(str(self.db_path)) as conn:
            for p in priors:
                src = p.get("source", "")
                tgt = p.get("target", "")
                conf = float(p.get("confidence", 0))

                if src not in VARIABLE_NAMES or tgt not in VARIABLE_NAMES:
                    skipped += 1
                    continue

                existing = conn.execute(
                    "SELECT id, confidence FROM priors WHERE source=? AND target=? AND origin=?",
                    (src, tgt, origin),
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE priors SET confidence=?, source_document=?, created_at=? WHERE id=?",
                        (conf, source_document, now, existing[0]),
                    )
                    updated += 1
                else:
                    conn.execute(
                        "INSERT INTO priors (source, target, confidence, origin, source_document, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (src, tgt, conf, origin, source_document, now),
                    )
                    added += 1

        return {"added": added, "updated": updated, "skipped": skipped}

    def detect_conflicts(self, dag_edges: List[tuple]) -> List[Dict[str, Any]]:
        """
        Compare active priors against current DAG edges.
        Detect: reversed edges, missing edges, contradictions.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        dag_set = set(dag_edges)
        dag_reverse = {(t, s) for s, t in dag_edges}
        conflicts = []

        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, source, target, confidence FROM priors WHERE active=1 AND confidence > 0.5"
            ).fetchall()

            for prior_id, src, tgt, conf in rows:
                edge = (src, tgt)

                if edge in dag_reverse and edge not in dag_set:
                    conflict = {
                        "prior_source": src,
                        "prior_target": tgt,
                        "prior_confidence": conf,
                        "dag_source": tgt,
                        "dag_target": src,
                        "conflict_type": "REVERSED",
                        "description": (
                            f"Prior says {src} -> {tgt} (conf={conf:.2f}), "
                            f"but DAG has {tgt} -> {src}. Direction conflict."
                        ),
                    }

                    # Store conflict
                    conn.execute(
                        """INSERT OR IGNORE INTO conflicts 
                        (prior_source, prior_target, prior_confidence, dag_source, dag_target, 
                         conflict_type, description, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (src, tgt, conf, tgt, src, "REVERSED", conflict["description"], now),
                    )
                    conflicts.append(conflict)

                elif edge not in dag_set and edge not in dag_reverse and conf > 0.7:
                    conflict = {
                        "prior_source": src,
                        "prior_target": tgt,
                        "prior_confidence": conf,
                        "dag_source": None,
                        "dag_target": None,
                        "conflict_type": "MISSING",
                        "description": (
                            f"Prior says {src} -> {tgt} (conf={conf:.2f}), "
                            f"but DAG has no edge between these nodes. Discovery gap."
                        ),
                    }
                    conn.execute(
                        """INSERT OR IGNORE INTO conflicts
                        (prior_source, prior_target, prior_confidence, 
                         conflict_type, description, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (src, tgt, conf, "MISSING", conflict["description"], now),
                    )
                    conflicts.append(conflict)

        return conflicts

    def adjudicate_conflict(
        self, conflict_id: int, action: str, reason: str = ""
    ) -> Dict[str, Any]:
        """
        Resolve a conflict via HITL adjudication.
        Actions: 'accept_prior', 'reject_prior', 'defer'
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        with sqlite3.connect(str(self.db_path)) as conn:
            conflict = conn.execute(
                "SELECT * FROM conflicts WHERE id=?", (conflict_id,)
            ).fetchone()
            if not conflict:
                return {"success": False, "message": f"Conflict {conflict_id} not found."}

            conn.execute(
                "UPDATE conflicts SET resolved=1, resolution=?, resolved_by='expert', resolved_at=? WHERE id=?",
                (action, now, conflict_id),
            )
            conn.execute(
                "INSERT INTO adjudication_log (conflict_id, action, reason, created_at) VALUES (?, ?, ?, ?)",
                (conflict_id, action, reason, now),
            )

            if action == "reject_prior":
                conn.execute(
                    "UPDATE priors SET active=0 WHERE source=? AND target=?",
                    (conflict[1], conflict[2]),
                )

        return {
            "success": True,
            "conflict_id": conflict_id,
            "action": action,
            "message": f"Conflict {conflict_id} resolved as '{action}'.",
        }

    def get_active_priors(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Return all active priors above confidence threshold."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT source, target, confidence, origin, source_document, created_at FROM priors WHERE active=1 AND confidence >= ?",
                (min_confidence,),
            ).fetchall()
        return [
            {
                "source": r[0], "target": r[1], "confidence": r[2],
                "origin": r[3], "source_document": r[4], "created_at": r[5],
            }
            for r in rows
        ]

    def get_pending_conflicts(self) -> List[Dict[str, Any]]:
        """Return unresolved conflicts for HITL review."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, prior_source, prior_target, prior_confidence, dag_source, dag_target, conflict_type, description, created_at FROM conflicts WHERE resolved=0"
            ).fetchall()
        return [
            {
                "id": r[0], "prior_source": r[1], "prior_target": r[2],
                "prior_confidence": r[3], "dag_source": r[4], "dag_target": r[5],
                "conflict_type": r[6], "description": r[7], "created_at": r[8],
            }
            for r in rows
        ]

    def export_for_pipeline(self) -> List[Dict[str, Any]]:
        """Export active priors in the format expected by GFCI discovery."""
        return self.get_active_priors(min_confidence=0.70)
