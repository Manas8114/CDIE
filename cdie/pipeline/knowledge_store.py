"""
CDIE v5 — Knowledge Store
SQLite-backed versioned store for extracted causal priors.
Replaces flat JSON files with an auditable, versioned knowledge base.
Supports: prior ingestion, conflict detection, HITL adjudication, cold-start seeding.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from cdie.pipeline.data_generator import DATA_DIR, VARIABLE_NAMES
from cdie.pipeline.datastore import DataStoreManager

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Versioned causal prior store with conflict detection and adjudication."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (DATA_DIR / 'knowledge.db')
        self._init_db()
        self._ensure_cold_start_seed()

    def _init_db(self) -> None:
        store = DataStoreManager.get_sqlite_store(self.db_path)
        store.execute_script("""
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

    def _ensure_cold_start_seed(self) -> None:
        """Load public playbook seed if no custom priors exist."""
        store = DataStoreManager.get_sqlite_store(self.db_path)
        count_row = store.fetch_one('SELECT COUNT(*) as count FROM priors')
        count = count_row['count'] if count_row else 0
        if count > 0:
            return

        seed_path = DATA_DIR / 'public_playbook_seed.json'
        if not seed_path.exists():
            logger.info('[KnowledgeStore] No seed file found — starting empty.')
            return

        try:
            with open(seed_path) as f:
                seed_priors = json.load(f)
            self.add_priors(
                seed_priors,
                origin='public_seed',
                source_document='GSMA/ITU Public Fraud Intelligence',
            )
            logger.info(f'[KnowledgeStore] Cold-start: loaded {len(seed_priors)} public seed priors.')
        except Exception as e:
            logger.error(f'[KnowledgeStore] Failed to load seed: {e}')

    def add_priors(
        self,
        priors: list[dict[str, Any]],
        origin: str = 'extracted',
        source_document: str = '',
    ) -> dict[str, Any]:
        """
        Add extracted priors to the store.
        Returns summary with counts and any conflicts detected.
        """
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ')
        added = 0
        updated = 0
        skipped = 0

        store = DataStoreManager.get_sqlite_store(self.db_path)
        for p in priors:
            src = p.get('source', '')
            tgt = p.get('target', '')
            conf = float(p.get('confidence', 0))

            if src not in VARIABLE_NAMES or tgt not in VARIABLE_NAMES:
                skipped += 1
                continue

            existing = store.fetch_one(
                'SELECT id, confidence FROM priors WHERE source=? AND target=? AND origin=?',
                (src, tgt, origin),
            )

            if existing:
                store.execute(
                    'UPDATE priors SET confidence=?, source_document=?, created_at=? WHERE id=?',
                    (conf, source_document, now, existing['id']),
                )
                updated += 1
            else:
                store.execute(
                    'INSERT INTO priors (source, target, confidence, origin, '
                    'source_document, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                    (src, tgt, conf, origin, source_document, now),
                )
                added += 1

        return {'added': added, 'updated': updated, 'skipped': skipped}

    def detect_conflicts(self, dag_edges: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """
        Compare active priors against current DAG edges.
        Detect: reversed edges, missing edges, contradictions.
        """
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ')
        dag_set = set(dag_edges)
        dag_reverse = {(t, s) for s, t in dag_edges}
        conflicts = []

        store = DataStoreManager.get_sqlite_store(self.db_path)
        rows = store.fetch_all('SELECT id, source, target, confidence FROM priors WHERE active=1 AND confidence > 0.5')

        for row in rows:
            _prior_id, src, tgt, conf = row['id'], row['source'], row['target'], row['confidence']
            edge = (src, tgt)

            if edge in dag_reverse and edge not in dag_set:
                conflict = {
                    'prior_source': src,
                    'prior_target': tgt,
                    'prior_confidence': conf,
                    'dag_source': tgt,
                    'dag_target': src,
                    'conflict_type': 'REVERSED',
                    'description': (
                        f'Prior says {src} -> {tgt} (conf={conf:.2f}), but DAG has {tgt} -> {src}. Direction conflict.'
                    ),
                }

                # Store conflict
                store.execute(
                    """INSERT OR IGNORE INTO conflicts
                    (prior_source, prior_target, prior_confidence, dag_source, dag_target,
                        conflict_type, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        src,
                        tgt,
                        conf,
                        tgt,
                        src,
                        'REVERSED',
                        conflict['description'],
                        now,
                    ),
                )
                conflicts.append(conflict)

            elif edge not in dag_set and edge not in dag_reverse and conf > 0.7:
                conflict = {
                    'prior_source': src,
                    'prior_target': tgt,
                    'prior_confidence': conf,
                    'dag_source': None,
                    'dag_target': None,
                    'conflict_type': 'MISSING',
                    'description': (
                        f'Prior says {src} -> {tgt} (conf={conf:.2f}), '
                        f'but DAG has no edge between these nodes. Discovery gap.'
                    ),
                }
                store.execute(
                    """INSERT OR IGNORE INTO conflicts
                    (prior_source, prior_target, prior_confidence,
                        conflict_type, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (src, tgt, conf, 'MISSING', conflict['description'], now),
                )
                conflicts.append(conflict)

        return conflicts

    def adjudicate_conflict(self, conflict_id: int, action: str, reason: str = '') -> dict[str, Any]:
        """
        Resolve a conflict via HITL adjudication.
        Actions: 'accept_prior', 'reject_prior', 'defer'
        """
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ')

        store = DataStoreManager.get_sqlite_store(self.db_path)
        conflict = store.fetch_one('SELECT * FROM conflicts WHERE id=?', (conflict_id,))
        if not conflict:
            return {
                'success': False,
                'message': f'Conflict {conflict_id} not found.',
            }

        store.execute(
            "UPDATE conflicts SET resolved=1, resolution=?, resolved_by='expert', resolved_at=? WHERE id=?",
            (action, now, conflict_id),
        )
        store.execute(
            'INSERT INTO adjudication_log (conflict_id, action, reason, created_at) VALUES (?, ?, ?, ?)',
            (conflict_id, action, reason, now),
        )

        if action == 'reject_prior':
            store.execute(
                'UPDATE priors SET active=0 WHERE source=? AND target=?',
                (conflict['prior_source'], conflict['prior_target']),
            )

        return {
            'success': True,
            'conflict_id': conflict_id,
            'action': action,
            'message': f"Conflict {conflict_id} resolved as '{action}'.",
        }

    def get_active_priors(self, min_confidence: float = 0.0) -> list[dict[str, Any]]:
        """Return all active priors above confidence threshold."""
        store = DataStoreManager.get_sqlite_store(self.db_path)
        rows = store.fetch_all(
            "SELECT source, target, confidence, origin, source_document, created_at "
            "FROM priors WHERE active=1 AND confidence >= ?",
            (min_confidence,),
        )
        return [
            {
                'source': r['source'],
                'target': r['target'],
                'confidence': r['confidence'],
                'origin': r['origin'],
                'source_document': r['source_document'],
                'created_at': r['created_at'],
            }
            for r in rows
        ]

    def get_pending_conflicts(self) -> list[dict[str, Any]]:
        """Return unresolved conflicts for HITL review."""
        store = DataStoreManager.get_sqlite_store(self.db_path)
        rows = store.fetch_all(
            "SELECT id, prior_source, prior_target, prior_confidence, "
            "dag_source, dag_target, conflict_type, description, created_at "
            "FROM conflicts WHERE resolved=0"
        )
        return [
            {
                'id': r['id'],
                'prior_source': r['prior_source'],
                'prior_target': r['prior_target'],
                'prior_confidence': r['prior_confidence'],
                'dag_source': r['dag_source'],
                'dag_target': r['dag_target'],
                'conflict_type': r['conflict_type'],
                'description': r['description'],
                'created_at': r['created_at'],
            }
            for r in rows
        ]

    def export_for_pipeline(self) -> list[dict[str, Any]]:
        """Export active priors in the format expected by GFCI discovery."""
        return self.get_active_priors(min_confidence=0.70)
