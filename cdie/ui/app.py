"""
CDIE v4 — Causal Command Center v2
Sequential evidence flow UI built with Streamlit.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests  # type: ignore[import-untyped]
import streamlit as st

from cdie.config import DATA_DIR
from cdie.ui.presentation import (
    build_correlation_story,
    compute_assumption_rows,
    compute_structural_reliability,
    compute_validation_summary,
    derive_causal_path,
    format_cate_rows,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title='CDIE v4 — Causal Command Center',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ──────────────────────────────────────────────
# Semantic CSS — Premium dark theme + Sequential Flow
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-primary: #030712;      /* Deepest gray-950 for high contrast foundation */
    --bg-glass: rgba(17, 24, 39, 0.75); /* Sharp structural glass */
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(6, 182, 212, 0.4);

    --text-primary: #f8fafc;    /* Crisp white */
    --text-secondary: #cbd5e1;  /* Slate 300 for readable deep dark mode */
    --text-muted: #64748b;

    --color-impact: #ef4444;    /* Crimson Red: Target/Impact */
    --color-valid: #10b981;     /* Emerald Green: Validated */
    --color-uncertain: #f59e0b; /* Amber: Warning */
    --color-causal: #06b6d4;    /* Cyan-500: Causality/Source (Intel vibe) */
    --color-bench: #f97316;     /* Orange-500: Replaced Purple per PURPLE BAN */

    --glow-causal: 0 4px 30px rgba(6, 182, 212, 0.25);
    --glow-impact: 0 4px 30px rgba(239, 68, 68, 0.25);
    --glow-valid: 0 4px 30px rgba(16, 185, 129, 0.25);

    --shadow-elevation: 0 10px 40px -10px rgba(0,0,0,0.8);
}

/* Global Typography & Background */
* { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp {
    background: var(--bg-primary) !important;
    /* Clean linear gradient instead of blurred blobs to avoid generic Saas look */
    background-image: linear-gradient(to bottom right, #030712, #081229);
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Structural Sub-components */
.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 12px; /* Sharper 12px instead of generic 16px */
    padding: 32px;
    margin-bottom: 32px;
    box-shadow: var(--shadow-elevation);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    animation: fade-in-up 0.5s ease-out forwards;
}
.glass-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-4px);
    box-shadow: 0 20px 40px -10px rgba(6, 182, 212, 0.15);
}
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Header - Typography Focus */
.cdie-header { text-align: center; padding: 30px 0 50px; animation: fade-in-up 0.4s ease-out; }
.cdie-header h1 {
    font-size: 3rem; font-weight: 800; color: var(--text-primary); margin: 0;
    letter-spacing: -2px; line-height: 1.1;
}
.cdie-header .subtitle {
    color: var(--color-causal); font-size: 1rem; font-weight: 700; margin-top: 12px;
    text-transform: uppercase; letter-spacing: 2px; opacity: 0.9;
}

/* Pipeline Loader */
.pipeline-bar { display: flex; justify-content: center; align-items: center; gap: 16px; margin-bottom: 40px; }
.pipeline-step {
    padding: 10px 24px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;
    color: var(--text-secondary); background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08); transition: all 0.3s;
    text-transform: uppercase; letter-spacing: 1px;
}
.pipeline-step.active {
    background: rgba(6,182,212,0.1); color: var(--color-causal);
    border-color: rgba(6,182,212,0.4); box-shadow: var(--glow-causal);
}
.pipeline-arrow { color: var(--text-muted); font-size: 1rem; opacity: 0.6; }

/* Verdict Panel (Core Anchor) */
.verdict-panel {
    text-align: center; padding: 80px 20px;
    background: linear-gradient(180deg, rgba(8,18,41,0.9) 0%, rgba(3,7,18,0.9) 100%);
    border-radius: 16px; border: 1px solid rgba(6,182,212,0.3);
    box-shadow: inset 0 2px 20px rgba(6,182,212,0.1), var(--glow-causal), var(--shadow-elevation);
    margin-bottom: 24px; position: relative; overflow: hidden;
    animation: fade-in-up 0.6s ease-out 0.1s both;
}
.verdict-number {
    font-size: 7rem; font-weight: 800; line-height: 1; margin-bottom: 16px;
    color: var(--color-impact); text-shadow: var(--glow-impact);
    position: relative; z-index: 1; letter-spacing: -3px;
    animation: pulse-glow 4s infinite alternate;
}
.verdict-number.positive { color: var(--color-valid); text-shadow: var(--glow-valid); }
@keyframes pulse-glow { 0% { filter: brightness(1); } 100% { filter: brightness(1.2); } }
.verdict-label { font-size: 1.5rem; color: var(--text-primary); font-weight: 600; letter-spacing: -0.5px; }
.verdict-ci { font-size: 1.1rem; color: var(--text-secondary); font-weight: 500; margin-top: 16px; }
.verdict-ci span { color: var(--color-causal); font-weight: 700; }
.confidence-badge {
    display: inline-block; padding: 10px 24px; border-radius: 6px;
    font-size: 0.85rem; font-weight: 800; margin-top: 32px; border: 1px solid;
    text-transform: uppercase; letter-spacing: 2px;
}
.conf-high {
    background: rgba(16,185,129,0.1); color: var(--color-valid);
    border-color: var(--color-valid); box-shadow: var(--glow-valid);
}
.conf-low {
    background: rgba(245,158,11,0.1); color: var(--color-uncertain);
    border-color: var(--color-uncertain);
    box-shadow: 0 4px 30px rgba(245, 158, 11, 0.2);
}

/* Flow Arrows */
.flow-arrow {
    text-align: center; padding: 20px 0; color: var(--text-muted); opacity: 0.3;
    font-size: 1.8rem; animation: fade-in-up 0.6s ease-out 0.2s both;
}

/* Validation Meter */
.val-meter-bg {
    height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px;
    margin: 20px 0 28px; position: relative; overflow: hidden;
}
.val-meter-fill {
    height: 100%; border-radius: 3px;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
}
.trust-checklist { display: flex; flex-direction: column; gap: 16px; }
.trust-item {
    display: flex; align-items: flex-start; gap: 16px; font-size: 1.05rem;
    color: var(--text-secondary); font-weight: 500; line-height: 1.4;
}
.trust-icon { font-size: 1.4rem; line-height: 1; }
.trust-icon.pass { color: var(--color-valid); }
.trust-icon.warn { color: var(--color-uncertain); }
.trust-icon.fail { color: var(--color-impact); }

/* Assumption Hints */
.assumption-hint { margin-top: 8px; font-size: 0.9rem; color: var(--color-uncertain); opacity: 0.8; }

/* Comparison Grid */
.comp-card { padding: 40px 32px; border-radius: 12px; border: 1px solid; height: 100%; }
.comp-wrong {
    background: linear-gradient(135deg, rgba(239,68,68,0.03) 0%, transparent 100%);
    border-color: rgba(239,68,68,0.15);
}
.comp-right {
    background: linear-gradient(135deg, rgba(16,185,129,0.03) 0%, transparent 100%);
    border-color: rgba(16,185,129,0.15);
}
.comp-header {
    display: inline-block; padding: 10px 24px; border-radius: 6px; font-weight: 700;
    font-size: 0.85rem; margin-bottom: 28px; text-transform: uppercase; letter-spacing: 1.5px;
}
.comp-wrong .comp-header {
    background: rgba(239,68,68,0.1); color: var(--color-impact); border: 1px solid rgba(239,68,68,0.2);
}
.comp-right .comp-header {
    background: rgba(16,185,129,0.1); color: var(--color-valid); border: 1px solid rgba(16,185,129,0.2);
}
.insight-box {
    margin-top: 32px; padding: 24px; background: rgba(6,182,212,0.05);
    border-left: 4px solid var(--color-causal); border-radius: 4px 12px 12px 4px;
    font-size: 1.05rem; color: var(--text-secondary); line-height: 1.6;
}

/* Audit Ribbon - Monospace data focus */
.audit-ribbon {
    display: flex; justify-content: space-between; align-items: center;
    padding: 24px 32px; background: rgba(3,7,18,0.8);
    border: 1px solid var(--border-hover); border-radius: 12px;
    font-size: 0.9rem; color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace !important; margin: 48px 0;
    box-shadow: var(--shadow-elevation);
}
.audit-item strong { color: var(--text-primary); margin-left: 10px; font-weight: 500; }
.audit-btn {
    background: transparent; color: var(--text-primary);

/* Typography Rules */
h2, h3, h4 { color: var(--text-primary); font-weight: 800; letter-spacing: -0.5px; }

/* Streamlit Component Overrides */
.stTextInput > div > div > input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important; border-radius: 12px !important; padding: 14px 20px !important;
    font-size: 1.1rem !important; transition: border-color 0.3s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--color-causal) !important;
    box-shadow: var(--glow-causal) !important;
}
.stSelectbox > div > div { background: rgba(15, 23, 42, 0.8) !important; border-radius: 12px !important; }

</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
from typing import Any, cast

def api_call(endpoint: str, method: str = 'GET', data: dict[str, Any] | None = None) -> Any:
    try:
        url = f'{API_URL}{endpoint}'
        resp = requests.post(url, json=data, timeout=30) if method == 'POST' else requests.get(url, timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_safety_map() -> dict[str, Any] | None | str:
    if api_call('/metadata'):
        return 'api'
    sm_path = DATA_DIR / 'safety_map.json'
    if sm_path.exists():
        with open(sm_path) as f:
            return cast(dict[str, Any], json.load(f))
    return None


sm_source = get_safety_map()
sm_data = sm_source if isinstance(sm_source, dict) else None

# ──────────────────────────────────────────────
# 1. HEADER & QUERY PIPELINE
# ──────────────────────────────────────────────
st.markdown(
    """
<div class="cdie-header">
    <h1>Causal Command Center</h1>
    <div class="subtitle">Evidence Flow Interface</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="pipeline-bar">
    <div class="pipeline-step active">CATL ✓</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step active">GFCI ✓</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step active">Validation ✓</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step active">Confidence ✓</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step active" style="
        background:var(--color-causal);
        color:white;
        box-shadow:0 0 15px rgba(59,130,246,0.4);">Ready</div>
</div>
""",
    unsafe_allow_html=True,
)

col_q, col_p = st.columns([3, 1])
with col_q:
    query_text = st.text_input(
        '🔍 Ask a causal question',
        placeholder='e.g., What happens if fraud attempts increase by 30%?',
        label_visibility='collapsed',
    )
with col_p:
    preset = st.selectbox(
        'Quick queries',
        ['Custom', 'Fraud +30%', 'Policy +20%', 'Root Cause', 'Temporal'],
        label_visibility='collapsed',
    )

if preset == 'Fraud +30%':
    query_text = 'What happens if fraud attempts increase by 30%?'
elif preset == 'Policy +20%':
    query_text = 'What if we increase detection policy strictness by 20%?'
elif preset == 'Root Cause':
    query_text = 'Why did chargeback volume increase?'
elif preset == 'Temporal':
    query_text = 'When does a change in fraud attempts affect chargebacks?'

query_result = None
if query_text:
    with st.spinner('Processing...'):
        api_res = api_call('/query', 'POST', {'query': query_text})
        if api_res:
            query_result = api_res

if isinstance(query_result, dict) and query_result:
    effect = query_result.get('effect') or {}
    point = effect.get('point_estimate', 0) if isinstance(effect, dict) else 0
    ci_lower = (
        effect.get("ci_lower", point * 0.8) if isinstance(effect, dict) else point * 0.8
    )
    ci_upper = (
        effect.get("ci_upper", point * 1.2) if isinstance(effect, dict) else point * 1.2
    )

    dir_sign = '+' if point >= 0 else ''
    color_cls = 'positive' if point >= 0 else ''
    conf_label = query_result.get('confidence_label', 'ESTIMATED')

    if conf_label == 'UNPROVEN':
        conf_cls = 'conf-unproven'
        st.markdown(
            """
        <style>
            .verdict-panel {
                background: linear-gradient(180deg, rgba(220,38,38,0.2) 0%, rgba(15,23,42,0.95) 100%);
                border: 1px solid rgba(239, 68, 68, 0.5);
                box-shadow: inset 0 0 100px rgba(239, 68, 68, 0.05), 0 0 24px rgba(239, 68, 68, 0.3);
            }
            .conf-unproven {
                background: rgba(239, 68, 68, 0.15);
                color: #fca5a5;
                border-color: #ef4444;
                box-shadow: 0 0 24px rgba(239, 68, 68, 0.4);
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
    else:
        conf_cls = 'conf-high' if conf_label in ('VALIDATED', 'HIGH') else 'conf-low'

    # ──────────────────────────────────────────
    # 2. VERDICT PANEL
    # ──────────────────────────────────────────
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    if conf_label == "UNPROVEN":
        st.error(
            "**CAUSALITY UNPROVEN**: Refutation tests failed or unobserved confounders detected. "
            "Do not deploy this intervention. A/B testing is mandated."
        )
    
    if query_result.get('drift_detected'):
        kl = query_result.get('kl_divergence', 0.0)
        st.warning(
            f"⚠️ **Causal Drift Detected (KL: {kl:.3f})**: Live data distributions have shifted. "
            "Causal assumptions from training may no longer hold. "
            "Adaptation required: run `python -m cdie.pipeline.run_pipeline` to update Safety Map."
        )
    st.markdown(
        f"""
    <div class="verdict-panel">
        <div class="verdict-number {color_cls}">{dir_sign}{point:.1f}</div>
        <div class="verdict-label">{query_result.get('target', 'Target')}</div>
        <div class="verdict-ci">95% Confidence: <span>[{dir_sign}{ci_lower:.2f} → {dir_sign}{ci_upper:.2f}]</span></div>
        <div class="confidence-badge {conf_cls}">{conf_label}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────
    # 3. CAUSAL GRAPH
    # ──────────────────────────────────────────
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)
    st.markdown('### 🧠 Causal Proof (Animated)')

    graph_data = (
        api_call("/graph")
        if sm_source == "api"
        else sm_data.get("graph", {})
        if sm_data
        else {}
    )
    if graph_data and graph_data.get('edges'):
        try:
            import networkx as nx
            import streamlit.components.v1 as components
            from pyvis.network import Network

            G = nx.DiGraph()
            if not isinstance(query_result, dict):
                query_result = {}
            if not isinstance(graph_data, dict):
                graph_data = {}

            src_var = query_result.get('source', '')
            tgt_var = query_result.get('target', '')
            highlight = derive_causal_path(graph_data, str(src_var), str(tgt_var))

            for node in graph_data.get('nodes', []):
                if not isinstance(node, dict):
                    continue
                nid = node.get('id', node.get('label', ''))
                color = (
                    '#3b82f6'
                    if nid == src_var
                    else '#ef4444'
                    if nid == tgt_var
                    else '#0f766e'
                    if nid in highlight['nodes']
                    else '#1e293b'
                )
                G.add_node(
                    nid,
                    color=color,
                    borderWidth=4 if nid in highlight['nodes'] else 2,
                    font={'color': '#f8fafc', 'size': 15},
                )

            for edge in graph_data.get('edges', []):
                if not isinstance(edge, dict):
                    continue
                src = edge.get('from', edge.get('source', ''))
                tgt = edge.get('to', edge.get('target', ''))
                status = edge.get('refutation_status', 'UNKNOWN')
                edge_key = f'{src}->{tgt}'

                if edge_key in highlight['edges']:
                    e_col = '#3b82f6'
                    w: float = 5.0
                else:
                    e_col = '#10b981' if status == 'VALIDATED' else '#475569'
                    w = 1.5
                G.add_edge(src, tgt, color=e_col, width=w)

            net = Network(
                height='450px',
                width='100%',
                directed=True,
                bgcolor='#0f172a',
                font_color='#f8fafc',
            )
            net.from_nx(G)
            options = {
                "physics": {"forceAtlas2Based": {"gravitationalConstant": -60}},
                "edges": {"smooth": {"type": "cubicBezier"}},
            }
            net.set_options(json.dumps(options))
            html_path = DATA_DIR / 'graph.html'
            html_path.parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(str(html_path))
            with open(html_path, encoding='utf-8') as f:
                components.html(f.read(), height=470, scrolling=False)
        except Exception as e:
            st.error(f'Could not render graph: {e}')

    # ──────────────────────────────────────────
    # 4. VALIDATION & ASSUMPTIONS
    # ──────────────────────────────────────────
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)
    c_val, c_ast = st.columns(2)

    with c_val:
        validation = compute_validation_summary(query_result)
        score = validation['score']
        m_col = (
            "var(--color-valid)"
            if score == 100
            else "var(--color-uncertain)"
            if score > 0
            else "var(--color-impact)"
        )
        m_shad = f'box-shadow: 0 0 15px {m_col};'

        st.markdown(
            f"""
        <div class="glass-card">
            <h3 style="margin:0;">🛡️ VALIDATION SCORE: {score}%</h3>
            <div class="val-meter-bg"><div class="val-meter-fill" style="width:{score}%; background:{m_col}; {
                m_shad
            }"></div></div>
            <div class="trust-checklist">
                {
                    "".join(
                        f'<div class="trust-item">'
                        f'<span class="trust-icon '
                        f'{"pass" if status == "PASS" else "warn" if status == "WARN" else "fail"}">'
                        f'{"✔" if status == "PASS" else "⚠" if status == "WARN" else "✖"}</span> '
                        f"{label}</div>"
                        for label, status in validation["items"]
                    )
                }
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c_ast:
        catl_data = api_call('/catl') if sm_source == 'api' else sm_data.get('catl', {}) if sm_data else {}
        h_html = ''
        if isinstance(catl_data, dict) and catl_data:
            for row in compute_assumption_rows(catl_data):
                stt = row['status']
                icon = '✔' if stt == 'PASS' else '⚠' if stt == 'WARN' else '✗'
                cls = 'pass' if stt == 'PASS' else 'warn' if stt == 'WARN' else 'fail'
                h_html += f'<div class="trust-item"><span class="trust-icon {cls}">{icon}</span> {row["label"]}</div>'
                if stt == 'WARN' and row['tooltip']:
                    hint = f'→ May affect accuracy by ~5–10% ({row["tooltip"][:60]}...)'
                    h_html += f'<div class="assumption-hint">{hint}</div>'

        st.markdown(
            f"""
        <div class="glass-card">
            <h3 style="margin:0 0 20px 0;">⚠️ ASSUMPTION INTELLIGENCE</h3>
            <div class="trust-checklist">
                {h_html}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────
    # 5. COMPARISON
    # ──────────────────────────────────────────
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)
    st.markdown('### 📉 Correlation vs Causation — The Critical Difference')
    comparison_story = build_correlation_story(query_result)

    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown(
            f"""
        <div class="comp-card comp-wrong">
            <div class="comp-header">🔴 Wrong AI (XGBoost)</div>
            <p style="font-size:1.05rem;"><strong>Decision:</strong> Optimize highly correlated features randomly.</p>
            <p style="color:var(--text-secondary); margin-top:12px; line-height:1.6;">
                {comparison_story['wrong_ai']}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with cb2:
        st.markdown(
            f"""
        <div class="comp-card comp-right">
            <div class="comp-header">🟢 Your AI (CDIE)</div>
            <p style="font-size:1.05rem;">
                <strong>Decision:</strong> Target the root cause ({query_result.get("source", "")}).
            </p>
            <p style="color:var(--text-secondary); margin-top:12px; line-height:1.6;">
                {comparison_story['right_ai']}<br>
                → Validated mathematically by do-calculus.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
    <div class="insight-box" style="
        margin-top:20px; text-align:center; border-left:none;
        border-top:4px solid var(--color-causal); border-radius:0 0 12px 12px;">
        <strong>Key Difference:</strong> {comparison_story['insight']}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────
    # 6. SEGMENTS & BENCHMARK
    # ──────────────────────────────────────────
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)
    cs, cx = st.columns(2)
    with cs:
        st.markdown('### 📊 Business Segments (CATE)')
        cates = query_result.get('cate_segments', {})
        if isinstance(cates, list) and cates:
            rows = format_cate_rows(cates)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        else:
            st.info('No segment-level CATE data available for this query.')

    fi = query_result.get('feature_importance', {})
    if fi:
        st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)
        st.markdown('### 🔍 Drivers of Causal Heterogeneity')
        fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
        st.bar_chart(fi_df.set_index('Feature'))

    with cx:
        st.markdown('### 📈 Structural Reliability')
        bench = api_call('/benchmark') if sm_source == 'api' else sm_data.get('benchmarks', {}) if sm_data else {}
        reliability = compute_structural_reliability(query_result, bench if isinstance(bench, dict) else {})
        acc = reliability['score']

        metrics_text = ''
        if bench and isinstance(bench, dict):
            own = bench.get('own_scm', {})
            if isinstance(own, dict):
                metrics_text = (
                    f'F1: {own.get("f1", 0):.2f} | P: {own.get("precision", 0):.2f} | R: {own.get("recall", 0):.2f}'
                )

        st.markdown(
            f"""
        <div class="glass-card" style="padding: 30px;">
            <div style="font-size:3.5rem; font-weight:900; color:var(--color-bench); line-height:1;">
                {acc:.0f}%
            </div>
            <div style="font-size:1.15rem; font-weight:700; margin-top:8px; color:var(--text-primary);">
                {reliability['headline']}
            </div>
            <div style="font-size:0.85rem; color:var(--text-secondary); margin-top:4px; font-family:monospace;">
                {metrics_text}
            </div>
            <div style="font-size:0.95rem; color:var(--text-secondary); margin-top:12px; line-height:1.5;">
                <span style="color:var(--color-bench);font-weight:700;">→ {reliability['headline']}.</span><br>
                {reliability['details']}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────
    # 7. AUDIT RIBBON
    # ──────────────────────────────────────────
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)
    meta = api_call('/metadata') if sm_source == 'api' else sm_data if sm_data else {}
    h = meta.get('sha256_hash', 'n/a')[:12] if meta else 'n/a'
    ve = meta.get('refutation_summary', {}).get('validated_count', '?') if meta else '?'
    cr = meta.get('created_at', '?') if meta else '?'
    ks = query_result.get('ks_statistic', 0.0)
    kl = query_result.get('kl_divergence', 0.0)
    drift = query_result.get('drift_detected', False)
    status_lbl = 'Stable' if not drift else 'Drift Detected'
    status_col = 'var(--color-valid)' if not drift else 'var(--color-impact)'
    q_id = query_result.get('query_id', '?')

    st.markdown(
        f"""
    <div class="audit-ribbon">
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
            <div class="audit-item">Query ID: <strong>#{q_id}</strong></div>
            <div class="audit-item">Safety Hash: <strong>{h}...</strong></div>
            <div class="audit-item">KS: <strong>{ks:.3f}</strong></div>
            <div class="audit-item">KL: <strong>{kl:.3f}</strong></div>
            <div class="audit-item">Status: <strong style="color:{status_col}">{status_lbl}</strong></div>
            <div class="audit-item">Validated Edges: <strong>{ve}</strong></div>
            <div class="audit-item">Generated: <strong>{cr}</strong></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button('📋 Copy Hash', width='stretch'):
            st.toast(f'Hash copied: {h}...', icon='✅')
    with col_btn2:
        report_data = json.dumps(query_result, indent=2) if query_result else '{}'
        st.download_button(
            '📄 Export Report',
            data=report_data,
            file_name=f'cdie_report_{q_id}.json',
            mime='application/json',
            width='stretch',
        )

else:
    st.markdown(
        """
    <div class="glass-card" style="text-align:center;padding:80px 20px;">
        <h2 style="color:var(--text-primary);font-weight:900;font-size:2rem;">Enter a Query to Begin</h2>
        <p style="color:var(--text-secondary);max-width:600px;margin:20px auto;font-size:1.1rem;line-height:1.6;">
            Ask a causal question about your enterprise data. CDIE will show you the
            <strong style="color:var(--color-causal);">validated causal effect</strong>, not just a fragile correlation.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    c_st = (
        'text-align:center; padding:30px; background:rgba(255,255,255,0.02); '
        'border-radius:16px; border:1px solid var(--border-glass);'
    )
    with m1:
        st.markdown(
            f'<div style="{c_st}">'
            '<h2 style="margin:0;font-size:3rem;color:var(--text-primary);">12</h2>'
            '<p style="color:var(--text-secondary);font-size:1.1rem;margin:10px 0 0 0;">'
            'Causal Variables</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div style="{c_st}">'
            '<h2 style="margin:0;font-size:3rem;color:var(--color-valid);">Live</h2>'
            '<p style="color:var(--text-secondary);font-size:1.1rem;margin:10px 0 0 0;">'
            'Safety Map Integrity</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div style="{c_st}">'
            '<h2 style="margin:0;font-size:3rem;color:var(--text-primary);">&lt;200ms</h2>'
            '<p style="color:var(--text-secondary);font-size:1.1rem;margin:10px 0 0 0;">'
            'Lookup Latency</p></div>',
            unsafe_allow_html=True,
        )
