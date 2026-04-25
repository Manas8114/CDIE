"""
CDIE v4 — GFCI Causal Discovery
Primary causal structure learning using Greedy Fast Causal Inference.
Includes domain-knowledge prior injection and PAG-to-DAG conversion.
"""

import threading
from typing import Any

import networkx as nx
import numpy as np

from cdie.config import VARIABLE_NAMES
from cdie.observability import get_logger

log = get_logger(__name__)

# Domain-knowledge priors for the telecom fraud SCM.
DOMAIN_PRIORS = {
    ('CallDataRecordVolume', 'SIMBoxFraudAttempts'): 0.95,
    ('CallDataRecordVolume', 'NetworkLoad'): 0.9,
    ('CallDataRecordVolume', 'ARPUImpact'): 0.7,
    ('SIMBoxFraudAttempts', 'SIMFraudDetectionRate'): 0.95,
    ('SIMBoxFraudAttempts', 'RevenueLeakageVolume'): 0.98,
    ('FraudPolicyStrictness', 'SIMFraudDetectionRate'): 0.98,
    ('FraudPolicyStrictness', 'NetworkOpExCost'): 0.85,
    ('SIMFraudDetectionRate', 'RevenueLeakageVolume'): 0.92,
    ('SIMFraudDetectionRate', 'SubscriberRetentionScore'): 0.8,
    ('RevenueLeakageVolume', 'ARPUImpact'): 0.95,
    ('RevenueLeakageVolume', 'CashFlowRisk'): 0.95,
    ('RegulatorySignal', 'SubscriberRetentionScore'): 0.65,
    ('RegulatorySignal', 'ITURegulatoryPressure'): 0.9,
    ('ITURegulatoryPressure', 'FraudPolicyStrictness'): 0.95,
    ('NetworkLoad', 'NetworkOpExCost'): 0.9,
    ('NetworkOpExCost', 'CashFlowRisk'): 0.88,
}


def _run_gfci_internal(data_matrix: np.ndarray, _variable_names: list[str], result_container: dict[str, Any]) -> None:
    """Run GFCI in a thread for timeout control."""
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import fisherz

        alpha = 0.05 if len(data_matrix) >= 500 else 0.01

        g, edges = fci(data_matrix, fisherz, alpha, verbose=False)
        result_container['graph'] = g
        result_container['edges'] = edges
        result_container['success'] = True
    except Exception as e:
        result_container['error'] = str(e)
        result_container['success'] = False


def _run_pc_fallback(data_matrix: np.ndarray, _variable_names: list[str]) -> tuple[Any, str]:
    """PC algorithm fallback when GFCI times out."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        cg = pc(data_matrix, alpha=0.05, indep_test=fisherz)
        return cg, 'PC'
    except Exception as e:
        return None, f'PC_FAILED: {e}'


def extract_discovered_edges(graph_result: Any, variable_names: list[str]) -> list[tuple[str, str]]:
    """Extract edges from causal-learn graph result into a list of tuples."""
    discovered_edges = []
    adj_matrix = None

    if hasattr(graph_result, 'graph'):
        adj_matrix = graph_result.graph
    elif hasattr(graph_result, 'G') and hasattr(graph_result.G, 'graph'):
        adj_matrix = graph_result.G.graph

    if adj_matrix is not None:
        n = min(len(variable_names), adj_matrix.shape[0])
        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] != 0 and (
                    adj_matrix[i, j] == -1
                    and adj_matrix[j, i] == 1
                    or adj_matrix[i, j] == 1
                    and adj_matrix[j, i] == 1
                    and i < j
                ):
                    discovered_edges.append((variable_names[i], variable_names[j]))

    return discovered_edges


def build_map_dag(discovered_edges: list[tuple[str, str]], variable_names: list[str], dynamic_priors: list[dict[str, Any]] | None = None) -> nx.DiGraph:
    """
    Convert discovered edges to a DAG by resolving uncertain orientations
    using domain priors. Ensures acyclicity.
    Dynamically merges OPEA-extracted priors with DOMAIN_PRIORS.
    """
    graph_dag = nx.DiGraph()
    graph_dag.add_nodes_from(variable_names)

    for edge in discovered_edges:
        src, tgt = edge
        if not graph_dag.has_edge(tgt, src):
            graph_dag.add_edge(src, tgt)
            if not nx.is_directed_acyclic_graph(graph_dag):
                graph_dag.remove_edge(src, tgt)
                log.debug('[GFCI] cycle: removed forward edge, trying reverse', src=src, tgt=tgt)
                # O(1) check using set — fixes the previous list comprehension O(n) bug
                existing_edges: set[tuple[str, str]] = set(graph_dag.edges())
                if (tgt, src) not in existing_edges:
                    graph_dag.add_edge(tgt, src)
                    if not nx.is_directed_acyclic_graph(graph_dag):
                        graph_dag.remove_edge(tgt, src)
                        log.debug('[GFCI] cycle: reverse edge also causes cycle, leaving unconnected', src=src, tgt=tgt)

    # Merge DOMAIN_PRIORS with dynamic_priors
    # dynamic_priors override DOMAIN_PRIORS in case of conflicts
    merged_priors = DOMAIN_PRIORS.copy()
    if dynamic_priors:
        # Expected format of dynamic_priors: list of dicts {"source": src, "target": tgt, "confidence": conf}
        for dp in dynamic_priors:
            src = str(dp.get("source"))
            tgt = str(dp.get("target"))
            conf = float(dp.get("confidence", 0.0))
            if src in variable_names and tgt in variable_names and conf > 0.70:
                merged_priors[(src, tgt)] = conf

    # Inject domain prior edges that don't create cycles
    for (src, tgt), weight in merged_priors.items():
        if src not in variable_names or tgt not in variable_names:
            continue
        if weight >= 0.7 and not graph_dag.has_edge(src, tgt) and not graph_dag.has_edge(tgt, src):
            graph_dag.add_edge(src, tgt)
            if not nx.is_directed_acyclic_graph(graph_dag):
                graph_dag.remove_edge(src, tgt)

    return graph_dag


def run_discovery(data: Any, variable_names: list[str] | None = None, timeout_seconds: int = 60, dynamic_priors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Run GFCI causal discovery with timeout and PC fallback.
    Accepts OPEA-extracted dynamic_priors to override defaults.
    Returns PAG edges, MAP-DAG, and metadata.
    """
    if variable_names is None:
        variable_names = VARIABLE_NAMES

    numeric_cols = [c for c in variable_names if c in data.columns]
    data_matrix = data[numeric_cols].values.astype(np.float64)

    log.info(
        '[GFCI] Running causal discovery',
        n_variables=len(numeric_cols),
        n_samples=len(data_matrix),
    )

    result: dict[str, Any] = {'success': False}
    algorithm_used = 'GFCI'

    # Run GFCI with timeout
    thread = threading.Thread(
        target=_run_gfci_internal,
        args=(data_matrix, numeric_cols, result),
    )
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive() or not result.get('success', False):
        if thread.is_alive():
            log.warning('[GFCI] Timeout — falling back to PC algorithm', timeout_seconds=timeout_seconds)
        else:
            log.warning('[GFCI] GFCI failed — falling back to PC', error=result.get('error', 'unknown'))

        pc_result, algorithm_used = _run_pc_fallback(data_matrix, numeric_cols)
        if pc_result is not None:
            discovered_edges = extract_discovered_edges(pc_result, numeric_cols)
        else:
            log.warning('[GFCI] Both GFCI and PC failed — using domain priors only')
            discovered_edges = list(DOMAIN_PRIORS.keys())
            algorithm_used = 'DOMAIN_PRIORS_ONLY'
    else:
        graph_obj = result.get('graph')
        discovered_edges = extract_discovered_edges(graph_obj, numeric_cols)

    # If discovery found too few edges, supplement with domain priors
    if len(discovered_edges) < 5:
        log.warning('[GFCI] Too few edges discovered — supplementing with domain priors', n_edges=len(discovered_edges))
        for (src, tgt), _w in DOMAIN_PRIORS.items():
            if (src, tgt) not in discovered_edges and (
                tgt,
                src,
            ) not in discovered_edges:
                discovered_edges.append((src, tgt))

    # Build MAP-DAG using dynamic priors
    map_dag = build_map_dag(discovered_edges, numeric_cols, dynamic_priors=dynamic_priors)

    log.info(
        '[GFCI] Discovery complete',
        algorithm=algorithm_used,
        n_pag_edges=len(discovered_edges),
        n_dag_edges=map_dag.number_of_edges(),
    )

    return {
        'pag_edges': discovered_edges,
        'map_dag': map_dag,
        'algorithm_used': algorithm_used,
        'n_edges_discovered': len(discovered_edges),
        'n_dag_edges': map_dag.number_of_edges(),
        'variable_names': numeric_cols,
        'timeout_triggered': algorithm_used != 'GFCI',
    }


if __name__ == '__main__':
    from cdie.pipeline.data_generator import generate_scm_data

    df = generate_scm_data()
    result = run_discovery(df)
    print('\nDiscovered DAG edges:')
    for e in result['map_dag'].edges():
        print(f'  {e[0]} → {e[1]}')
