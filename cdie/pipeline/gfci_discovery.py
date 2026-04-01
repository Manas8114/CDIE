"""
CDIE v4 — GFCI Causal Discovery
Primary causal structure learning using Greedy Fast Causal Inference.
Includes domain-knowledge prior injection and PAG-to-DAG conversion.
"""

import threading
import numpy as np
import networkx as nx

from cdie.pipeline.data_generator import VARIABLE_NAMES


# Domain-knowledge priors for financial fraud (replaces LLM priors)
DOMAIN_PRIORS = {
    ("TransactionVolume", "FraudAttempts"): 0.8,
    ("TransactionVolume", "SystemLoad"): 0.7,
    ("FraudAttempts", "ChargebackVolume"): 0.85,
    ("DetectionPolicyStrictness", "FraudDetectionRate"): 0.9,
    ("FraudDetectionRate", "ChargebackVolume"): 0.75,
    ("RegulatoryPressure", "DetectionPolicyStrictness"): 0.8,
    ("ExternalNewsSignal", "RegulatoryPressure"): 0.7,
    ("ChargebackVolume", "RevenueImpact"): 0.8,
    ("OperationalCost", "LiquidityRisk"): 0.7,
}


def _run_gfci_internal(data_matrix, variable_names, result_container):
    """Run GFCI in a thread for timeout control."""
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import fisherz

        alpha = 0.05 if len(data_matrix) >= 500 else 0.01

        g, edges = fci(data_matrix, fisherz, alpha, verbose=False)
        result_container["graph"] = g
        result_container["edges"] = edges
        result_container["success"] = True
    except Exception as e:
        result_container["error"] = str(e)
        result_container["success"] = False


def _run_pc_fallback(data_matrix, variable_names):
    """PC algorithm fallback when GFCI times out."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        cg = pc(data_matrix, alpha=0.05, indep_test=fisherz)
        return cg, "PC"
    except Exception as e:
        return None, f"PC_FAILED: {e}"


def extract_discovered_edges(graph_result, variable_names):
    """Extract edges from causal-learn graph result into a list of tuples."""
    discovered_edges = []
    adj_matrix = None

    if hasattr(graph_result, "graph"):
        adj_matrix = graph_result.graph
    elif hasattr(graph_result, "G") and hasattr(graph_result.G, "graph"):
        adj_matrix = graph_result.G.graph

    if adj_matrix is not None:
        n = min(len(variable_names), adj_matrix.shape[0])
        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] != 0:
                    if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                        discovered_edges.append((variable_names[i], variable_names[j]))
                    elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
                        if i < j:
                            discovered_edges.append(
                                (variable_names[i], variable_names[j])
                            )

    return discovered_edges


def build_map_dag(discovered_edges, variable_names, dynamic_priors=None):
    """
    Convert discovered edges to a DAG by resolving uncertain orientations
    using domain priors. Ensures acyclicity.
    Dynamically merges OPEA-extracted priors with DOMAIN_PRIORS.
    """
    G = nx.DiGraph()
    G.add_nodes_from(variable_names)

    for edge in discovered_edges:
        src, tgt = edge
        if not G.has_edge(tgt, src):
            G.add_edge(src, tgt)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(src, tgt)
                if (tgt, src) not in [(e[0], e[1]) for e in G.edges()]:
                    G.add_edge(tgt, src)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(tgt, src)

    # Merge DOMAIN_PRIORS with dynamic_priors
    # dynamic_priors override DOMAIN_PRIORS in case of conflicts
    merged_priors = DOMAIN_PRIORS.copy()
    if dynamic_priors:
        # Expected format of dynamic_priors: list of dicts {"source": src, "target": tgt, "confidence": conf}
        for dp in dynamic_priors:
            src = dp.get("source")
            tgt = dp.get("target")
            conf = dp.get("confidence", 0.0)
            if src and tgt and conf > 0.70:
                merged_priors[(src, tgt)] = conf

    # Inject domain prior edges that don't create cycles
    for (src, tgt), weight in merged_priors.items():
        if weight >= 0.7 and not G.has_edge(src, tgt) and not G.has_edge(tgt, src):
            G.add_edge(src, tgt)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(src, tgt)

    return G


def run_discovery(data, variable_names=None, timeout_seconds=60, dynamic_priors=None):
    """
    Run GFCI causal discovery with timeout and PC fallback.
    Accepts OPEA-extracted dynamic_priors to override defaults.
    Returns PAG edges, MAP-DAG, and metadata.
    """
    if variable_names is None:
        variable_names = VARIABLE_NAMES

    numeric_cols = [c for c in variable_names if c in data.columns]
    data_matrix = data[numeric_cols].values.astype(np.float64)

    print(
        f"[GFCI] Running causal discovery on {len(numeric_cols)} variables, {len(data_matrix)} samples..."
    )

    result = {"success": False}
    algorithm_used = "GFCI"

    # Run GFCI with timeout
    thread = threading.Thread(
        target=_run_gfci_internal,
        args=(data_matrix, numeric_cols, result),
    )
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive() or not result.get("success", False):
        if thread.is_alive():
            print(
                f"[GFCI] Timeout after {timeout_seconds}s — falling back to PC algorithm"
            )
        else:
            print(
                f"[GFCI] GFCI failed: {result.get('error', 'unknown')} — falling back to PC"
            )

        pc_result, algorithm_used = _run_pc_fallback(data_matrix, numeric_cols)
        if pc_result is not None:
            discovered_edges = extract_discovered_edges(pc_result, numeric_cols)
        else:
            print("[GFCI] Both GFCI and PC failed — using domain priors only")
            discovered_edges = list(DOMAIN_PRIORS.keys())
            algorithm_used = "DOMAIN_PRIORS_ONLY"
    else:
        graph_obj = result.get("graph")
        discovered_edges = extract_discovered_edges(graph_obj, numeric_cols)

    # If discovery found too few edges, supplement with domain priors
    if len(discovered_edges) < 5:
        print(
            f"[GFCI] Only {len(discovered_edges)} edges found — supplementing with domain priors"
        )
        for (src, tgt), w in DOMAIN_PRIORS.items():
            if (src, tgt) not in discovered_edges and (
                tgt,
                src,
            ) not in discovered_edges:
                discovered_edges.append((src, tgt))

    # Build MAP-DAG using dynamic priors
    map_dag = build_map_dag(
        discovered_edges, numeric_cols, dynamic_priors=dynamic_priors
    )

    print(
        f"[GFCI] Discovery complete. Algorithm: {algorithm_used}. Edges: {len(discovered_edges)}. DAG edges: {map_dag.number_of_edges()}"
    )

    return {
        "pag_edges": discovered_edges,
        "map_dag": map_dag,
        "algorithm_used": algorithm_used,
        "n_edges_discovered": len(discovered_edges),
        "n_dag_edges": map_dag.number_of_edges(),
        "variable_names": numeric_cols,
        "timeout_triggered": algorithm_used != "GFCI",
    }


if __name__ == "__main__":
    from cdie.pipeline.data_generator import generate_scm_data

    df = generate_scm_data()
    result = run_discovery(df)
    print("\nDiscovered DAG edges:")
    for e in result["map_dag"].edges():
        print(f"  {e[0]} → {e[1]}")
