"""
CDIE v5 — Intent Parser Constants
"""

from cdie.pipeline.data_generator import GROUND_TRUTH_EDGES, VARIABLE_NAMES

# Simplified variable names for matching
VARIABLE_ALIASES = {
    # Telecom mapping
    'sim box': 'SIMBoxFraudAttempts',
    'sim box fraud': 'SIMBoxFraudAttempts',
    'sim fraud': 'SIMBoxFraudAttempts',
    'cdr': 'CallDataRecordVolume',
    'call data': 'CallDataRecordVolume',
    'call records': 'CallDataRecordVolume',
    'arpu': 'ARPUImpact',
    'arpu impact': 'ARPUImpact',
    'retention': 'SubscriberRetentionScore',
    'subscriber retention': 'SubscriberRetentionScore',
    'subscriber score': 'SubscriberRetentionScore',
    'network opex': 'NetworkOpExCost',
    'opex': 'NetworkOpExCost',
    'operating cost': 'NetworkOpExCost',
    'cash flow': 'CashFlowRisk',
    'cashflow': 'CashFlowRisk',
    'regulatory signal': 'RegulatorySignal',
    'itu pressure': 'ITURegulatoryPressure',
    'fraud attempts': 'SIMBoxFraudAttempts',
    'fraud prob': 'SIMBoxFraudAttempts',
    'fraud': 'SIMBoxFraudAttempts',
    'fraud policy': 'FraudPolicyStrictness',
    'detection policy': 'FraudPolicyStrictness',
    'policy strictness': 'FraudPolicyStrictness',
    'policy increase': 'FraudPolicyStrictness',
    'increase policy': 'FraudPolicyStrictness',
    'strictness': 'FraudPolicyStrictness',
    'detection rate': 'SIMFraudDetectionRate',
    'fraud detection': 'SIMFraudDetectionRate',
    'detection': 'SIMFraudDetectionRate',
    'revenue impact': 'ARPUImpact',
    'revenue leakage': 'RevenueLeakageVolume',
    'leakage': 'RevenueLeakageVolume',
    'revenue': 'RevenueLeakageVolume',
    'cost': 'NetworkOpExCost',
    'network load': 'NetworkLoad',
    'load': 'NetworkLoad',
    'regulatory pressure': 'ITURegulatoryPressure',
    'regulation': 'RegulatorySignal',
    'regulatory': 'RegulatorySignal',
    'itu': 'ITURegulatoryPressure',
}

# Add exact variable names to aliases
for v in VARIABLE_NAMES:
    VARIABLE_ALIASES[v.lower()] = v

DEFAULT_TARGETS = {
    'CallDataRecordVolume': 'NetworkLoad',
    'SIMBoxFraudAttempts': 'RevenueLeakageVolume',
    'FraudPolicyStrictness': 'SIMFraudDetectionRate',
    'SIMFraudDetectionRate': 'RevenueLeakageVolume',
    'RevenueLeakageVolume': 'ARPUImpact',
    'SubscriberRetentionScore': 'ARPUImpact',
    'NetworkLoad': 'NetworkOpExCost',
    'NetworkOpExCost': 'CashFlowRisk',
    'CashFlowRisk': 'ARPUImpact',
    'RegulatorySignal': 'FraudPolicyStrictness',
    'ITURegulatoryPressure': 'FraudPolicyStrictness',
}

VARIABLE_DESCRIPTIONS = {
    'CallDataRecordVolume': 'Volume of call detail records processed across the network.',
    'SIMBoxFraudAttempts': 'Observed or inferred SIM-box fraud attempt pressure.',
    'FraudPolicyStrictness': 'How aggressively fraud controls and blocking policies are enforced.',
    'SIMFraudDetectionRate': 'Rate at which SIM-box or fraud activity is detected.',
    'RevenueLeakageVolume': 'Estimated revenue lost due to bypass or fraud leakage.',
    'SubscriberRetentionScore': 'Retention health of subscribers after fraud-control actions.',
    'ARPUImpact': 'Impact on average revenue per user.',
    'NetworkOpExCost': 'Operational cost burden caused by network and fraud-control actions.',
    'CashFlowRisk': 'Cash-flow exposure caused by fraud or cost escalation.',
    'NetworkLoad': 'Load or pressure on the network infrastructure.',
    'RegulatorySignal': 'External regulatory activity or warning signal.',
    'ITURegulatoryPressure': 'ITU-related regulatory pressure affecting fraud policy decisions.',
}

DOWNSTREAM_GRAPH: dict[str, list[str]] = {}
for src, tgt in GROUND_TRUTH_EDGES:
    DOWNSTREAM_GRAPH.setdefault(src, []).append(tgt)

INTERVENTION_PATTERNS = [
    r'what\s+(?:happens|if|would happen)\s+(?:if|when)',
    r'(?:increase|decrease|raise|lower|change|reduce|boost|double|triple)\s+\w+',
    r'(?:impact|effect)\s+of\s+(?:increasing|decreasing|changing|reducing)',
    r'how\s+(?:does|would|will)\s+(?:increasing|decreasing|changing)',
    r'simulate',
    r'\d+%\s+(?:increase|decrease|change|rise|drop|reduction)',
]

COUNTERFACTUAL_PATTERNS = [
    r'what\s+would\s+have\s+happened',
    r'if\s+we\s+had\s+(?:not|never)',
    r'had\s+we\s+(?:not|instead)',
    r'counterfactual',
    r'what\s+if\s+.*\s+(?:had|were|was)\s+(?:not|different|lower|higher)',
]

ROOT_CAUSE_PATTERNS = [
    r'why\s+(?:did|does|is|has|was)',
    r'what\s+(?:causes|caused|drives|drove)',
    r'root\s+cause',
    r'reason\s+(?:for|behind|why)',
    r'explain\s+(?:the|why|how)',
    r'what\s+led\s+to',
]

TEMPORAL_PATTERNS = [
    r'when\s+does\s+\w+\s+affect',
    r'(?:time|temporal)\s+(?:lag|delay|effect)',
    r'how\s+(?:long|soon|quickly)',
    r'lagged?\s+effect',
    r'over\s+time',
    r'delayed?\s+impact',
]

# Pre-defined query presets for demo — Telecom Domain
DEMO_QUERIES = {
    'What happens if fraud attempts increase by 30%?': {
        'source': 'SIMBoxFraudAttempts',
        'target': 'RevenueLeakageVolume',
        'value': 30.0,
    },
    'What if we increase detection policy strictness by 20%?': {
        'source': 'FraudPolicyStrictness',
        'target': 'SIMFraudDetectionRate',
        'value': 20.0,
    },
    'Why did chargeback volume increase?': {
        'source': 'SIMBoxFraudAttempts',
        'target': 'RevenueLeakageVolume',
        'value': 10.0,
    },
    'Show me the temporal impact of fraud on trust.': {
        'source': 'SIMBoxFraudAttempts',
        'target': 'SubscriberRetentionScore',
        'value': -5.0,
    },
}
