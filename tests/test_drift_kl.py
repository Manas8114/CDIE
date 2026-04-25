import unittest
import numpy as np
from cdie.api.lookup import SafetyMapLookup

class TestDriftKL(unittest.TestCase):
    def test_kl_divergence_computation(self):
        lookup = SafetyMapLookup()
        
        # Mock query history and training distribution
        source = "FraudAttempts"
        lookup.query_history[source].extend([1.0, 1.1, 1.2, 0.9, 1.0, 1.1, 1.0, 0.9, 1.1, 1.0])
        
        # Mock training distribution in json_store or mock the _get_store_val
        lookup.loaded = True
        lookup.json_store = {
            'training_distributions': {
                source: {
                    'sample_values': [10.0, 10.1, 10.2, 9.9, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0] # Drastically different
                }
            }
        }
        
        result = lookup.check_staleness(source, 1.0)
        print(f"Drift Result: {result}")
        
        self.assertTrue(result['drift_detected'])
        self.assertGreater(result['kl_divergence'], 0.5)
        self.assertIn('kl_divergence', result)

    def test_no_drift(self):
        lookup = SafetyMapLookup()
        source = "FraudAttempts"
        vals = [1.0, 1.1, 1.2, 0.9, 1.0, 1.1, 1.0, 0.9, 1.1, 1.0]
        lookup.query_history[source].extend(vals)
        lookup.loaded = True
        lookup.json_store = {
            'training_distributions': {
                source: {
                    'sample_values': vals
                }
            }
        }
        
        result = lookup.check_staleness(source, 1.0)
        print(f"No Drift Result: {result}")
        self.assertFalse(result['drift_detected'])
        self.assertLess(result['kl_divergence'], 0.1)

if __name__ == '__main__':
    unittest.main()
