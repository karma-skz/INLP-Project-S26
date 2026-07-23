import unittest
from src.analysis.sae_tracker import SAETracker

class TestSAEIntegration(unittest.TestCase):
    def test_feature_extraction(self):
        tracker = SAETracker(sae_model_paths={})
        hidden_states = {0: [0.1, 0.2, 0.3], 1: [0.4, 0.5, 0.6]}
        features = tracker.extract_features(hidden_states)
        
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0], [0.1, 0.2, 0.3])
        self.assertEqual(features[1], [0.4, 0.5, 0.6])

if __name__ == '__main__':
    unittest.main()
