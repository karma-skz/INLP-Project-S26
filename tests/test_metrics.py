import unittest
from src.analysis.cis_metric import CISMetric
from src.analysis.latent_probes import LatentTruthProbe
import numpy as np

class TestMetrics(unittest.TestCase):
    def test_cis_metric(self):
        cis = CISMetric()
        # Test attenuation
        self.assertEqual(cis.compute_cis(10.0, 5.0), 2.0)
        # Test no attenuation
        self.assertEqual(cis.compute_cis(10.0, 10.0), 1.0)
        # Test div by zero protection
        self.assertEqual(cis.compute_cis(10.0, 0.0), float('inf'))

    def test_ltvt_probe(self):
        probe = LatentTruthProbe(d_model=4)
        acts_p = {0: np.array([[1, 1, 1, 1], [1, 1, 1, 1]])}
        acts_not_p = {0: np.array([[0, 0, 0, 0], [0, 0, 0, 0]])}
        probe.train_probes(acts_p, acts_not_p)
        
        test_acts = {0: np.array([0.5, 0.5, 0.5, 0.5])}
        scores = probe.compute_ltvt(test_acts)
        self.assertEqual(scores[0], 2.0) # dot product of [0.5]*4 and [1]*4

if __name__ == '__main__':
    unittest.main()
