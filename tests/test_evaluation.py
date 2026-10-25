import unittest
import numpy as np
from fuentes.evaluation import metrics,choose_threshold
class TestMetrics(unittest.TestCase):
    def test_confusion(self):
        m=metrics([0,0,1,1],[0.2,0.8,0.9,0.1],.5)
        self.assertEqual((m["tn"],m["fp"],m["fn"],m["tp"]),(1,1,1,1))
        self.assertAlmostEqual(m["f1"],.5)
    def test_threshold_uses_only_input(self):
        m,curves,feasible=choose_threshold(
            np.array([0,0,1,1,1,0]),np.array([.1,.2,.4,.6,.8,.95]),.2)
        self.assertTrue(.05<=m["threshold"]<=.85)
        self.assertEqual(len(curves),81)
if __name__=="__main__":
    unittest.main()
