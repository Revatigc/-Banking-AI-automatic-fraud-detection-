import unittest

import torch

from src.model import FraudMLP


class FraudMLPTests(unittest.TestCase):
    def test_forward_emits_one_logit_per_row(self):
        self.assertEqual(tuple(FraudMLP(7)(torch.randn(5, 7)).shape), (5,))
