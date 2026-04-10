import unittest

import torch

from evch.models.gaussian_nll import GaussianNLLRegressor
from evch.models.quantile import QuantileRegressor


class LossesTest(unittest.TestCase):
    def test_gaussian_nll_loss_is_finite(self) -> None:
        model = GaussianNLLRegressor(input_dim=4, hidden_dims=[8])
        inputs = torch.randn(5, 4)
        targets = torch.randn(5, 1)
        mu, log_sigma = model(inputs)
        loss = model.gaussian_nll(mu, log_sigma, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), -10.0)

    def test_quantile_loss_is_non_negative(self) -> None:
        model = QuantileRegressor(input_dim=4, hidden_dims=[8], quantiles=[0.05, 0.5, 0.95])
        inputs = torch.randn(6, 4)
        targets = torch.randn(6, 1)
        predictions = model(inputs)
        loss = model.quantile_loss(predictions, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
