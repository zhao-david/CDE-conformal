import numpy as np
import torch
import local_conformal as lc

def test_QuantilePerceptron__loss_fn():
    preds = torch.from_numpy(np.arange(20).reshape((5,4))).float()
    target = torch.from_numpy(np.arange(5).reshape((5,1))).float()
    quantiles = torch.from_numpy(np.arange(1,5)/5)

    model = lc.QuantilePerceptron(10,10, quantiles = quantiles)

    def loss_fn2(quantiles, preds, target):
            """
            Computes the quantile regression loss across predicts from model and
            true values.


            Arguments:
            ----------
            preds: predicted quantiles from model (n, num_quantiles)
            target: potential vector of targe (y) values (n, 1) torch.Tensor

            Details:
            --------
            This loss is from: https://github.com/ceshine/quantile-regression-tensorflow/blob/master/notebooks/03-sklearn-example-pytorch.ipynb


            """
            assert not target.requires_grad
            assert preds.size(0) == target.size(0)
            losses = []
            for i, q in enumerate(quantiles):
                errors = target.reshape(-1) - preds[:, i]
                losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
            loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
            return loss


    model_loss = model.loss_fn(preds, target)
    old_loss = loss_fn2(quantiles, preds, target)

    assert model_loss == old_loss, \
        "losses should return same value"
