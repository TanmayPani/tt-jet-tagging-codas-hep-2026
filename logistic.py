from functools import partial
import numpy as np

import torch
from torch import nn

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline

from skorch import NeuralNetClassifier


def _device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


class Logistic(nn.Module):
    """Multinomial logistic regression as a Module — returns raw class **logits** (no
    terminal sigmoid/softmax) so it drops straight into a CrossEntropy/focal-loss +
    softmax training loop (`training.torch_fit_predict`). For `n_outputs` classes this is
    softmax regression; recover a probability with softmax (multiclass) at inference."""

    def __init__(self, n_outputs):
        super().__init__()
        self.linear = torch.nn.LazyLinear(n_outputs)

    def forward(self, x):
        return self.linear(x)


def _softmax_np(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class LogitsNetClassifier(NeuralNetClassifier):
    """skorch classifier for a logits-output `nn.Module`: `predict_proba` softmaxes the
    raw logits so the net's probabilities feed the sklearn CV harness and ROC/PR metrics
    (the stock classifier assumes a log-prob output and would mis-scale them)."""

    def predict_proba(self, X):
        return _softmax_np(super().predict_proba(X))


class LogisticRegression(LogitsNetClassifier):
    """skorch net that reproduces sklearn `LogisticRegression(penalty='l2', C,
    class_weight='balanced')` — same objective, no hand-written loop:

      * balanced-class-weighted cross-entropy (`fit` sets `criterion__weight` per fold to
        `n / (n_classes * count_c)`, summed to match sklearn's Σᵢ);
      * L2 on the **weight matrix only** (intercept unpenalized), added in `get_loss` and
        scaled to sklearn's `0.5‖W‖² + C·Σᵢ sᵢ·CEᵢ`;
      * minimized by **LBFGS** (skorch drives the closure), sklearn's default solver.

    The module is the 2-output softmax `TorchLogReg`, so this matches sklearn's
    *multinomial* form — identical probabilities to the binary default, up to the L2 gauge."""

    def __init__(self, *args, C=0.5, **kwargs):
        super().__init__(Logistic, *args, **kwargs)
        self.C = C

    def fit(self, X, y, **fit_params):
        counts = np.bincount(np.asarray(y))
        cw = len(y) / (len(counts) * counts)  # class_weight='balanced'
        self.set_params(
            criterion__weight=torch.as_tensor(
                cw, dtype=torch.float32, device=self.device
            )
        )
        return super().fit(X, y, **fit_params)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        ce_sum = super().get_loss(y_pred, y_true, X=X, training=training)
        w = self.module_.linear.weight

        return self.C * ce_sum + 0.5 * (w**2).sum()


def logistic_pipeline(C=0.5):
    """Torch logistic head as an sklearn Pipeline that MIRRORS
    `LogisticRegression(penalty='l2', C, class_weight='balanced').
    Preprocessing: median-impute the engineered features (ROCKET features are already dense →
    `StandardScaler` only) → float32 cast → the skorch net. `n_inputs` is the feature count;
    wire into a `model_fn_dict` with `partial(logreg_fn, n_inputs=X.shape[1])` (as with
    `partial(catboost_fn, ...)`). `C` mirrors sklearn's inverse-regularization (default 0.5;
    the ROCKET head uses 1.0)."""
    _net = LogisticRegression(
        # module__n_inputs=n_inputs,
        module__n_outputs=2,
        criterion=nn.CrossEntropyLoss,
        criterion__reduction="sum",  # Σᵢ, so C·Σ + ½‖W‖² matches sklearn's objective
        optimizer=torch.optim.LBFGS,
        lr=1.0,
        optimizer__max_iter=500,
        optimizer__line_search_fn="strong_wolfe",
        max_epochs=1,  # one LBFGS.step (max_iter internal iters) = solve to convergence
        batch_size=-1,  # full-batch (LBFGS is a batch solver)
        train_split=None,
        device=_device(),
        verbose=0,
        C=C,
    )

    return make_pipeline(
        StandardScaler(),
        FunctionTransformer(partial(np.asarray, dtype=np.float32)),
        _net,
    )
