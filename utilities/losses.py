import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------#
# -------------- pinball loss -------------- #
# -------------------------------------------#
class PinballLoss(nn.Module):
    """
    Quantile regression loss.

    For quantile q in (0, 1):
        L_q(y_pred, y_true) = max(q * (y_true - y_pred),
                                  (q - 1) * (y_true - y_pred))
    """

    def __init__(self, quantile: float, reduction: str = "mean"):
        super().__init__()

        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.quantile = quantile
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
            )

        error = target - pred
        loss = torch.maximum(self.quantile * error, (self.quantile - 1.0) * error)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# -------------------------------------------#
# ----------------- ssim loss -------------- #
# -------------------------------------------#
class SSIMLoss(nn.Module):

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1.0 - S.mean()
        else:
            return 1.0 - S
