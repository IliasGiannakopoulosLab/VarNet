import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.functions import preprocess_for_yolo_batch

# -------------------------------------------#
# ----------- yolo perceptual loss --------- #
# -------------------------------------------#
class YOLOPerceptualLoss(nn.Module):
    def __init__(
        self,
        yolo_model,
        gradcam,
        feature_layers,
        imgsz=(640,640),
        layer_weights=None,
        pl_weight=0.1
    ):
        super().__init__()

        self.yolo = yolo_model.eval()
        self.gradcam = gradcam
        self.feature_layers = feature_layers
        self.imgsz = imgsz
        self.pl_weight = pl_weight

        if layer_weights is None:
            layer_weights = [1.0]*len(feature_layers)
        self.layer_weights = layer_weights

    # -------------------------------------------------
    def forward(self, target, output, return_stats=False):

        tgt = preprocess_for_yolo_batch(target, self.imgsz)
        out = preprocess_for_yolo_batch(output, self.imgsz)

        device = output.device   # always correct in Lightning
        self.yolo = self.yolo.to(device)

        for p in self.yolo.parameters():
            p.requires_grad = False

        tgt = tgt.to(device)
        out = out.to(device)

        with torch.no_grad():
            feat_t = self.yolo(tgt)
        feat_o = self.yolo(out)

        cam = self.gradcam(tgt)

        flat = cam.flatten()
        k = max(int(0.10 * flat.numel()), 1)
        topk_vals, _ = torch.topk(flat, k)
        threshold = topk_vals[-1]

        mask = (cam >= threshold).float()
        cam = mask[None, None]
        #cam = torch.clamp(cam - threshold, min=0.0)
        #cam = cam / (cam.max() + 1e-8)
        #cam = cam ** 0.5
        #cam = cam[None, None]

        # -------------------------------------------------
        total = 0.0
        raw_losses = []
        weighted_losses = []

        for i, (ft, fo) in enumerate(zip(feat_t, feat_o)):

            diff = torch.abs(fo - ft)

            if cam is not None:
                cam_resized = F.interpolate(
                    cam,
                    size=ft.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                cam_resized = (cam_resized > 0.5).float()
                diff = diff * cam_resized

                num_active = cam_resized.sum() * diff.shape[1]
                layer_loss = diff.sum() / (num_active)
                #layer_loss = diff.sum() / (num_active + 1e-8)
            else:
                layer_loss = diff.mean()

            raw_losses.append(layer_loss.detach())

            weighted = self.layer_weights[i] * layer_loss
            weighted_losses.append(weighted.detach())

            total += weighted

        total = total / len(self.feature_layers)
        total = self.pl_weight * total

        if not return_stats:
            return total

        stats = {
            "total": float(total.detach().cpu()),
            "per_layer_raw": [float(v.cpu()) for v in raw_losses],
            "per_layer_weighted": [float(v.cpu()) for v in weighted_losses],
            "layer_ids": list(self.feature_layers),
        }

        return total, stats


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


# -------------------------------------------#
# ---------------- ssim 3d loss ------------ #
# -------------------------------------------#
class SSIM3DLoss(nn.Module):
    """
    Slice-wise 2D SSIM loss for 3D / multislice reconstruction.

    Input:
        X: [B, 1, D, H, W]
        Y: [B, 1, D, H, W]

    Equivalent logic to evaluation.ssim:
        maxval = gt.max()
        for each slice:
            SSIM(slice, data_range=maxval)
        average over slices
    """

    def __init__(self):
        super().__init__()
        self.ssim2d = SSIMLoss()

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        # one max value per 3D ground-truth volume
        maxval = Y.amax(dim=(1, 2, 3, 4))

        ssim_val = 0.0

        for slice_num in range(Y.shape[2]):
            ssim_val = ssim_val + 1.0 - self.ssim2d(
                X[:, :, slice_num, :, :],
                Y[:, :, slice_num, :, :],
                data_range=maxval,
            )
            
        return 1.0 - ssim_val / Y.shape[2]