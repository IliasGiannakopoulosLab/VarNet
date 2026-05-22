import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.optimize import brentq
from scipy.stats import binom
from torch.nn.modules.module import _IncompatibleKeys

from utilities.functions import center_crop, center_crop_to_smallest
from utilities.losses import PinballLoss


class UncertaintyModule3D(pl.LightningModule):
    """
    Separate Lightning module for 3D / multislice uncertainty training/testing.

    Design:
        - loads or receives a pretrained 3D VarNet model
        - freezes the 3D VarNet permanently
        - trains only the 3D uncertainty network
        - calibrates lambda using volume-level misses
        - appends uncertainty_map to pre-existing 3D VarNet H5 reconstructions
        - keeps the same checkpoint strategy as the 2D uncertainty module:
          only uncertainty weights are stored in the uncertainty checkpoint

    Assumptions:
        - varnet_model returns a reconstruction volume [B, D, H, W]
        - uncertainty_model takes recon [B, D, H, W] and returns
              lower_pred [B, D, H, W], upper_pred [B, D, H, W]
        - batch comes from VolumeVarNetDataTransform
    """

    def __init__(
        self,
        varnet_model: torch.nn.Module,
        uncertainty_model: torch.nn.Module,
        learnable_mask: bool = False,
        model_name: str = "default_uncertainty_model_3d",
        # -------- varnet checkpoint loading --------
        varnet_ckpt: Optional[Path] = None,
        varnet_ckpt_prefix: str = "varnet.",
        strict_varnet_load: bool = False,
        # -------- optimizer --------
        lr: float = 3e-4,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        max_steps: int = 65450,
        ramp_steps: int = 2618,
        cosine_decay_start: int = 32725,
        # -------- quantile loss --------
        alpha: float = 0.1,
        # -------- calibration --------
        calibration_delta: float = 0.1,
        calibration_lambdas_start: float = 1.0,
        calibration_lambdas_end: float = 2.0,
        calibration_lambdas_steps: int = 20,
        calibration_sample_rate: float = 1.0,
        # -------- test append --------
        reconstructions_dir: Optional[Path] = None,
    ):
        super().__init__()

        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if calibration_lambdas_steps < 2:
            raise ValueError("calibration_lambdas_steps must be >= 2")

        self.varnet_model = varnet_model
        self.uncertainty_model = uncertainty_model
        self.learnable_mask = learnable_mask
        self.model_name = model_name

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.ramp_steps = ramp_steps
        self.cosine_decay_start = cosine_decay_start

        self.alpha = alpha
        self.lower_pinball = PinballLoss(quantile=alpha / 2.0)
        self.upper_pinball = PinballLoss(quantile=1.0 - alpha / 2.0)

        self.calibration_delta = calibration_delta
        self.calibration_lambdas_start = calibration_lambdas_start
        self.calibration_lambdas_end = calibration_lambdas_end
        self.calibration_lambdas_steps = calibration_lambdas_steps
        self.calibration_sample_rate = calibration_sample_rate

        self.calibration_lambda: Optional[float] = None
        self.reconstructions_dir = (
            Path(reconstructions_dir) if reconstructions_dir is not None else None
        )

        if varnet_ckpt is not None:
            self._load_varnet_checkpoint(
                ckpt_path=Path(varnet_ckpt),
                prefix=varnet_ckpt_prefix,
                strict=strict_varnet_load,
            )

        self._freeze_varnet()

    # ============================================================
    # CHECKPOINT HELPERS
    # ============================================================
    def _prefixed_uncertainty_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            f"uncertainty_model.{k}": v
            for k, v in self.uncertainty_model.state_dict().items()
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Keep full Lightning trainer state, but store only uncertainty weights.
        The frozen VarNet is loaded separately from its own checkpoint.
        """
        checkpoint["state_dict"] = self._prefixed_uncertainty_state_dict()
        checkpoint["calibration_lambda"] = self.calibration_lambda
        checkpoint["alpha"] = self.alpha
        checkpoint["model_name"] = self.model_name

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "calibration_lambda" in checkpoint:
            lam = checkpoint["calibration_lambda"]
            self.calibration_lambda = None if lam is None else float(lam)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """
        Allow Lightning resume checkpoints containing only uncertainty_model.*
        weights while ignoring frozen VarNet keys.
        """
        incompatible = super().load_state_dict(state_dict, strict=False)

        missing = [
            k for k in incompatible.missing_keys
            if not k.startswith("varnet_model.")
        ]
        unexpected = list(incompatible.unexpected_keys)

        if strict and (missing or unexpected):
            raise RuntimeError(
                "Error(s) in loading state_dict for UncertaintyModule3D:\n"
                f"  Missing key(s): {missing}\n"
                f"  Unexpected key(s): {unexpected}"
            )

        return _IncompatibleKeys(missing, unexpected)

    # ============================================================
    # INIT HELPERS
    # ============================================================
    def _move_batch_to_device(self, batch, device):
        """
        Move only tensor fields of the NamedTuple batch to the requested device.
        """
        updates = {}

        for field in batch._fields:
            value = getattr(batch, field)
            if torch.is_tensor(value):
                updates[field] = value.to(device)

        return batch._replace(**updates)

    def _load_varnet_checkpoint(
        self,
        ckpt_path: Path,
        prefix: str = "varnet.",
        strict: bool = False,
    ):
        """
        Load reconstruction weights into the frozen 3D VarNet model.

        `prefix` should usually match the module attribute used in the reconstruction
        Lightning checkpoint, e.g. "varnet." for VarNetModule3D or "fi_varnet."
        if using the compatibility alias.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        if prefix:
            filtered = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        else:
            filtered = state_dict

        if len(filtered) == 0 and prefix == "varnet.":
            # Compatibility fallback for checkpoints saved through the alias used
            # by the reconstruction runner helper functions.
            alt_prefix = "fi_varnet."
            filtered = {
                k[len(alt_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(alt_prefix)
            }

        missing, unexpected = self.varnet_model.load_state_dict(filtered, strict=strict)

        print(f"[UncertaintyModule3D] loaded VarNet checkpoint from {ckpt_path}")
        print(f"[UncertaintyModule3D] missing VarNet keys: {len(missing)}")
        print(f"[UncertaintyModule3D] unexpected VarNet keys: {len(unexpected)}")

    def _freeze_varnet(self):
        for p in self.varnet_model.parameters():
            p.requires_grad = False
        self.varnet_model.eval()

    # ============================================================
    # SMALL SHAPE HELPERS
    # ============================================================
    @staticmethod
    def _first_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.view(-1)[0].item())
        return int(value)

    @staticmethod
    def _get_fname(fname, index: int):
        if isinstance(fname, (list, tuple)):
            return fname[index]
        return fname

    def _resolve_crop_size(self, batch, recon: torch.Tensor) -> Tuple[int, int]:
        """
        Resolve crop size for batched 3D recon [B, D, H, W].
        DataLoader may collate crop_size=(H, W) into [tensor([H]), tensor([W])].
        """
        crop_size = batch.crop_size

        if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
            crop_h = self._first_int(crop_size[0])
            crop_w = self._first_int(crop_size[1])
        elif torch.is_tensor(crop_size) and crop_size.numel() >= 2:
            flat = crop_size.view(-1)
            crop_h = int(flat[0].item())
            crop_w = int(flat[1].item())
        else:
            crop_h = recon.shape[-2]
            crop_w = recon.shape[-1]

        crop_h = min(crop_h, recon.shape[-2])
        crop_w = min(crop_w, recon.shape[-1])

        return crop_h, crop_w

    # ============================================================
    # MODEL FORWARD
    # ============================================================
    def _run_varnet(self, batch) -> torch.Tensor:
        """
        Reuse the reconstruction routing logic:
            fixed-mask:     masked_kspace + mask
            learnable-mask: full_kspace + acq_start/acq_end
        """
        self.varnet_model.eval()

        if self.learnable_mask:
            recon = self.varnet_model(
                full_kspace=batch.full_kspace,
                acq_start=batch.acq_start,
                acq_end=batch.acq_end,
                crop_size=batch.crop_size,
                return_mask_extras=False,
            )
        else:
            recon = self.varnet_model(
                batch.masked_kspace,
                batch.mask,
                batch.num_low_frequencies,
                crop_size=batch.crop_size,
            )

        if isinstance(recon, tuple):
            raise ValueError(
                "varnet_model returned a tuple. UncertaintyModule3D expects the "
                "VarNet path to return only the reconstruction volume."
            )
        if recon.ndim != 4:
            raise ValueError(
                f"Expected 3D reconstruction with shape [B, D, H, W], got {recon.shape}"
            )

        return recon

    def forward(self, batch):
        with torch.no_grad():
            recon = self._run_varnet(batch)

        lower_pred, upper_pred = self.uncertainty_model(recon)

        if lower_pred.shape != recon.shape or upper_pred.shape != recon.shape:
            raise ValueError(
                "uncertainty_model must return lower/upper predictions with the "
                f"same shape as recon. Got recon={recon.shape}, "
                f"lower={lower_pred.shape}, upper={upper_pred.shape}."
            )

        return recon, lower_pred, upper_pred

    # ============================================================
    # LOSS / CROP
    # ============================================================
    def _crop_to_target(
        self,
        target: torch.Tensor,
        recon: torch.Tensor,
        lower_pred: torch.Tensor,
        upper_pred: torch.Tensor,
    ):
        """
        Center-crop only the image-plane dimensions. The slice dimension D is preserved.
        """
        target, recon = center_crop_to_smallest(target, recon)
        _, lower_pred = center_crop_to_smallest(target, lower_pred)
        _, upper_pred = center_crop_to_smallest(target, upper_pred)
        return target, recon, lower_pred, upper_pred

    def _compute_pinball_loss(
        self,
        target: torch.Tensor,
        lower_pred: torch.Tensor,
        upper_pred: torch.Tensor,
    ):
        lower_loss = self.lower_pinball(lower_pred, target)
        upper_loss = self.upper_pinball(upper_pred, target)
        total = lower_loss + upper_loss
        return total, lower_loss, upper_loss

    # ============================================================
    # TRAIN / VAL
    # ============================================================
    def training_step(self, batch, batch_idx):
        recon, lower_pred, upper_pred = self.forward(batch)
        target, recon, lower_pred, upper_pred = self._crop_to_target(
            batch.target,
            recon,
            lower_pred,
            upper_pred,
        )

        loss, lower_loss, upper_loss = self._compute_pinball_loss(
            target,
            lower_pred,
            upper_pred,
        )

        self.log("train/loss", loss.detach(), sync_dist=True)
        self.log("train/pinball", loss.detach(), sync_dist=True)
        self.log("train/pinball_lower", lower_loss.detach(), sync_dist=True)
        self.log("train/pinball_upper", upper_loss.detach(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        recon, lower_pred, upper_pred = self.forward(batch)
        target, recon, lower_pred, upper_pred = self._crop_to_target(
            batch.target,
            recon,
            lower_pred,
            upper_pred,
        )

        loss, lower_loss, upper_loss = self._compute_pinball_loss(
            target,
            lower_pred,
            upper_pred,
        )

        return {
            "val_loss": loss.detach(),
            "pinball": loss.detach(),
            "pinball_lower": lower_loss.detach(),
            "pinball_upper": upper_loss.detach(),
        }

    def validation_epoch_end(self, val_logs):
        if len(val_logs) == 0:
            return

        val_loss = torch.mean(torch.cat([x["val_loss"].view(-1) for x in val_logs]))
        pinball = torch.mean(torch.cat([x["pinball"].view(-1) for x in val_logs]))
        pinball_lower = torch.mean(
            torch.cat([x["pinball_lower"].view(-1) for x in val_logs])
        )
        pinball_upper = torch.mean(
            torch.cat([x["pinball_upper"].view(-1) for x in val_logs])
        )

        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/pinball", pinball, prog_bar=True, sync_dist=True)
        self.log("val/pinball_lower", pinball_lower, sync_dist=True)
        self.log("val/pinball_upper", pinball_upper, sync_dist=True)

    # ============================================================
    # CALIBRATION
    # ============================================================
    @staticmethod
    def _fraction_missed_loss(
        lower: torch.Tensor,
        upper: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns one miss fraction per volume: [B].

        Do not squeeze here. For B=1, squeeze would remove the batch dimension
        and accidentally average per-slice or per-row instead of per-volume.
        """
        if lower.shape != upper.shape or lower.shape != target.shape:
            raise ValueError(
                f"lower, upper, target must match. Got {lower.shape}, "
                f"{upper.shape}, {target.shape}."
            )
        if target.ndim != 4:
            raise ValueError(
                f"Expected 3D target with shape [B, D, H, W], got {target.shape}"
            )

        misses = (lower > target).float()
        misses += (upper < target).float()
        misses = torch.clamp(misses, 0.0, 1.0)

        return misses.mean(dim=(1, 2, 3))

    @staticmethod
    def _h1(y, mu):
        return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))

    @classmethod
    def _hoeffding_plus(cls, mu, x, n):
        return -n * cls._h1(np.minimum(mu, x), mu)

    @staticmethod
    def _bentkus_plus(mu, x, n):
        return np.log(max(binom.cdf(np.floor(n * x), n, mu), 1e-10)) + 1

    @classmethod
    def _HB_mu_plus(cls, muhat, n, delta, maxiters=1000):
        def _tailprob(mu):
            hoeffding_mu = cls._hoeffding_plus(mu, muhat, n)
            bentkus_mu = cls._bentkus_plus(mu, muhat, n)
            return min(hoeffding_mu, bentkus_mu) - np.log(delta)

        if _tailprob(1 - 1e-10) > 0:
            return 1.0

        try:
            return brentq(_tailprob, muhat, 1 - 1e-10, maxiter=maxiters)
        except Exception:
            return 1.0

    def _apply_lambda(
        self,
        recon: torch.Tensor,
        lower_pred: torch.Tensor,
        upper_pred: torch.Tensor,
        lam: float,
    ):
        lower_pred = torch.minimum(lower_pred, recon)
        upper_pred = torch.maximum(upper_pred, recon)

        lower_cal = torch.clamp(recon - lam * (recon - lower_pred), min=0.0)
        upper_cal = recon + lam * (upper_pred - recon)

        return lower_cal, upper_cal

    def get_calibration_losses_from_dataloader(
        self,
        dataloader,
        lam: float,
        device: Optional[str] = None,
        keep_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.eval()
        self.to(device)

        losses = []
        batch_counter = 0

        with torch.no_grad():
            for batch in dataloader:
                if keep_indices is not None and batch_counter not in keep_indices:
                    batch_counter += 1
                    continue

                batch = self._move_batch_to_device(batch, device)

                recon, lower_pred, upper_pred = self.forward(batch)
                target = batch.target

                target, recon, lower_pred, upper_pred = self._crop_to_target(
                    target,
                    recon,
                    lower_pred,
                    upper_pred,
                )

                lower_cal, upper_cal = self._apply_lambda(
                    recon,
                    lower_pred,
                    upper_pred,
                    lam,
                )

                loss = self._fraction_missed_loss(lower_cal, upper_cal, target).cpu()
                losses.append(loss)
                batch_counter += 1

        if len(losses) == 0:
            raise RuntimeError("No calibration losses were collected.")

        return torch.cat(losses, dim=0)

    def calibrate(
        self,
        dataloader,
        device: Optional[str] = None,
    ) -> float:
        """
        Run RCPS-style calibration and store lambda inside the module.
        Call this explicitly after training.
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        lambdas = torch.linspace(
            self.calibration_lambdas_start,
            self.calibration_lambdas_end,
            steps=self.calibration_lambdas_steps,
        )
        dlambda = lambdas[1] - lambdas[0]

        n_total = len(dataloader)
        keep_indices = None

        if self.calibration_sample_rate < 1.0:
            n_keep = max(1, int(n_total * self.calibration_sample_rate))
            rng = np.random.RandomState(42)
            keep_indices = set(rng.choice(n_total, n_keep, replace=False).tolist())

        best_lambda = None

        for lam in reversed(lambdas):
            losses = self.get_calibration_losses_from_dataloader(
                dataloader=dataloader,
                lam=float((lam - dlambda).item()),
                device=device,
                keep_indices=keep_indices,
            )

            Rhat = losses.mean().item()
            RhatPlus = self._HB_mu_plus(Rhat, losses.shape[0], self.calibration_delta)

            print(
                f"[Calibration3D] lambda={lam.item():.4f} "
                f"Rhat={Rhat:.4f} RhatPlus={RhatPlus:.4f}"
            )

            if Rhat >= self.alpha or RhatPlus > self.alpha:
                best_lambda = float(lam.item())
                break

        if best_lambda is None:
            raise RuntimeError("Optimal calibration lambda not found in scanned range.")

        self.calibration_lambda = best_lambda
        print(f"[Calibration3D] selected lambda = {self.calibration_lambda:.6f}")
        return self.calibration_lambda

    # ============================================================
    # TEST
    # ============================================================
    def test_step(self, batch, batch_idx):
        if self.calibration_lambda is None:
            raise RuntimeError(
                "calibration_lambda is not set. Run calibrate(...) or load an "
                "uncertainty checkpoint first."
            )

        recon, lower_pred, upper_pred = self.forward(batch)

        crop_size = self._resolve_crop_size(batch, recon)
        recon = center_crop(recon, crop_size)
        lower_pred = center_crop(lower_pred, crop_size)
        upper_pred = center_crop(upper_pred, crop_size)

        lower_cal, upper_cal = self._apply_lambda(
            recon,
            lower_pred,
            upper_pred,
            self.calibration_lambda,
        )
        uncertainty_map = upper_cal - lower_cal

        return {
            "fname": batch.fname,
            "uncertainty_map": uncertainty_map.detach().cpu().numpy(),
        }

    def test_epoch_end(self, test_logs):
        if self.reconstructions_dir is None:
            raise ValueError(
                "reconstructions_dir must be set so 3D uncertainty maps can be "
                "appended to the pre-existing 3D VarNet H5 files."
            )

        for log in test_logs:
            maps = log["uncertainty_map"]
            batch_size = maps.shape[0]

            for i in range(batch_size):
                fname = self._get_fname(log["fname"], i)
                file_path = self.reconstructions_dir / fname

                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Expected pre-existing reconstruction file not found: {file_path}"
                    )

                with h5py.File(file_path, "a") as hf:
                    if "uncertainty_map" in hf:
                        del hf["uncertainty_map"]
                    hf.create_dataset("uncertainty_map", data=maps[i])

        self.print(
            f"Appended 3D uncertainty maps to existing H5 files in {self.reconstructions_dir}"
        )

    # ============================================================
    # CHECKPOINT I/O HELPERS
    # ============================================================
    def get_export_checkpoint(self) -> Dict[str, Any]:
        return {
            "state_dict": self.uncertainty_model.state_dict(),
            "calibration_lambda": self.calibration_lambda,
            "alpha": self.alpha,
            "model_name": self.model_name,
        }

    def save_export_checkpoint(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_export_checkpoint(), path)

    def load_uncertainty_checkpoint(self, path: Path, strict: bool = True):
        """
        Load either:
            - a full Lightning checkpoint with uncertainty_model.* keys, or
            - a slim export checkpoint with only uncertainty_model keys.
        """
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        if any(k.startswith("uncertainty_model.") for k in state_dict.keys()):
            state_dict = {
                k[len("uncertainty_model."):]: v
                for k, v in state_dict.items()
                if k.startswith("uncertainty_model.")
            }

        self.uncertainty_model.load_state_dict(state_dict, strict=strict)

        if "calibration_lambda" in ckpt:
            lam = ckpt["calibration_lambda"]
            self.calibration_lambda = None if lam is None else float(lam)

    def update_checkpoint_file(self, src_path: Path, dst_path: Optional[Path] = None):
        """
        Update an existing checkpoint file with current uncertainty weights and
        calibration lambda. Full Lightning checkpoints keep trainer state;
        slim checkpoints are written back as slim export checkpoints.
        """
        src_path = Path(src_path)
        dst_path = src_path if dst_path is None else Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = torch.load(src_path, map_location="cpu")

        if isinstance(ckpt, dict) and any(
            key in ckpt
            for key in [
                "optimizer_states",
                "lr_schedulers",
                "loops",
                "pytorch-lightning_version",
                "global_step",
                "epoch",
            ]
        ):
            ckpt["state_dict"] = self._prefixed_uncertainty_state_dict()
            ckpt["calibration_lambda"] = self.calibration_lambda
            ckpt["alpha"] = self.alpha
            ckpt["model_name"] = self.model_name
            torch.save(ckpt, dst_path)
        else:
            self.save_export_checkpoint(dst_path)

    # ============================================================
    # OPTIMIZER
    # ============================================================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.uncertainty_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma,
        )

        return [optimizer], [scheduler]

    # ============================================================
    # CLI
    # ============================================================
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--lr", default=3e-4, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--ramp_steps", default=2618, type=int)
        parser.add_argument("--cosine_decay_start", default=32725, type=int)
        parser.add_argument("--lr_step_size", default=40, type=int)
        parser.add_argument("--lr_gamma", default=0.1, type=float)
        parser.add_argument("--alpha", default=0.1, type=float)
        parser.add_argument("--calibration_delta", default=0.1, type=float)
        parser.add_argument("--calibration_lambdas_start", default=1.0, type=float)
        parser.add_argument("--calibration_lambdas_end", default=2.0, type=float)
        parser.add_argument("--calibration_lambdas_steps", default=20, type=int)
        parser.add_argument("--calibration_sample_rate", default=1.0, type=float)

        return parser