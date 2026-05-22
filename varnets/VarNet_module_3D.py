from argparse import ArgumentParser
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

from .Network_module_3D import MriModule3D
from .VarNet import E2EVarNet3D
from utilities.functions import center_crop, center_crop_to_smallest
from utilities.losses import SSIM3DLoss


class VarNetModule3D(MriModule3D):
    """
    Lightning module for 3D / multislice VarNet reconstruction.

    Design assumptions:
        - Input batch is produced by VolumeVarNetDataTransform.
        - k-space shape is [B, coils, D, H, W, 2].
        - fixed mask shape is [B, 1, D, 1, W, 1].
        - target/output image volumes are [B, D, H, W].
        - The base reconstruction model is E2EVarNet3D.
        - Perceptual / YOLO loss is intentionally not used here.
        - Learnable-mask mode uses LearnableMaskedVarNet3D, which receives
          full k-space and internally builds a learned D x ky Cartesian mask.
    """

    def __init__(
        self,
        varnet: E2EVarNet3D,
        learnable_mask: bool = False,
        model_name: str = "default_model_3d",
        lr: float = 3e-4,
        lr_base: Optional[float] = None,
        lr_mask: Optional[float] = None,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        max_epochs: int = 50,
        weight_decay: float = 0.0,
        drop_prob: float = 0.0,
        max_steps: int = 65450,
        ramp_steps: int = 2618,
        cosine_decay_start: int = 32725,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self.lr = lr
        self.lr_base = lr if lr_base is None else lr_base
        self.lr_mask = lr if lr_mask is None else lr_mask
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.ramp_steps = ramp_steps
        self.cosine_decay_start = cosine_decay_start
        self.weight_decay = weight_decay
        self.drop_prob = drop_prob
        self.learnable_mask = learnable_mask

        self.varnet = varnet

        # Compatibility alias: this lets existing runner helper functions that expect
        # module.fi_varnet keep working with minimal changes.
        self.fi_varnet = self.varnet

        self.ssim_loss = SSIM3DLoss()

    # ============================================================
    # HELPERS
    # ============================================================
    @staticmethod
    def _as_batch_tensor(value, device, dtype=torch.float32):
        if torch.is_tensor(value):
            out = value.to(device=device)
        else:
            out = torch.as_tensor(value, device=device)

        if out.ndim == 0:
            out = out.view(1)

        return out.to(dtype=dtype)

    @staticmethod
    def _first_int(value) -> int:
        """
        Robustly extracts an int from values produced by default_collate.
        Handles plain ints, scalar tensors, and one-element batch tensors.
        """
        if torch.is_tensor(value):
            return int(value.view(-1)[0].item())
        return int(value)

    def _resolve_crop_size(self, batch, output: torch.Tensor) -> Tuple[int, int]:
        """
        Resolve crop size for batched 3D output [B, D, H, W].
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
            crop_h = output.shape[-2]
            crop_w = output.shape[-1]

        crop_h = min(crop_h, output.shape[-2])
        crop_w = min(crop_w, output.shape[-1])

        return crop_h, crop_w

    @staticmethod
    def _extract_mask_2d(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Convert fixed 3D Cartesian mask to [B, D, W].

        Expected input from VolumeVarNetDataTransform after batching:
            [B, 1, D, 1, W, 1]
        """
        if mask is None:
            return None

        if mask.ndim == 6:
            return mask[:, 0, :, 0, :, 0].to(torch.uint8)

        if mask.ndim == 5:
            # Unbatched fallback: [1, D, 1, W, 1]
            return mask[None, 0, :, 0, :, 0].to(torch.uint8)

        raise ValueError(f"Unexpected 3D mask shape: {mask.shape}")

    @staticmethod
    def _compute_effective_acceleration(
        mask_2d: Optional[torch.Tensor],
        support_width: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if mask_2d is None:
            return None

        # mask_2d: [B, D, W]
        depth = mask_2d.shape[1]
        sampled_points = mask_2d.sum(dim=(1, 2)).to(torch.float32)
        support_points = support_width.to(torch.float32) * float(depth)

        return support_points / sampled_points.clamp_min(1.0)

    @staticmethod
    def _mask_2d_from_mask_info(mask_info: Optional[dict]) -> Optional[torch.Tensor]:
        if mask_info is None:
            return None
        mask_2d = mask_info.get("hard_mask_2d", None)
        if mask_2d is None:
            return None
        return mask_2d.to(torch.uint8)

    def _get_latest_mask_info(self):
        if not self.learnable_mask:
            return None
        if not hasattr(self.varnet, "latest_mask_info"):
            return None
        return self.varnet.latest_mask_info

    def _log_mask_scalar(self, stage: str, info: dict, key: str, log_name: Optional[str] = None):
        if key not in info:
            return
        value = info[key]
        if not torch.is_tensor(value):
            return
        name = key if log_name is None else log_name
        self.log(f"{stage}/{name}", value.float().mean().detach(), sync_dist=True)

    def _log_learnable_mask_stats(self, stage: str):
        info = self._get_latest_mask_info()
        if info is None:
            return

        self._log_mask_scalar(stage, info, "sampled_points")
        self._log_mask_scalar(stage, info, "sampled_lines")
        self._log_mask_scalar(stage, info, "support_width")
        self._log_mask_scalar(stage, info, "num_slices")
        self._log_mask_scalar(stage, info, "effective_acceleration", "acceleration")
        self._log_mask_scalar(stage, info, "outer_prob_raw_mean", "mask_prob_raw_mean")
        self._log_mask_scalar(stage, info, "outer_prob_mean", "mask_prob_mean")
        self._log_mask_scalar(stage, info, "target_outer_mean")
        self._log_mask_scalar(stage, info, "target_total_ratio")
        self._log_mask_scalar(stage, info, "target_sampled_points")
        self._log_mask_scalar(stage, info, "center_points")
        self._log_mask_scalar(stage, info, "outer_count")

        if "prob_mask_2d" in info and torch.is_tensor(info["prob_mask_2d"]):
            self.log(
                f"{stage}/mask_prob_std",
                info["prob_mask_2d"].float().std(unbiased=False).detach(),
                sync_dist=True,
            )

        if "per_slice_effective_acceleration" in info and torch.is_tensor(info["per_slice_effective_acceleration"]):
            per_slice_acc = info["per_slice_effective_acceleration"].float()
            self.log(f"{stage}/per_slice_acc_mean", per_slice_acc.mean().detach(), sync_dist=True)
            self.log(f"{stage}/per_slice_acc_std", per_slice_acc.std(unbiased=False).detach(), sync_dist=True)

    # ============================================================
    # MODEL FORWARD / LOSS
    # ============================================================
    def forward(self, batch, return_mask_extras: bool = False):
        return self._run_reconstruction_model(batch, return_mask_extras=return_mask_extras)

    def _run_reconstruction_model(self, batch, return_mask_extras: bool = False):
        if self.learnable_mask:
            return self.varnet(
                full_kspace=batch.full_kspace,
                acq_start=batch.acq_start,
                acq_end=batch.acq_end,
                crop_size=batch.crop_size,
                return_mask_extras=return_mask_extras,
            )

        output = self.varnet(
            masked_kspace=batch.masked_kspace,
            mask=batch.mask,
            num_low_frequencies=batch.num_low_frequencies,
            crop_size=batch.crop_size,
        )

        if return_mask_extras:
            return output, None

        return output

    def _compute_reconstruction_loss(self, batch, output):
        target, output = center_crop_to_smallest(batch.target, output)

        ssim_loss = self.ssim_loss(
            output.unsqueeze(1),
            target.unsqueeze(1).float(),
        )

        reconstruction_loss = ssim_loss

        return target, output, reconstruction_loss, ssim_loss

    # ============================================================
    # TRAIN / VAL / TEST
    # ============================================================
    def training_step(self, batch, batch_idx):
        output = self._run_reconstruction_model(batch)
        _, _, loss, ssim_loss = self._compute_reconstruction_loss(batch, output)

        self.log("train/ssim", 1.0 - ssim_loss.detach(), sync_dist=True)
        self._log_learnable_mask_stats("train")
        self.log("train/loss", loss.detach(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.learnable_mask:
            output, mask_info = self._run_reconstruction_model(
                batch,
                return_mask_extras=True,
            )
            mask_2d = self._mask_2d_from_mask_info(mask_info)
        else:
            output = self._run_reconstruction_model(batch)
            mask_2d = self._extract_mask_2d(batch.mask)

        target, output, reconstruction_loss, ssim_loss = self._compute_reconstruction_loss(
            batch,
            output,
        )

        self._log_learnable_mask_stats("val")

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": reconstruction_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "ssim_loss": ssim_loss.detach(),
            "mask_2d": mask_2d,
        }

    def test_step(self, batch, batch_idx):
        mask_info = None

        if self.learnable_mask:
            output, mask_info = self._run_reconstruction_model(
                batch,
                return_mask_extras=True,
            )
            mask_2d = self._mask_2d_from_mask_info(mask_info)
        else:
            output = self._run_reconstruction_model(batch)
            mask_2d = self._extract_mask_2d(batch.mask)

        crop_size = self._resolve_crop_size(batch, output)
        output = center_crop(output, crop_size)

        acq_start = self._as_batch_tensor(
            batch.acq_start,
            device=output.device,
            dtype=torch.long,
        )
        acq_end = self._as_batch_tensor(
            batch.acq_end,
            device=output.device,
            dtype=torch.long,
        )

        if self.learnable_mask and mask_info is not None:
            support_width = mask_info.get("support_width", None)
            effective_acceleration = mask_info.get("effective_acceleration", None)
        else:
            support_width = (acq_end - acq_start).to(torch.float32)
            effective_acceleration = self._compute_effective_acceleration(
                mask_2d,
                support_width,
            )

        self._log_learnable_mask_stats("test")

        return {
            "fname": batch.fname,
            "output": output.detach().cpu().numpy(),
            "mask_2d": None if mask_2d is None else mask_2d.detach().cpu().numpy(),
            "support_width": None if support_width is None else support_width.detach().cpu().numpy(),
            "effective_acceleration": (
                None
                if effective_acceleration is None
                else effective_acceleration.detach().cpu().numpy()
            ),
            "acq_start": acq_start.detach().cpu().numpy(),
            "acq_end": acq_end.detach().cpu().numpy(),
        }

    # ============================================================
    # OPTIMIZER
    # ============================================================
    def configure_optimizers(self):
        lr_base = self.lr if self.lr_base is None else self.lr_base
        lr_mask = self.lr if self.lr_mask is None else self.lr_mask

        if (
            self.learnable_mask
            and hasattr(self.varnet, "base_varnet")
            and hasattr(self.varnet, "learnable_mask")
        ):
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.varnet.base_varnet.parameters(),
                        "lr": lr_base,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": self.varnet.learnable_mask.parameters(),
                        "lr": lr_mask,
                        "weight_decay": self.weight_decay,
                    },
                ]
            )
        else:
            optimizer = torch.optim.Adam(
                self.varnet.parameters(),
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
    # ARGS
    # ============================================================
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule3D.add_model_specific_args(parser)

        parser.add_argument("--num_cascades", type=int, default=12)
        parser.add_argument("--pools", type=int, default=4)
        parser.add_argument("--chans", type=int, default=18)
        parser.add_argument("--sens_pools", type=int, default=4)
        parser.add_argument("--sens_chans", type=int, default=8)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_base", type=float, default=None)
        parser.add_argument("--lr_mask", type=float, default=None)
        parser.add_argument("--lr_step_size", type=int, default=40)
        parser.add_argument("--lr_gamma", type=float, default=0.1)
        parser.add_argument("--ramp_steps", type=int, default=2618)
        parser.add_argument("--cosine_decay_start", type=int, default=32725)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--drop_prob", type=float, default=0.0)

        parser.add_argument("--kernel_size", nargs=3, type=int, default=[3, 3, 3])
        parser.add_argument("--pool_kernel_size", nargs=3, type=int, default=[1, 2, 2])

        return parser