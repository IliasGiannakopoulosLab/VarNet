from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import math
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

from .Network_module import MriModule
from .VarNet import FIVarNet, E2EVarNet
from utilities.feature_extractor import YOLOv5FeatureExtractor, YOLOGradCAM
from utilities.functions import center_crop_to_smallest, center_crop
from utilities.losses import SSIMLoss, YOLOPerceptualLoss

class FIVarNetModule(MriModule):
    def __init__(
        self,
        fi_varnet: FIVarNet,
        learnable_mask: bool = False,
        model_name: str = "default_model",
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
        # -------- YOLO --------
        yolo_cfg: str = "path/to/yolo.yaml",
        yolo_repo: str = "path/to/yolo/",
        yolo_weights: str = "path/to/weights.pt",
        yolo_flag: int = 1,
        device: str = "0",
        imgsz: tuple = (640, 640),
        half: bool = False,
        # -------- perceptual --------
        pl_weight: float = 1.0,
        feature_layers=None,
        layer_weights=None,
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

        self.fi_varnet = fi_varnet
        self.ssim_loss = SSIMLoss()
        self.learnable_mask = learnable_mask

        self.imgsz = imgsz
        self.yolo_flag = yolo_flag

        if self.yolo_flag == 1:
            if feature_layers is None:
                feature_layers = [4, 6, 9]
            self.feature_layers = feature_layers

            if layer_weights is None:
                layer_weights = [1.0, 1.0, 1.41]
            self.layer_weights = layer_weights

            self.yolo_feat = YOLOv5FeatureExtractor(
                cfg=yolo_cfg,
                weights=yolo_weights,
                device="cpu",
                imgsz=self.imgsz,
                feature_layers=self.feature_layers,
            )
            for p in self.yolo_feat.parameters():
                p.requires_grad = False
            self.yolo_feat.eval()

            self.yolo = torch.hub.load(
                str(yolo_repo),
                "custom",
                path=yolo_weights,
                source="local",
                autoshape=False,
            ).eval()
            for p in self.yolo.parameters():
                p.requires_grad = False

            self.gradcam = YOLOGradCAM(self.yolo)
            self.perceptual_loss = YOLOPerceptualLoss(
                yolo_model=self.yolo_feat,
                gradcam=self.gradcam,
                feature_layers=self.feature_layers,
                imgsz=self.imgsz,
                layer_weights=self.layer_weights,
                pl_weight=pl_weight,
            )

    def forward(self, batch, return_mask_extras: bool = False):
        return self._run_reconstruction_model(batch, return_mask_extras=return_mask_extras)

    def _run_reconstruction_model(self, batch, return_mask_extras: bool = False):
        if self.learnable_mask:
            return self.fi_varnet(
                full_kspace=batch.full_kspace,
                acq_start=batch.acq_start,
                acq_end=batch.acq_end,
                crop_size=batch.crop_size,
                return_mask_extras=return_mask_extras,
            )

        return self.fi_varnet(
            batch.masked_kspace,
            batch.mask,
            batch.num_low_frequencies,
            crop_size=batch.crop_size,
        )

    def _get_latest_mask_info(self):
        if not self.learnable_mask:
            return None
        if not hasattr(self.fi_varnet, "latest_mask_info"):
            return None
        return self.fi_varnet.latest_mask_info

    def _log_learnable_mask_stats(self, stage: str):
        info = self._get_latest_mask_info()
        if info is None:
            return

        if "sampled_lines" in info:
            self.log(f"{stage}/sampled_lines", info["sampled_lines"].mean().detach(), sync_dist=True)
        if "support_width" in info:
            self.log(f"{stage}/support_width", info["support_width"].mean().detach(), sync_dist=True)
        if "effective_acceleration" in info:
            self.log(f"{stage}/acceleration", info["effective_acceleration"].mean().detach(), sync_dist=True)
        if "outer_prob_raw_mean" in info:
            self.log(f"{stage}/mask_prob_raw_mean", info["outer_prob_raw_mean"].mean().detach(), sync_dist=True)
        if "outer_prob_mean" in info:
            self.log(f"{stage}/mask_prob_mean", info["outer_prob_mean"].mean().detach(), sync_dist=True)
        if "prob_mask_1d" in info:
            prob_mask_1d = info["prob_mask_1d"]
            self.log(f"{stage}/mask_prob_std", prob_mask_1d.std().detach(), sync_dist=True)
        if "target_outer_mean" in info:
            self.log(f"{stage}/target_outer_mean", info["target_outer_mean"].mean().detach(), sync_dist=True)
        if "target_total_ratio" in info:
            self.log(f"{stage}/target_total_ratio", info["target_total_ratio"].mean().detach(), sync_dist=True)

    def _compute_reconstruction_loss(self, batch, output):
        target, output = center_crop_to_smallest(batch.target, output)

        ssim_loss = self.ssim_loss(
            output.unsqueeze(1),
            target.unsqueeze(1).float(),
            data_range=batch.max_value,
        )

        if self.yolo_flag == 1:
            perceptual_loss, stats = self.perceptual_loss(
                target.float(), output, return_stats=True
            )
            reconstruction_loss = ssim_loss + perceptual_loss
        else:
            perceptual_loss = torch.tensor(0.0, device=ssim_loss.device)
            reconstruction_loss = ssim_loss
            stats = None

        return target, output, reconstruction_loss, ssim_loss, perceptual_loss, stats

    def training_step(self, batch, batch_idx):
        output = self._run_reconstruction_model(batch, return_mask_extras=False)
        target, output, loss, ssim_loss, perceptual_loss, _ = self._compute_reconstruction_loss(batch, output)

        self.log("train/ssim", ssim_loss.detach(), sync_dist=True)
        if self.yolo_flag == 1:
            self.log("train/pl", perceptual_loss.detach(), sync_dist=True)

        self._log_learnable_mask_stats("train")
        self.log("train/loss", loss.detach(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mask_1d = None

        if self.learnable_mask:
            output, mask_info = self._run_reconstruction_model(batch, return_mask_extras=True)
            mask_1d = mask_info.get("hard_mask_1d", None)
        else:
            output = self._run_reconstruction_model(batch, return_mask_extras=False)

        target, output, reconstruction_loss, ssim_loss, perceptual_loss, stats = self._compute_reconstruction_loss(batch, output)

        self._log_learnable_mask_stats("val")

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": reconstruction_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "perceptual_loss": perceptual_loss,
            "ssim_loss": ssim_loss,
            "pl_stats": stats,
            "mask_1d": mask_1d,
        }

    def test_step(self, batch, batch_idx):
        mask_1d = None
        support_width = None
        effective_acceleration = None

        if self.learnable_mask:
            output, mask_info = self._run_reconstruction_model(
                batch, return_mask_extras=True
            )
            mask_1d = mask_info.get("hard_mask_1d", None)
            support_width = mask_info.get("support_width", None)
            effective_acceleration = mask_info.get("effective_acceleration", None)
        else:
            output = self._run_reconstruction_model(batch, return_mask_extras=False)

            fixed_mask = batch.mask
            if fixed_mask is not None:
                # batch.mask shape: [B, 1, 1, W, 1]
                mask_1d = fixed_mask[:, 0, 0, :, 0].to(torch.uint8)

            support_width = (batch.acq_end - batch.acq_start).to(torch.float32)

            if mask_1d is not None:
                sampled_lines = mask_1d.sum(dim=1).to(torch.float32)
                effective_acceleration = support_width / sampled_lines.clamp_min(1.0)

        crop_size = (
            (output.shape[-1], output.shape[-1])
            if output.shape[-1] < batch.crop_size[1]
            else batch.crop_size
        )
        output = center_crop(output, crop_size)

        self._log_learnable_mask_stats("test")

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
            "mask_1d": None if mask_1d is None else mask_1d.cpu().numpy(),
            "support_width": None if support_width is None else support_width.cpu().numpy(),
            "effective_acceleration": None if effective_acceleration is None else effective_acceleration.cpu().numpy(),
            "acq_start": batch.acq_start.cpu().numpy(),
            "acq_end": batch.acq_end.cpu().numpy(),
        }

    def configure_optimizers(self):
        cosine_steps = self.max_steps - self.cosine_decay_start

        def step_fn(step):
            if step < self.cosine_decay_start:
                return min(step / self.ramp_steps, 1.0)
            angle = (step - self.cosine_decay_start) / cosine_steps * math.pi / 2
            return max(math.cos(angle), 1e-8)

        lr_base = self.lr if self.lr_base is None else self.lr_base
        lr_mask = self.lr if self.lr_mask is None else self.lr_mask

        if (
            self.learnable_mask
            and hasattr(self.fi_varnet, "base_varnet")
            and hasattr(self.fi_varnet, "learnable_mask")
        ):
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.fi_varnet.base_varnet.parameters(),
                        "lr": lr_base,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": self.fi_varnet.learnable_mask.parameters(),
                        "lr": lr_mask,
                        "weight_decay": self.weight_decay,
                    },
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                self.fi_varnet.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, step_fn)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.yolo_flag == 0:
            state_dict = {
                k: v for k, v in state_dict.items()
                if not (
                    k.startswith("yolo")
                    or k.startswith("yolo_feat")
                    or k.startswith("perceptual_loss")
                    or k.startswith("gradcam")
                )
            }
            strict = False
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

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

        return parser
