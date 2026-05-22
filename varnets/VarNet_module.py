from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import math
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

from .Network_module import MriModule
from .VarNet import FIVarNet, E2EVarNet
from utilities.functions import center_crop_to_smallest, center_crop
from utilities.losses import SSIMLoss

class FIVarNetModule(MriModule):
    def __init__(
        self,
        fi_varnet: FIVarNet,
        model_name: str = "default_model",
        lr: float = 3e-4,
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

    def forward(self, batch, return_mask_extras: bool = False):
        return self._run_reconstruction_model(batch, return_mask_extras=return_mask_extras)

    def _run_reconstruction_model(self, batch, return_mask_extras: bool = False):
        
        return self.fi_varnet(
            batch.masked_kspace,
            batch.mask,
            batch.num_low_frequencies,
            crop_size=batch.crop_size,
        )

    def _compute_reconstruction_loss(self, batch, output):
        target, output = center_crop_to_smallest(batch.target, output)

        ssim_loss = self.ssim_loss(
            output.unsqueeze(1),
            target.unsqueeze(1).float(),
            data_range=batch.max_value,
        )

        reconstruction_loss = ssim_loss

        return target, output, reconstruction_loss, ssim_loss

    def training_step(self, batch, batch_idx):
        output = self._run_reconstruction_model(batch, return_mask_extras=False)
        target, output, loss, ssim_loss = self._compute_reconstruction_loss(batch, output)

        self.log("train/ssim", ssim_loss.detach(), sync_dist=True)
        self.log("train/loss", loss.detach(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self._run_reconstruction_model(batch, return_mask_extras=False)

        target, output, reconstruction_loss, ssim_loss = self._compute_reconstruction_loss(batch, output)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": reconstruction_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "ssim_loss": ssim_loss,
        }

    def test_step(self, batch, batch_idx):
       
        output = self._run_reconstruction_model(batch, return_mask_extras=False)

        crop_size = (
            (output.shape[-1], output.shape[-1])
            if output.shape[-1] < batch.crop_size[1]
            else batch.crop_size
        )
        output = center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        cosine_steps = self.max_steps - self.cosine_decay_start

        def step_fn(step):
            if step < self.cosine_decay_start:
                return min(step / self.ramp_steps, 1.0)
            angle = (step - self.cosine_decay_start) / cosine_steps * math.pi / 2
            return max(math.cos(angle), 1e-8)

        optimizer = torch.optim.AdamW(
            self.fi_varnet.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, step_fn)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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
        parser.add_argument("--lr_step_size", type=int, default=40)
        parser.add_argument("--lr_gamma", type=float, default=0.1)
        parser.add_argument("--ramp_steps", type=int, default=2618)
        parser.add_argument("--cosine_decay_start", type=int, default=32725)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--drop_prob", type=float, default=0.0)

        return parser
