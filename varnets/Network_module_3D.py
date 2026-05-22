import pathlib
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch

from utilities import evaluation
from utilities.functions import save_reconstructions


class MriModule3D(pl.LightningModule):

    def __init__(
        self,
        num_log_images: int = 16,
        model_name: str = "default_model_3d",
        log_slice: str = "middle",
    ):
        super().__init__()

        if log_slice not in ("first", "middle"):
            raise ValueError("log_slice must be 'first' or 'middle'.")

        self.num_log_images = num_log_images
        self.model_name = model_name
        self.log_slice = log_slice
        self.val_log_indices = None

    # ============================================================
    # HELPERS
    # ============================================================
    def _get_batch_size(self, fname):
        if isinstance(fname, (list, tuple)):
            return len(fname)
        return 1

    def _get_fname(self, fname, index: int):
        if isinstance(fname, (list, tuple)):
            return fname[index]
        return fname

    def _get_log_slice_index(self, depth: int) -> int:
        if self.log_slice == "first":
            return 0
        return depth // 2

    def _safe_normalize(self, image: torch.Tensor) -> torch.Tensor:
        max_val = image.max()
        if max_val > 0:
            return image / max_val
        return image

    # ============================================================
    # VALIDATION STEP END
    # ============================================================
    def validation_step_end(self, val_logs):

        required = [
            "batch_idx",
            "fname",
            "max_value",
            "output",
            "target",
            "val_loss",
        ]
        for k in required:
            if k not in val_logs:
                raise RuntimeError(f"Missing key {k} from validation_step.")

        # Expected:
        # output: [B, D, H, W]
        # target: [B, D, H, W]
        if val_logs["output"].ndim != 4:
            raise RuntimeError(
                f"Expected 3D output with shape [B, D, H, W], got {val_logs['output'].shape}"
            )
        if val_logs["target"].ndim != 4:
            raise RuntimeError(
                f"Expected 3D target with shape [B, D, H, W], got {val_logs['target'].shape}"
            )

        # choose validation volumes once
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(
                    len(self.trainer.val_dataloaders[0])
                )[: self.num_log_images]
            )

        batch_idx = val_logs["batch_idx"]

        # log one representative slice from selected validation volumes
        if batch_idx in self.val_log_indices:
            depth = val_logs["output"].shape[1]
            slice_idx = self._get_log_slice_index(depth)

            batch_size = val_logs["output"].shape[0]

            for i in range(batch_size):
                key = f"val_images_idx_{batch_idx}"
                if batch_size > 1:
                    key = f"{key}_vol_{i}"

                target_slice = val_logs["target"][i, slice_idx].unsqueeze(0)
                output_slice = val_logs["output"][i, slice_idx].unsqueeze(0)
                error_slice = torch.abs(target_slice - output_slice)

                output_slice = self._safe_normalize(output_slice)
                target_slice = self._safe_normalize(target_slice)
                error_slice = self._safe_normalize(error_slice)

                self.log_image(f"{key}/target", target_slice)
                self.log_image(f"{key}/reconstruction", output_slice)
                self.log_image(f"{key}/error", error_slice)

                if val_logs.get("mask_2d") is not None:
                    # mask_2d: [B, D, W]
                    mask_slice = val_logs["mask_2d"][i, slice_idx]
                    mask_slice = mask_slice.float().unsqueeze(0).unsqueeze(0)
                    self.log_image(f"{key}/mask", mask_slice)

        mse_vals = {}
        target_norms = {}
        ssim_vals = {}
        psnr_vals = {}
        max_vals = {}

        batch_size = val_logs["output"].shape[0]

        for i in range(batch_size):
            fname = self._get_fname(val_logs["fname"], i)
            maxval = val_logs["max_value"][i].detach().cpu().numpy()

            output = val_logs["output"][i].detach().cpu().numpy()
            target = val_logs["target"][i].detach().cpu().numpy()

            mse_val = evaluation.mse(target, output)
            target_norm = evaluation.mse(target, np.zeros_like(target))
            ssim_val = evaluation.ssim(target, output, maxval=maxval)
            psnr_val = evaluation.psnr(target, output, maxval=maxval)

            mse_vals[fname] = torch.tensor(mse_val).view(1)
            target_norms[fname] = torch.tensor(target_norm).view(1)
            ssim_vals[fname] = torch.tensor(ssim_val).view(1)
            psnr_vals[fname] = torch.tensor(psnr_val).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "psnr_vals": psnr_vals,
            "max_vals": max_vals,
        }

    # ============================================================
    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    # ============================================================
    # VALIDATION EPOCH END
    # ============================================================
    def validation_epoch_end(self, val_logs):

        if len(val_logs) == 0:
            return

        losses = []
        mse_vals = {}
        target_norms = {}
        ssim_vals = {}
        psnr_vals = {}

        for log in val_logs:
            losses.append(log["val_loss"].view(-1))

            mse_vals.update(log["mse_vals"])
            target_norms.update(log["target_norms"])
            ssim_vals.update(log["ssim_vals"])
            psnr_vals.update(log["psnr_vals"])

        metrics = {
            "nmse": 0,
            "ssim": 0,
            "psnr": 0,
        }

        local_examples = len(mse_vals)

        for fname in mse_vals:
            mse_val = mse_vals[fname].to(self.device)
            target_norm = target_norms[fname].to(self.device)

            metrics["nmse"] += mse_val / target_norm
            metrics["ssim"] += ssim_vals[fname].to(self.device)
            metrics["psnr"] += psnr_vals[fname].to(self.device)

        tot_examples = torch.tensor(local_examples, dtype=torch.float, device=self.device)
        val_loss = torch.sum(torch.cat(losses)).to(self.device)
        tot_volume_examples = torch.tensor(len(losses), dtype=torch.float, device=self.device)

        self.log("val/loss", val_loss / tot_volume_examples, prog_bar=True, sync_dist=True)

        for m, v in metrics.items():
            self.log(f"val/{m}", v / tot_examples, sync_dist=True)

    # ============================================================
    # TEST
    # ============================================================
    def test_epoch_end(self, test_logs):

        packed = {}

        for log in test_logs:
            batch_size = log["output"].shape[0]

            for i in range(batch_size):
                fname = self._get_fname(log["fname"], i)

                item = {
                    "reconstruction": log["output"][i],
                }

                if log.get("mask_2d") is not None:
                    item["mask_2d"] = log["mask_2d"][i].astype(np.uint8)

                if log.get("support_width") is not None:
                    item["support_width"] = log["support_width"][i]

                if log.get("effective_acceleration") is not None:
                    item["effective_acceleration"] = log["effective_acceleration"][i]

                if log.get("acq_start") is not None:
                    item["acq_start"] = log["acq_start"][i]

                if log.get("acq_end") is not None:
                    item["acq_end"] = log["acq_end"][i]

                packed[fname] = item

        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / f"reconstructions_{self.model_name}"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"

        self.print(f"Saving 3D reconstructions to {save_path}")
        save_reconstructions(packed, save_path)

    # ============================================================
    # STATIC
    # ============================================================
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--num_log_images", default=16, type=int)
        parser.add_argument(
            "--log_slice",
            default="middle",
            choices=("first", "middle"),
            type=str,
        )

        return parser