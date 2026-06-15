import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch

from utilities import evaluation
from utilities.functions import save_reconstructions


class MriModule(pl.LightningModule):
    def __init__(self, num_log_images: int = 16, model_name: str = "default_model"):
        super().__init__()

        self.num_log_images = num_log_images
        self.model_name = model_name
        self.val_log_indices = None

    def validation_step_end(self, val_logs):
        required = [
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ]
        for key in required:
            if key not in val_logs:
                raise RuntimeError(f"Missing key {key} from validation_step.")

        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        if val_logs.get("mask_1d") is not None and val_logs["mask_1d"].ndim == 1:
            val_logs["mask_1d"] = val_logs["mask_1d"].unsqueeze(0)

        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        batch_indices = (
            [val_logs["batch_idx"]]
            if isinstance(val_logs["batch_idx"], int)
            else val_logs["batch_idx"]
        )

        for index, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"

                target = val_logs["target"][index].unsqueeze(0)
                output = val_logs["output"][index].unsqueeze(0)
                error = torch.abs(target - output)

                output = output / output.max().clamp_min(torch.finfo(output.dtype).eps)
                target = target / target.max().clamp_min(torch.finfo(target.dtype).eps)
                error = error / error.max().clamp_min(torch.finfo(error.dtype).eps)

                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

                if val_logs.get("mask_1d") is not None:
                    mask_1d = val_logs["mask_1d"][index].float()[None, None, :]
                    self.log_image(f"{key}/mask_1d", mask_1d)

        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = {}

        for index, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][index].cpu())
            maxval = val_logs["max_value"][index].cpu().numpy()
            output = val_logs["output"][index].cpu().numpy()
            target = val_logs["target"][index].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluation.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluation.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluation.ssim(
                    target[None, ...],
                    output[None, ...],
                    maxval=maxval,
                )
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, val_logs):
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = {}

        for log in val_logs:
            losses.append(log["val_loss"].view(-1))

            for key in log["mse_vals"]:
                mse_vals[key].update(log["mse_vals"][key])
            for key in log["target_norms"]:
                target_norms[key].update(log["target_norms"][key])
            for key in log["ssim_vals"]:
                ssim_vals[key].update(log["ssim_vals"][key])
            for key in log["max_vals"]:
                max_vals[key] = log["max_vals"][key]

        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0

        for fname in mse_vals:
            local_examples += 1
            mse_val = torch.mean(
                torch.cat([value.view(-1) for value in mse_vals[fname].values()])
            )
            target_norm = torch.mean(
                torch.cat(
                    [value.view(-1) for value in target_norms[fname].values()]
                )
            )

            metrics["nmse"] += mse_val / target_norm
            metrics["psnr"] += 20 * torch.log10(
                torch.tensor(
                    max_vals[fname],
                    dtype=mse_val.dtype,
                    device=mse_val.device,
                )
            ) - 10 * torch.log10(mse_val)
            metrics["ssim"] += torch.mean(
                torch.cat([value.view(-1) for value in ssim_vals[fname].values()])
            )

        tot_examples = torch.tensor(local_examples, device=self.device)
        val_loss = torch.sum(torch.cat(losses)).to(self.device)
        tot_slice_examples = torch.tensor(
            len(losses),
            dtype=torch.float,
            device=self.device,
        )

        self.log(
            "val/loss",
            val_loss / tot_slice_examples,
            prog_bar=True,
            sync_dist=True,
        )

        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value / tot_examples,
                sync_dist=True,
            )

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(dict)
        masks_1d = defaultdict(dict)
        support_widths = defaultdict(dict)
        accelerations = defaultdict(dict)
        acq_starts = defaultdict(dict)
        acq_ends = defaultdict(dict)

        for log in test_logs:
            for index, (fname, slice_num) in enumerate(
                zip(log["fname"], log["slice"])
            ):
                slice_index = int(slice_num.cpu())
                outputs[fname][slice_index] = log["output"][index]

                if log.get("mask_1d") is not None:
                    masks_1d[fname][slice_index] = log["mask_1d"][index]
                if log.get("support_width") is not None:
                    support_widths[fname][slice_index] = log["support_width"][index]
                if log.get("effective_acceleration") is not None:
                    accelerations[fname][slice_index] = log[
                        "effective_acceleration"
                    ][index]
                if log.get("acq_start") is not None:
                    acq_starts[fname][slice_index] = int(log["acq_start"][index])
                if log.get("acq_end") is not None:
                    acq_ends[fname][slice_index] = int(log["acq_end"][index])

        packed = {}

        for fname in outputs:
            item = {
                "reconstruction": np.stack(
                    [output for _, output in sorted(outputs[fname].items())]
                )
            }

            if masks_1d[fname]:
                item["mask_1d"] = np.stack(
                    [mask for _, mask in sorted(masks_1d[fname].items())]
                ).astype(np.uint8)
            if support_widths[fname]:
                item["support_width"] = np.asarray(
                    [value for _, value in sorted(support_widths[fname].items())]
                )
            if accelerations[fname]:
                item["effective_acceleration"] = np.asarray(
                    [value for _, value in sorted(accelerations[fname].items())]
                )
            if acq_starts[fname]:
                item["acq_start"] = np.asarray(
                    [value for _, value in sorted(acq_starts[fname].items())]
                )
            if acq_ends[fname]:
                item["acq_end"] = np.asarray(
                    [value for _, value in sorted(acq_ends[fname].items())]
                )

            packed[fname] = item

        if hasattr(self, "trainer"):
            save_path = (
                pathlib.Path(self.trainer.default_root_dir)
                / f"reconstructions_{self.model_name}"
            )
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"

        self.print(f"Saving reconstructions to {save_path}")
        save_reconstructions(packed, save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_log_images", default=16, type=int)
        return parser
