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

    # ============================================================
    # VALIDATION STEP END
    # ============================================================
    def validation_step_end(self, val_logs):

        required = [
            "batch_idx","fname","slice_num","max_value",
            "output","target","val_loss"
        ]
        for k in required:
            if k not in val_logs:
                raise RuntimeError(f"Missing key {k} from validation_step.")

        # ensure correct shape
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        if val_logs.get("mask_1d") is not None and val_logs["mask_1d"].ndim == 1:
            val_logs["mask_1d"] = val_logs["mask_1d"].unsqueeze(0)

        # choose validation images once
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(
                    len(self.trainer.val_dataloaders[0])
                )[: self.num_log_images]
            )

        # log images
        batch_indices = (
            [val_logs["batch_idx"]]
            if isinstance(val_logs["batch_idx"], int)
            else val_logs["batch_idx"]
        )

        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"

                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)

                output = output / output.max()
                target = target / target.max()
                error = error / error.max()

                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

                if val_logs.get("mask_1d") is not None:
                    mask_1d = val_logs["mask_1d"][i].float().unsqueeze(0).unsqueeze(0)
                    self.log_image(f"{key}/mask_1d", mask_1d)

        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        perceptual_vals = defaultdict(dict)

        pl_layer_raw_vals = defaultdict(lambda: defaultdict(dict))
        pl_layer_weighted_vals = defaultdict(lambda: defaultdict(dict))
        max_vals = {}

        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluation.mse(target, output)
            ).view(1)

            target_norms[fname][slice_num] = torch.tensor(
                evaluation.mse(target, np.zeros_like(target))
            ).view(1)

            ssim_vals[fname][slice_num] = torch.tensor(
                evaluation.ssim(target[None,...], output[None,...], maxval=maxval)
            ).view(1)

            if val_logs.get("perceptual_loss") is not None:
                perceptual_vals[fname][slice_num] = torch.tensor(
                    val_logs["perceptual_loss"].item()
                ).view(1)

            if val_logs.get("pl_stats") is not None:
                stats = val_logs["pl_stats"]

                for l_idx, val in enumerate(stats["per_layer_raw"]):
                    pl_layer_raw_vals[fname][l_idx][slice_num] = torch.tensor(val).view(1)

                for l_idx, val in enumerate(stats["per_layer_weighted"]):
                    pl_layer_weighted_vals[fname][l_idx][slice_num] = torch.tensor(val).view(1)

            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "perceptual_vals": dict(perceptual_vals),
            "pl_layer_raw_vals": dict(pl_layer_raw_vals),
            "pl_layer_weighted_vals": dict(pl_layer_weighted_vals),
            "max_vals": max_vals,
        }

    # ============================================================
    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    # ============================================================
    # VALIDATION EPOCH END
    # ============================================================
    def validation_epoch_end(self, val_logs):

        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        perceptual_vals = defaultdict(dict)
        pl_layer_raw_vals = defaultdict(lambda: defaultdict(dict))
        pl_layer_weighted_vals = defaultdict(lambda: defaultdict(dict))
        max_vals = {}

        for log in val_logs:
            losses.append(log["val_loss"].view(-1))

            for k in log["mse_vals"]:
                mse_vals[k].update(log["mse_vals"][k])
            for k in log["target_norms"]:
                target_norms[k].update(log["target_norms"][k])
            for k in log["ssim_vals"]:
                ssim_vals[k].update(log["ssim_vals"][k])
            for k in log["perceptual_vals"]:
                perceptual_vals[k].update(log["perceptual_vals"][k])

            for k in log.get("pl_layer_raw_vals", {}):
                pl_layer_raw_vals[k].update(log["pl_layer_raw_vals"][k])

            for k in log.get("pl_layer_weighted_vals", {}):
                pl_layer_weighted_vals[k].update(log["pl_layer_weighted_vals"][k])

            for k in log["max_vals"]:
                max_vals[k] = log["max_vals"][k]

        metrics = {
            "nmse":0,
            "ssim":0,
            "psnr":0,
            "perceptual_loss":0,
        }

        local_examples = 0

        for fname in mse_vals:
            local_examples += 1

            mse_val = torch.mean(torch.cat([v.view(-1) for v in mse_vals[fname].values()]))
            target_norm = torch.mean(torch.cat([v.view(-1) for v in target_norms[fname].values()]))

            metrics["nmse"] += mse_val / target_norm

            metrics["psnr"] += (
                20*torch.log10(torch.tensor(max_vals[fname],dtype=mse_val.dtype,device=mse_val.device))
                -10*torch.log10(mse_val)
            )

            metrics["ssim"] += torch.mean(torch.cat([v.view(-1) for v in ssim_vals[fname].values()]))

            if fname in perceptual_vals:
                metrics["perceptual_loss"] += torch.mean(
                    torch.cat([v.view(-1) for v in perceptual_vals[fname].values()])
                )

        tot_examples = torch.tensor(local_examples, device=self.device)
        val_loss = torch.sum(torch.cat(losses)).to(self.device)
        tot_slice_examples = torch.tensor(len(losses), dtype=torch.float, device=self.device)

        self.log("val/loss", val_loss/tot_slice_examples, prog_bar=True, sync_dist=True)

        for m,v in metrics.items():
            self.log(f"val/{m}", v/tot_examples, sync_dist=True)

        # =====================================
        # Per-layer perceptual metrics
        # =====================================
        if pl_layer_weighted_vals:

            example_key = next(iter(pl_layer_weighted_vals))
            num_layers = len(pl_layer_weighted_vals[example_key])

            for layer_idx in range(num_layers):

                layer_w = 0

                for fname in pl_layer_weighted_vals:
                    if layer_idx in pl_layer_weighted_vals[fname]:
                        layer_w += torch.mean(
                            torch.cat([
                                v.view(-1)
                                for v in pl_layer_weighted_vals[fname][layer_idx].values()
                            ])
                        )

                self.log(
                    f"val/pl_layer_{layer_idx}",
                    layer_w / tot_examples,
                    sync_dist=True
                )

    # ============================================================
    # TEST
    # ============================================================
    def test_epoch_end(self, test_logs):

        outputs = defaultdict(dict)
        masks_1d = defaultdict(dict)
        support_widths = defaultdict(dict)
        accelerations = defaultdict(dict)
        acq_starts = defaultdict(dict)
        acq_ends = defaultdict(dict)

        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                sl = int(slice_num.cpu())

                outputs[fname][sl] = log["output"][i]

                if log.get("mask_1d") is not None:
                    masks_1d[fname][sl] = log["mask_1d"][i]

                if log.get("support_width") is not None:
                    support_widths[fname][sl] = log["support_width"][i]

                if log.get("effective_acceleration") is not None:
                    accelerations[fname][sl] = log["effective_acceleration"][i]

                if log.get("acq_start") is not None:
                    acq_starts[fname][sl] = int(log["acq_start"][i])

                if log.get("acq_end") is not None:
                    acq_ends[fname][sl] = int(log["acq_end"][i])

        packed = {}

        for fname in outputs:
            item = {
                "reconstruction": np.stack(
                    [out for _, out in sorted(outputs[fname].items())]
                )
            }

            if fname in masks_1d and len(masks_1d[fname]) > 0:
                item["mask_1d"] = np.stack(
                    [m for _, m in sorted(masks_1d[fname].items())]
                ).astype(np.uint8)

            if fname in support_widths and len(support_widths[fname]) > 0:
                item["support_width"] = np.array(
                    [v for _, v in sorted(support_widths[fname].items())]
                )

            if fname in accelerations and len(accelerations[fname]) > 0:
                item["effective_acceleration"] = np.array(
                    [v for _, v in sorted(accelerations[fname].items())]
                )

            if fname in acq_starts and len(acq_starts[fname]) > 0:
                item["acq_start"] = np.array(
                    [v for _, v in sorted(acq_starts[fname].items())]
                )

            if fname in acq_ends and len(acq_ends[fname]) > 0:
                item["acq_end"] = np.array(
                    [v for _, v in sorted(acq_ends[fname].items())]
                )

            packed[fname] = item

        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / f"reconstructions_{self.model_name}"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"

        self.print(f"Saving reconstructions to {save_path}")
        save_reconstructions(packed, save_path)

    # ============================================================
    # STATIC
    # ============================================================
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_log_images",default=16,type=int)
        return parser
