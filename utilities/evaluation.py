from argparse import ArgumentParser
from typing import Optional
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import argparse
import pathlib
import h5py
import numpy as np

from utilities.functions import center_crop

# -------------------------------------------#
# -------------------- mse ----------------- #
# -------------------------------------------#
def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.mean((gt - pred) ** 2)


# -------------------------------------------#
# ------------------- nmse ----------------- #
# -------------------------------------------#
def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


# -------------------------------------------#
# ------------------- psnr ----------------- #
# -------------------------------------------#
def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


# -------------------------------------------#
# ------------------- ssim ----------------- #
# -------------------------------------------#
def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim_val = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim_val = ssim_val + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim_val / gt.shape[0]


# -------------------------------------------#
# ---------------- metric funcs ------------ #
# -------------------------------------------#
METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


# -------------------------------------------#
# ------------------- metrics -------------- #
# -------------------------------------------#
class Metrics:

    def __init__(self, metric_funcs):
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        def format_value(value):
            if isinstance(value, np.ndarray):
                return "[" + ", ".join(f"{v:.4g}" for v in value) + "]"
            else:
                return f"{value:.4g}"

        def format_stddev(value):
            if isinstance(value, np.ndarray):
                return "[" + ", ".join(f"{2 * v:.4g}" for v in value) + "]"
            else:
                return f"{2 * value:.4g}"

        return " ".join(
            f"{name} = {format_value(means[name])} +/- {format_stddev(stddevs[name])}"
            for name in metric_names
        )


# -------------------------------------------#
# ------------------ evaluate -------------- #
# -------------------------------------------#
def evaluate(target_path, predictions_path):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(
            predictions_path / tgt_file.name, "r"
        ) as recons:

            target = target["reconstruction_rss"][()]
            recons = recons["reconstruction"][()]

            target = center_crop(
                target, (target.shape[-1], target.shape[-1])
            )
            recons = center_crop(
                recons, (target.shape[-1], target.shape[-1])
            )

            metrics.push(target, recons)

    return metrics


# -------------------------------------------#
# ------------------- main ----------------- #
# -------------------------------------------#
if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )

    args = parser.parse_args()

    metrics = evaluate(args.target_path, args.predictions_path)
    print(metrics)
