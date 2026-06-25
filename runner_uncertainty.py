import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_module import DataModule
from data.data_transforms import VarNetDataTransform
from data.undersampling_patterns import create_mask_for_mask_type
from varnets.VarNet import (
    E2EVarNet,
    FIVarNet,
    LearnableMaskedVarNet,
    UncertaintyNetwork,
)
from varnets.uncertainty_module import UncertaintyModule


torch.set_float32_matmul_precision("high")


# ============================================================
# UTILS
# ============================================================
def check_gpu_availability() -> int:
    command = "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l"
    output = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    return int(output)


def _adapt_state_dict_for_model(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Adapts checkpoint keys for wrapped learnable-mask models.

    Cases handled:
      source checkpoint = plain VarNet
      target model      = LearnableMaskedVarNet(...)

    Then we remap:
      conv.weight -> base_varnet.conv.weight

    If the source checkpoint is already a learnable-mask wrapper, the keys are
    left unchanged after the Lightning-module prefix is removed.
    """
    target_keys = set(model.state_dict().keys())

    target_is_wrapper = any(k.startswith("base_varnet.") for k in target_keys)
    source_is_wrapper = any(
        k.startswith("base_varnet.") or k.startswith("learnable_mask.")
        for k in state_dict.keys()
    )

    if target_is_wrapper and not source_is_wrapper:
        state_dict = {f"base_varnet.{k}": v for k, v in state_dict.items()}

    return state_dict


def _filter_state_dict_by_prefix(state_dict: dict, module_name: str) -> dict:
    lm = len(module_name)
    return {
        k[lm:]: v
        for k, v in state_dict.items()
        if k.startswith(module_name)
    }


def _get_varnet_state_dict_from_checkpoint(
    weight_path: Path,
    module_name: Optional[str],
) -> dict:
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    prefixes_to_try = []
    if module_name:
        prefixes_to_try.append(module_name)

    for prefix in ("fi_varnet.", "varnet."):
        if prefix not in prefixes_to_try:
            prefixes_to_try.append(prefix)

    for prefix in prefixes_to_try:
        filtered = _filter_state_dict_by_prefix(state_dict, prefix)
        if len(filtered) > 0:
            return filtered

    if any(k.startswith("base_varnet.") for k in state_dict.keys()):
        return state_dict

    raise RuntimeError(
        f"No VarNet weights found in {weight_path}. "
        f"Tried prefixes: {prefixes_to_try}."
    )


def load_varnet_weights_only(
    model: torch.nn.Module,
    weight_path: Path,
    module_name: Optional[str] = "fi_varnet.",
):
    filtered = _get_varnet_state_dict_from_checkpoint(
        weight_path=weight_path,
        module_name=module_name,
    )
    filtered = _adapt_state_dict_for_model(model, filtered)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("load_varnet_weights_only:")
    print("  missing keys   :", len(missing))
    print("  unexpected keys:", len(unexpected))

    return model


def _get_single_learnable_mask_params(args):
    if len(args.accelerations) != 1:
        raise ValueError(
            "Learnable mask mode currently requires exactly one acceleration."
        )
    if len(args.center_fractions) != 1:
        raise ValueError(
            "Learnable mask mode currently requires exactly one center fraction."
        )

    acceleration = int(args.accelerations[0])
    center_fraction = float(args.center_fractions[0])

    return acceleration, center_fraction


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_base_varnet(args, acceleration: int):
    if args.varnet_type == "fi_varnet":
        print(f"BUILDING FI VARNET, chans={args.chans}")
        return FIVarNet(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            acceleration=acceleration,
        )

    if args.varnet_type == "e2e_varnet":
        print(f"BUILDING E2E VARNET, chans={args.chans}")
        return E2EVarNet(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
        )

    raise ValueError(f"Unknown varnet_type: {args.varnet_type}")


def fetch_varnet_model(args):
    if args.mask_mode == "learnable":
        acceleration, center_fraction = _get_single_learnable_mask_params(args)
        base_varnet = build_base_varnet(args, acceleration=acceleration)

        print(
            f"WRAPPING WITH LEARNABLE CARTESIAN MASK, "
            f"acceleration={acceleration}, center_fraction={center_fraction}"
        )

        return LearnableMaskedVarNet(
            base_varnet=base_varnet,
            acceleration=acceleration,
            center_fraction=center_fraction,
            num_logits=args.num_logits,
        )

    acceleration = int(round(sum(args.accelerations) / len(args.accelerations)))
    return build_base_varnet(args, acceleration=acceleration)


def fetch_uncertainty_model(args):
    return UncertaintyNetwork(
        head_chans=args.head_chans,
        head_pools=args.head_pools,
        drop_prob=args.uncertainty_drop_prob,
    )


# ============================================================
# TRANSFORMS
# ============================================================
def build_transforms(args):
    if args.mask_mode == "learnable":
        train_transform = VarNetDataTransform(
            mask_func=None,
            use_seed=False,
            learnable_mask=True,
        )
        val_transform = VarNetDataTransform(
            mask_func=None,
            use_seed=True,
            learnable_mask=True,
        )
        cal_transform = VarNetDataTransform(
            mask_func=None,
            use_seed=True,
            learnable_mask=True,
        )
        test_transform = VarNetDataTransform(
            mask_func=None,
            use_seed=True,
            learnable_mask=True,
        )
        return train_transform, val_transform, cal_transform, test_transform

    mask = create_mask_for_mask_type(
        args.mask_type,
        args.center_fractions,
        args.accelerations,
    )

    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    cal_transform = VarNetDataTransform(mask_func=mask)

    if args.mode == "test_val":
        test_transform = VarNetDataTransform(mask_func=mask)
    else:
        test_transform = VarNetDataTransform()

    return train_transform, val_transform, cal_transform, test_transform


# ============================================================
# TRAIN / CALIBRATE / TEST
# ============================================================
def _resolve_uncertainty_resume_ckpt(args) -> Optional[Path]:
    if args.resume_from_checkpoint is not None:
        return Path(args.resume_from_checkpoint)

    checkpoint_dir = args.default_root_dir / "checkpoints" / f"{args.model_name}"
    last_ckpt = checkpoint_dir / "last.ckpt"

    if last_ckpt.exists():
        print(f"Auto-resuming uncertainty training from {last_ckpt}")
        return last_ckpt

    return None


def _resolve_uncertainty_inference_ckpt(args) -> Optional[Path]:
    if args.uncertainty_ckpt is not None:
        return Path(args.uncertainty_ckpt)

    checkpoint_dir = args.default_root_dir / "checkpoints" / f"{args.model_name}"
    best_ckpt = checkpoint_dir / "best.ckpt"
    last_ckpt = checkpoint_dir / "last.ckpt"

    if best_ckpt.exists():
        print(f"Using best uncertainty checkpoint from {best_ckpt}")
        return best_ckpt

    if last_ckpt.exists():
        print(f"Best checkpoint not found, falling back to {last_ckpt}")
        return last_ckpt

    return None


def cli_main(args):
    pl.seed_everything(args.seed)

    original_mode = args.mode

    if args.varnet_ckpt is None:
        raise ValueError("--varnet_ckpt is required for uncertainty training/calibration/testing.")

    train_transform, val_transform, cal_transform, test_transform = build_transforms(args)

    if original_mode == "test_val":
        args.mode = "test"

    data_module = DataModule(
        data_path=args.data_path,
        data_path_train=args.data_path_train,
        data_path_val=args.data_path_val,
        data_path_cal=args.data_path_cal,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        cal_transform=cal_transform,
        combine_train_val=args.combine_train_val,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        val_sample_rate=args.val_sample_rate,
        cal_sample_rate=args.cal_sample_rate,
        test_sample_rate=args.test_sample_rate,
        file_split=args.file_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    varnet_model = fetch_varnet_model(args)
    varnet_model = load_varnet_weights_only(
        varnet_model,
        Path(args.varnet_ckpt),
        module_name=args.varnet_ckpt_prefix,
    )

    uncertainty_model = fetch_uncertainty_model(args)

    pl_module = UncertaintyModule(
        varnet_model=varnet_model,
        uncertainty_model=uncertainty_model,
        learnable_mask=(args.mask_mode == "learnable"),
        model_name=args.model_name,
        varnet_ckpt=None,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        ramp_steps=args.ramp_steps,
        cosine_decay_start=args.cosine_decay_start,
        alpha=args.alpha,
        calibration_delta=args.calibration_delta,
        calibration_lambdas_start=args.calibration_lambdas_start,
        calibration_lambdas_end=args.calibration_lambdas_end,
        calibration_lambdas_steps=args.calibration_lambdas_steps,
        calibration_sample_rate=args.calibration_sample_rate,
        reconstructions_dir=args.reconstructions_dir,
    )

    checkpoint_dir = args.default_root_dir / "checkpoints" / f"{args.model_name}"
    callbacks = []
    ckpt_path = None

    if args.mode == "train":
        ckpt_path = _resolve_uncertainty_resume_ckpt(args)
        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best",
                auto_insert_metric_name=False,
                save_last=True,
                save_top_k=1,
                verbose=True,
                monitor="val/loss",
                mode="min",
            ),
            LearningRateMonitor(),
        ]

    elif args.mode in ("calibrate", "test"):
        resolved_unc_ckpt = _resolve_uncertainty_inference_ckpt(args)
        if resolved_unc_ckpt is None:
            raise ValueError(
                f"No uncertainty checkpoint found for mode={args.mode}. "
                f"Pass --uncertainty_ckpt or make sure best.ckpt exists."
            )

        args.uncertainty_ckpt = resolved_unc_ckpt
        print(f"Loading uncertainty checkpoint from {args.uncertainty_ckpt}")
        pl_module.load_uncertainty_checkpoint(Path(args.uncertainty_ckpt), strict=True)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    args.resume_from_checkpoint = None
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    if args.mode == "train":
        trainer.fit(pl_module, datamodule=data_module, ckpt_path=ckpt_path)

    elif args.mode == "calibrate":
        if args.devices not in (1, "1") and args.accelerator in ("gpu", "ddp", "ddp_cpu"):
            raise ValueError("Calibration mode currently expects a single process/device.")

        calib_loader = data_module.calib_dataloader()
        pl_module.calibrate(calib_loader)

        output_ckpt = (
            Path(args.output_uncertainty_ckpt)
            if args.output_uncertainty_ckpt is not None
            else Path(args.uncertainty_ckpt)
        )
        print(f"Saving calibrated uncertainty checkpoint to {output_ckpt}")
        pl_module.update_checkpoint_file(Path(args.uncertainty_ckpt), output_ckpt)

    elif args.mode == "test":
        if args.reconstructions_dir is None:
            raise ValueError("--reconstructions_dir is required for uncertainty test mode.")
        trainer.test(pl_module, datamodule=data_module)


# ============================================================
# ARGS
# ============================================================
def build_args(cluster_launch: bool = True):
    num_gpus = check_gpu_availability()

    parser = ArgumentParser()
    parser.add_argument("--log_path", type=Path, default=None)
    parser.add_argument("--model_name", type=str, default="Uncertainty")
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "calibrate", "test", "test_val"),
        type=str,
    )

    parser.add_argument("--varnet_ckpt", type=Path, default=None)
    parser.add_argument("--varnet_ckpt_prefix", type=str, default=None)
    parser.add_argument("--uncertainty_ckpt", type=Path, default=None)
    parser.add_argument("--output_uncertainty_ckpt", type=Path, default=None)
    parser.add_argument("--reconstructions_dir", type=Path, default=None)

    parser.add_argument(
        "--mask_mode",
        choices=("fixed", "learnable"),
        default="fixed",
        type=str,
    )

    parser.add_argument(
        "--mask_type",
        choices=("equispaced_fraction", "equispaced"),
        default="equispaced_fraction",
        type=str,
    )

    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int)
    parser.add_argument("--num_logits", type=int, default=320)

    parser.add_argument(
        "--varnet_type",
        choices=("fi_varnet", "e2e_varnet"),
        default="fi_varnet",
    )

    parser.add_argument("--head_chans", type=int, default=32)
    parser.add_argument("--head_pools", type=int, default=4)
    parser.add_argument("--uncertainty_drop_prob", type=float, default=0.0)

    parser = DataModule.add_data_specific_args(parser)

    args, _ = parser.parse_known_args()
    if args.log_path is None:
        raise ValueError("--log_path must be provided.")

    default_root_dir = args.log_path / args.varnet_type / "uncertainty"
    parser = UncertaintyModule.add_model_specific_args(parser)

    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--chans", type=int, default=32)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--sens_chans", type=int, default=8)

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        devices=num_gpus,
        replace_sampler_ddp=True,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        seed=42,
        default_root_dir=default_root_dir,
        max_steps=210000,
        detect_anomaly=False,
        gradient_clip_val=1.0,
        mask_type="equispaced_fraction",
        batch_size=1,
        test_path=None,
        combine_train_val=True,
        varnet_type=args.varnet_type,
    )

    args = parser.parse_args()

    if args.varnet_ckpt_prefix is None:
        args.varnet_ckpt_prefix = "fi_varnet."

    args.logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=f"{args.model_name}",
    )

    checkpoint_dir = args.default_root_dir / "checkpoints" / f"{args.model_name}"
    if not dist.is_initialized() or dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return args


def run_cli():
    args = build_args(cluster_launch=True)
    cli_main(args)


if __name__ == "__main__":
    run_cli()
