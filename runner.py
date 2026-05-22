import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_module import DataModule
from data.data_transforms import VarNetDataTransform, VolumeVarNetDataTransform
from data.undersampling_patterns import create_mask_for_mask_type
from varnets.VarNet_module import FIVarNetModule
from varnets.VarNet_module_3D import VarNetModule3D
from varnets.VarNet import (
    E2EVarNet,
    E2EVarNet3D,
    FIVarNet,
    LearnableMaskedVarNet,
    LearnableMaskedVarNet3D,
)

torch.set_float32_matmul_precision("high")


class GPUMemoryPrintCallback(pl.Callback):
    def __init__(self, every_n_steps: int = 10, print_all_ranks: bool = True):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.print_all_ranks = print_all_ranks

    def _should_print(self, trainer) -> bool:
        if self.every_n_steps <= 0:
            return False

        step = trainer.global_step
        if step == 0:
            return False

        if step % self.every_n_steps != 0:
            return False

        if not torch.cuda.is_available():
            return False

        if self.print_all_ranks:
            return True

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0

        return True

    def _print_memory(self, trainer, stage: str):
        if not self._should_print(trainer):
            return

        device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

        free, total = torch.cuda.mem_get_info(device)
        free = free / 1024**3
        total = total / 1024**3
        used_total = total - free

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(
            f"[GPU MEM][{stage}] "
            f"step={trainer.global_step} "
            f"rank={rank} local_rank={local_rank} cuda={device} | "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB, "
            f"max_allocated={max_allocated:.2f} GB, "
            f"max_reserved={max_reserved:.2f} GB, "
            f"device_used={used_total:.2f}/{total:.2f} GB",
            flush=True,
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        unused: int = 0,
    ):
        self._print_memory(trainer, stage="train")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ):
        self._print_memory(trainer, stage="val")

# ============================================================
# UTILS
# ============================================================
def check_gpu_availability():
    command = "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l"
    output = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    return int(output)


def _is_volume_mode(args) -> bool:
    return getattr(args, "dataset_format", "slice") == "volume"


def _adapt_state_dict_for_model(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Adapts checkpoint keys for wrapped learnable-mask models.

    Case handled:
      source checkpoint = plain VarNet
      target model      = LearnableMaskedVarNet(base_varnet=...)

    Then we remap:
      conv.weight  -> base_varnet.conv.weight
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
    module_name: str = "fi_varnet.",
) -> dict:
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    filtered = _filter_state_dict_by_prefix(state_dict, module_name)

    # 3D checkpoints are usually saved under "varnet.". The 3D module also
    # exposes a compatibility alias "fi_varnet", so accept both prefixes.
    if len(filtered) == 0 and module_name == "fi_varnet.":
        filtered = _filter_state_dict_by_prefix(state_dict, "varnet.")
    elif len(filtered) == 0 and module_name == "varnet.":
        filtered = _filter_state_dict_by_prefix(state_dict, "fi_varnet.")

    if len(filtered) == 0:
        raise RuntimeError(
            f"No VarNet weights found in {weight_path} using prefix '{module_name}'. "
            "Tried compatibility prefixes 'fi_varnet.' and 'varnet.' as applicable."
        )

    return filtered


def load_weights_only(
    module,
    weight_path: Path,
    module_name: str = "fi_varnet.",
):
    filtered = _get_varnet_state_dict_from_checkpoint(
        weight_path=weight_path,
        module_name=module_name,
    )
    filtered = _adapt_state_dict_for_model(module.fi_varnet, filtered)

    missing, unexpected = module.fi_varnet.load_state_dict(filtered, strict=False)
    print("load_weights_only:")
    print("  missing keys   :", len(missing))
    print("  unexpected keys:", len(unexpected))

    return module


def reload_state_dict(
    module,
    fname: Path,
    module_name: str = "fi_varnet.",
):
    print(f"loading model from {fname}")

    state_dict = _get_varnet_state_dict_from_checkpoint(
        weight_path=fname,
        module_name=module_name,
    )
    state_dict = _adapt_state_dict_for_model(module.fi_varnet, state_dict)
    module.fi_varnet.load_state_dict(state_dict, strict=False)

    return module


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
    if _is_volume_mode(args):
        if args.varnet_type != "e2e_varnet":
            raise ValueError(
                "dataset_format='volume' currently supports only varnet_type='e2e_varnet'. "
                "FIVarNet3D is not implemented."
            )

        print(f"BUILDING 3D E2E VARNET, chans={args.chans}")

        if args.freeze_sens_net and args.sens_ckpt is None:
            raise ValueError(
                "--freeze_sens_net requires --sens_ckpt. "
                "Do not freeze a randomly initialized sensitivity model."
            )

        return E2EVarNet3D(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            kernel_size=tuple(args.kernel_size),
            pool_kernel_size=tuple(args.pool_kernel_size),
            sens_ckpt=None if args.sens_ckpt is None else str(args.sens_ckpt),
            freeze_sens_net=args.freeze_sens_net,
            strict_sens_load=not args.non_strict_sens_load,
        )

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


def fetch_model(args):
    if args.mask_mode == "learnable":
        acceleration, center_fraction = _get_single_learnable_mask_params(args)
        base_varnet = build_base_varnet(args, acceleration=acceleration)

        if _is_volume_mode(args):
            print(
                f"WRAPPING 3D E2E VARNET WITH LEARNABLE 2D CARTESIAN MASK, "
                f"acceleration={acceleration}, center_fraction={center_fraction}, "
                f"num_slice_logits={args.num_slice_logits}, "
                f"num_phase_logits={args.num_logits}"
            )

            return LearnableMaskedVarNet3D(
                base_varnet=base_varnet,
                acceleration=acceleration,
                center_fraction=center_fraction,
                num_phase_logits=args.num_logits,
                num_slice_logits=args.num_slice_logits,
            )

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


def fetch_lightning_module(args):
    model = fetch_model(args)

    if _is_volume_mode(args):
        return VarNetModule3D(
            varnet=model,
            learnable_mask=(args.mask_mode == "learnable"),
            lr=args.lr,
            lr_base=args.lr_base,
            lr_mask=args.lr_mask,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            model_name=args.model_name,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            ramp_steps=args.ramp_steps,
            cosine_decay_start=args.cosine_decay_start,
            drop_prob=args.drop_prob,
            log_slice=args.log_slice,
            num_log_images=args.num_log_images,
        )

    return FIVarNetModule(
        fi_varnet=model,
        learnable_mask=(args.mask_mode == "learnable"),
        lr=args.lr,
        lr_base=args.lr_base,
        lr_mask=args.lr_mask,
        model_name=args.model_name,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        ramp_steps=args.ramp_steps,
        cosine_decay_start=args.cosine_decay_start,
        yolo_cfg=args.yolo_cfg,
        yolo_repo=args.yolo_repo,
        yolo_weights=args.yolo_weights,
        yolo_flag=args.yolo_flag,
        device=args.device,
        imgsz=tuple(args.imgsz),
        half=args.half,
        pl_weight=args.pl_weight,
        feature_layers=args.feature_layers,
        layer_weights=args.layer_weights,
    )


# ============================================================
# TRANSFORMS
# ============================================================
def build_transforms(args):
    transform_cls = VolumeVarNetDataTransform if _is_volume_mode(args) else VarNetDataTransform

    if args.mask_mode == "learnable":
        train_transform = transform_cls(
            mask_func=None,
            use_seed=False,
            learnable_mask=True,
        )
        val_transform = transform_cls(
            mask_func=None,
            use_seed=True,
            learnable_mask=True,
        )
        test_transform = transform_cls(
            mask_func=None,
            use_seed=True,
            learnable_mask=True,
        )
        return train_transform, val_transform, test_transform

    mask = create_mask_for_mask_type(
        args.mask_type,
        args.center_fractions,
        args.accelerations,
    )

    train_transform = transform_cls(mask_func=mask, use_seed=False)
    val_transform = transform_cls(mask_func=mask)

    if args.mode == "test_val":
        test_transform = transform_cls(mask_func=mask)
    else:
        test_transform = transform_cls()

    return train_transform, val_transform, test_transform


# ============================================================
# TRAIN / TEST
# ============================================================
def cli_main(args):
    pl.seed_everything(args.seed)

    original_mode = args.mode
    train_transform, val_transform, test_transform = build_transforms(args)

    if original_mode == "test_val":
        args.mode = "test"
        if hasattr(args, "yolo_flag"):
            args.yolo_flag = 0

    data_module = DataModule(
        data_path=args.data_path,
        data_path_train=args.data_path_train,
        data_path_val=args.data_path_val,
        data_path_cal=args.data_path_cal,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=args.combine_train_val,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        val_sample_rate=args.val_sample_rate,
        test_sample_rate=args.test_sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        val_volume_sample_rate=args.val_volume_sample_rate,
        test_volume_sample_rate=args.test_volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        dataset_format=args.dataset_format,
    )

    pl_module = fetch_lightning_module(args)
    resume_ckpt = args.resume_from_checkpoint

    if args.mode == "fine_tune":
        ckpt_prefix = "varnet." if _is_volume_mode(args) else "fi_varnet."
        pl_module = load_weights_only(
            pl_module,
            Path(args.fine_tune_ckpt),
            module_name=ckpt_prefix,
        )
        ckpt_path = None
    elif args.mode == "train":
        ckpt_path = resume_ckpt if resume_ckpt is not None else None
    elif args.mode == "test":
        ckpt_path = resume_ckpt if resume_ckpt is not None else None
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    args.resume_from_checkpoint = None
    if _is_volume_mode(args): args.log_every_n_steps = 1
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=args.callbacks,
    )

    if args.mode in ["train", "fine_tune"]:
        trainer.fit(
            pl_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
    elif args.mode == "test":
        trainer.test(
            pl_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )


# ============================================================
# ARGS
# ============================================================
def build_args(cluster_launch: bool = True):
    num_gpus = check_gpu_availability()

    parser = ArgumentParser()
    parser.add_argument("--log_path", type=Path, default=None)
    parser.add_argument("--model_name", type=str, default="VarNet")
    parser.add_argument("--fine_tune_ckpt", type=str, default=None)
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test", "test_val", "fine_tune"),
        type=str,
    )

    parser.add_argument(
        "--mask_mode",
        choices=("fixed", "learnable"),
        default="fixed",
        type=str,
    )

    parser.add_argument(
        "--mask_type",
        choices=("equispaced_fraction", "equispaced", "edge_dominant"),
        default="equispaced_fraction",
        type=str,
    )

    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int)
    # For 1D learnable masks, --num_logits is the canonical ky resolution.
    # For 2D multislice learnable masks, --num_logits is the phase-axis
    # resolution and --num_slice_logits is the slice-axis resolution.
    parser.add_argument("--num_logits", type=int, default=320)
    parser.add_argument("--num_slice_logits", type=int, default=40)

    # GPU memory debug printing. Set to 0 to disable.
    parser.add_argument("--gpu_mem_log_every", type=int, default=0)
    parser.add_argument("--gpu_mem_log_all_ranks", action="store_true")

    parser.add_argument(
        "--varnet_type",
        choices=("fi_varnet", "e2e_varnet"),
        default="fi_varnet",
    )

    # 2D-only YOLO/perceptual options. These are ignored in volume mode.
    parser.add_argument("--yolo_repo", type=str, default="path/to/yolo/")
    parser.add_argument("--yolo_cfg", type=str, default="path/to/yolo.yaml")
    parser.add_argument("--yolo_weights", type=str, default="path/to/weights.pt")
    parser.add_argument("--yolo_flag", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--pl_weight", type=float, default=1.0)
    parser.add_argument("--feature_layers", nargs="+", default=None)
    parser.add_argument("--layer_weights", nargs="+", type=float, default=None)

    parser = DataModule.add_data_specific_args(parser)

    # First pass: decide which model module owns the architecture args and log root.
    args, _ = parser.parse_known_args()
    if args.log_path is None:
        raise ValueError("--log_path must be provided.")

    if args.dataset_format == "volume":
        default_root_dir = args.log_path / "e2e_varnet_3d"
        module_arg_adder = VarNetModule3D.add_model_specific_args
    elif args.varnet_type == "e2e_varnet":
        default_root_dir = args.log_path / "e2e_varnet"
        module_arg_adder = FIVarNetModule.add_model_specific_args
    elif args.varnet_type == "fi_varnet":
        default_root_dir = args.log_path / "fi_varnet"
        module_arg_adder = FIVarNetModule.add_model_specific_args
    else:
        raise ValueError(f"Unknown varnet_type: {args.varnet_type}")

    parser.set_defaults(
        mask_type="equispaced_fraction",
        batch_size=1,
        test_path=None,
        combine_train_val=True,
        varnet_type="e2e_varnet" if args.dataset_format == "volume" else args.varnet_type,
    )

    parser = module_arg_adder(parser)

    # ------------------------------------------------------------ #
    # -------- optional frozen 2D sensitivity for 3D VarNet --------#
    # ------------------------------------------------------------ #
    parser.add_argument(
        "--sens_ckpt",
        type=Path,
        default=None,
        help="Optional 2D VarNet checkpoint used to initialize the 3D slice-wise sensitivity model.",
    )

    parser.add_argument(
        "--freeze_sens_net",
        action="store_true",
        help="Freeze the 3D slice-wise sensitivity model and run it without gradients.",
    )

    parser.add_argument(
        "--non_strict_sens_load",
        action="store_true",
        help="Load sensitivity weights with strict=False.",
    )

    # Shared reconstruction defaults. 3D module ignores lr_base/lr_mask/YOLO fields.
    parser.set_defaults(
        num_cascades=12,
        pools=4,
        chans=32,
        sens_pools=4,
        sens_chans=8,
        lr=0.0003,
        lr_base=None,
        lr_mask=None,
        ramp_steps=7500,
        cosine_decay_start=150000,
        weight_decay=0.0,
    )

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
    )

    args = parser.parse_args()
    args.logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=f"{args.model_name}",
    )
    checkpoint_dir = args.default_root_dir / "checkpoints" / f"{args.model_name}"

    if not dist.is_initialized() or dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,
            save_top_k=1,
            verbose=True,
            monitor="val/loss",
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(),
    ]

    if args.gpu_mem_log_every > 0:
        args.callbacks.append(
            GPUMemoryPrintCallback(
                every_n_steps=args.gpu_mem_log_every,
                print_all_ranks=args.gpu_mem_log_all_ranks,
            )
        )

    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args(cluster_launch=True)
    cli_main(args)


if __name__ == "__main__":
    run_cli()