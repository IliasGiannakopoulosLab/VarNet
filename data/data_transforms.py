from typing import Dict, NamedTuple, Optional, Tuple
import numpy as np
import torch

from data.undersampling_patterns import MaskFunc
from utilities.functions import apply_mask, to_tensor


# -------------------------------------------#
# --------------- varnet sample ------------ #
# -------------------------------------------#
class VarNetSample(NamedTuple):
    full_kspace: torch.Tensor
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
    acq_start: int
    acq_end: int


# -------------------------------------------#
# ----------- varnet data transform -------- #
# -------------------------------------------#
class VarNetDataTransform:
    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        learnable_mask: bool = False,
    ):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.learnable_mask = learnable_mask

    # -------------------------------------------#
    # -------- full-acquisition mask helper ---- #
    # -------------------------------------------#
    def _build_full_acq_mask(
        self,
        kspace_torch: torch.Tensor,
        acq_start: int,
        acq_end: int,
    ) -> torch.Tensor:
        shape = np.array(kspace_torch.shape)
        num_cols = shape[-2]

        shape[:-3] = 1
        mask_shape = [1] * len(shape)
        mask_shape[-2] = num_cols

        mask_torch = torch.ones(*mask_shape, dtype=torch.float32)
        mask_torch[:, :, :acq_start] = 0
        mask_torch[:, :, acq_end:] = 0

        return mask_torch

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> VarNetSample:

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        full_kspace = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))

        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # ------------------------------------------------ #
        # learnable-mask mode: we do NOT undersample here  #
        # ------------------------------------------------ #
        if self.learnable_mask:
            full_acq_mask = self._build_full_acq_mask(
                full_kspace,
                acq_start,
                acq_end,
            )

            sample = VarNetSample(
                full_kspace=full_kspace,
                masked_kspace=full_kspace,  # placeholder for compatibility
                mask=full_acq_mask.to(torch.bool),  # acquisition support only
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
                acq_start=acq_start,
                acq_end=acq_end,
            )

        # ------------------------------------------------ #
        # fixed-mask training/validation path (current)   #
        # ------------------------------------------------ #
        elif self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                full_kspace,
                self.mask_func,
                seed=seed,
                padding=(acq_start, acq_end),
            )

            sample = VarNetSample(
                full_kspace=full_kspace,
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
                acq_start=acq_start,
                acq_end=acq_end,
            )

        # ------------------------------------------------ #
        # no mask_func path (e.g. test with stored mask)   #
        # ------------------------------------------------ #
        else:
            masked_kspace = full_kspace
            shape = np.array(full_kspace.shape)
            num_cols = shape[-2]

            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols

            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)

            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = VarNetSample(
                full_kspace=full_kspace,
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
                acq_start=acq_start,
                acq_end=acq_end,
            )

        return sample
