from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch

from data.undersampling_patterns import MaskFunc
from utilities.functions import apply_mask, to_tensor


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


class VarNetDataTransform:
    """Transform a fastMRI slice for fixed- or learnable-mask VarNet training."""

    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        learnable_mask: bool = False,
    ):
        if learnable_mask and mask_func is not None:
            raise ValueError("learnable_mask=True requires mask_func=None")

        self.mask_func = mask_func
        self.use_seed = use_seed
        self.learnable_mask = learnable_mask

    @staticmethod
    def _build_acquisition_support_mask(
        full_kspace: torch.Tensor,
        acq_start: int,
        acq_end: int,
    ) -> torch.Tensor:
        num_cols = full_kspace.shape[-2]
        mask_shape = [1] * full_kspace.ndim
        mask_shape[-2] = num_cols

        support_mask = torch.zeros(*mask_shape, dtype=torch.float32)
        support_mask[..., acq_start:acq_end, :] = 1.0
        return support_mask

    def __call__(
        self,
        kspace: np.ndarray,
        mask: Optional[np.ndarray],
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

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])
        acq_start = int(attrs["padding_left"])
        acq_end = int(attrs["padding_right"])

        if self.learnable_mask:
            mask_torch = self._build_acquisition_support_mask(
                full_kspace,
                acq_start,
                acq_end,
            )
            masked_kspace = full_kspace
            num_low_frequencies = 0

        elif self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                full_kspace,
                self.mask_func,
                seed=seed,
                padding=(acq_start, acq_end),
            )

        else:
            if mask is None:
                raise ValueError(
                    "No stored mask was found. Provide mask_func for synthetic fixed "
                    "undersampling or use learnable_mask=True."
                )

            masked_kspace = full_kspace
            num_cols = full_kspace.shape[-2]
            mask_shape = [1] * full_kspace.ndim
            mask_shape[-2] = num_cols

            mask_torch = torch.from_numpy(
                mask.reshape(*mask_shape).astype(np.float32)
            )
            mask_torch[..., :acq_start, :] = 0
            mask_torch[..., acq_end:, :] = 0
            num_low_frequencies = 0

        return VarNetSample(
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
