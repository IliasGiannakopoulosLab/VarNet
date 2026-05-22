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



# -------------------------------------------#
# ---------- volume varnet sample ---------- #
# -------------------------------------------#
class VolumeVarNetSample(NamedTuple):
    full_kspace: torch.Tensor
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    max_value: float
    crop_size: Tuple[int, int]
    acq_start: int
    acq_end: int


# -------------------------------------------#
# ------- volume varnet data transform ------#
# -------------------------------------------#
class VolumeVarNetDataTransform:
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
    # ---- full-acquisition mask helper 3d ------#
    # -------------------------------------------#
    def _build_full_acq_mask_3d(
        self,
        kspace_torch: torch.Tensor,
        acq_start: int,
        acq_end: int,
    ) -> torch.Tensor:
        """
        Input:
            kspace_torch: [C, D, H, W, 2]

        Output:
            mask: [1, D, 1, W, 1]
        """

        num_slices = kspace_torch.shape[1]
        num_cols = kspace_torch.shape[-2]

        mask_torch = torch.ones(
            1,
            num_slices,
            1,
            num_cols,
            1,
            dtype=torch.float32,
        )

        mask_torch[..., :acq_start, :] = 0
        mask_torch[..., acq_end:, :] = 0

        return mask_torch

    # -------------------------------------------#
    # -------- fixed shared 1d mask 3d ----------#
    # -------------------------------------------#
    def _apply_fixed_mask_3d(
        self,
        full_kspace: torch.Tensor,
        acq_start: int,
        acq_end: int,
        seed: Optional[Tuple[int, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Apply one fixed 1D Cartesian ky mask to the full volume.

        Input:
            full_kspace: [C, D, H, W, 2]

        Output:
            masked_kspace:       [C, D, H, W, 2]
            mask_torch:          [1, D, 1, W, 1]
            num_low_frequencies: int

        Notes:
            - The same ky mask is shared by all slices.
            - This matches standard multislice Cartesian undersampling.
            - The network still sees a 3D/multislice volume, but the sampling
              pattern is not artificially different slice-by-slice.
        """

        if self.mask_func is None:
            raise ValueError("mask_func must be provided for fixed-mask 3D mode.")

        num_slices = full_kspace.shape[1]
        num_cols = full_kspace.shape[-2]

        # Shape expected by MaskFunc for one 2D slice:
        #     [1, H, W, 2]
        # It returns:
        #     [1, 1, W, 1]
        slice_shape = (1,) + tuple(full_kspace.shape[-3:])

        # One mask per volume/file, shared across all slices.
        mask_slice, num_low_frequencies = self.mask_func(
            slice_shape,
            seed=seed,
        )

        mask_slice[..., :acq_start, :] = 0
        mask_slice[..., acq_end:, :] = 0

        # [1, 1, W, 1] -> [1, D, 1, W, 1]
        mask_torch = mask_slice.expand(
            1,
            num_slices,
            1,
            num_cols,
            1,
        ).clone()

        masked_kspace = full_kspace * mask_torch + 0.0

        return masked_kspace, mask_torch, num_low_frequencies

    # -------------------------------------------#
    # ------------------ call ------------------ #
    # -------------------------------------------#
    def __call__(
        self,
        kspace: np.ndarray,
        mask: Optional[np.ndarray],
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
    ) -> VolumeVarNetSample:

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        # H5 volume kspace is expected as:
        #     [D, C, H, W]
        #
        # to_tensor gives:
        #     [D, C, H, W, 2]
        #
        # model expects:
        #     [C, D, H, W, 2]
        full_kspace = to_tensor(kspace).permute(1, 0, 2, 3, 4).contiguous()

        seed = None if not self.use_seed else tuple(map(ord, fname))

        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # ------------------------------------------------ #
        # learnable-mask mode: we do NOT undersample here  #
        # ------------------------------------------------ #
        if self.learnable_mask:

            full_acq_mask = self._build_full_acq_mask_3d(
                full_kspace,
                acq_start,
                acq_end,
            )

            sample = VolumeVarNetSample(
                full_kspace=full_kspace,
                masked_kspace=full_kspace,          # placeholder
                mask=full_acq_mask.to(torch.bool),  # acquisition support only
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                max_value=max_value,
                crop_size=crop_size,
                acq_start=acq_start,
                acq_end=acq_end,
            )

        # ------------------------------------------------ #
        # fixed-mask mode: 1D mask per slice, stacked D×ky #
        # ------------------------------------------------ #
        elif self.mask_func is not None:

            masked_kspace, mask_torch, num_low_frequencies = self._apply_fixed_mask_3d(
                full_kspace=full_kspace,
                acq_start=acq_start,
                acq_end=acq_end,
                seed=seed,
            )

            sample = VolumeVarNetSample(
                full_kspace=full_kspace,
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                max_value=max_value,
                crop_size=crop_size,
                acq_start=acq_start,
                acq_end=acq_end,
            )

        # ------------------------------------------------ #
        # no mask_func path: use stored H5 mask if present  #
        # ------------------------------------------------ #
        else:

            if mask is not None:
                mask_np = np.asarray(mask).astype(np.float32)
                num_slices = full_kspace.shape[1]
                num_cols = full_kspace.shape[-2]

                if mask_np.ndim == 1:
                    # Stored fastMRI-style mask shared by all slices: [W].
                    mask_np = np.broadcast_to(mask_np[None, :], (num_slices, num_cols))
                elif mask_np.ndim == 2:
                    # Already slice-dependent: [D, W].
                    if mask_np.shape != (num_slices, num_cols):
                        raise ValueError(
                            f"Stored 3D mask has shape {mask_np.shape}, expected "
                            f"({num_slices}, {num_cols})."
                        )
                else:
                    raise ValueError(
                        f"Stored 3D mask must have shape [W] or [D, W], got {mask_np.shape}."
                    )

                mask_torch = torch.from_numpy(mask_np).float()
                mask_torch[:, :acq_start] = 0
                mask_torch[:, acq_end:] = 0
                mask_torch = mask_torch[None, :, None, :, None]

                masked_kspace = full_kspace * mask_torch + 0.0

            else:
                # Fallback for data that is already masked but does not store an
                # explicit sampling mask. This preserves the old behavior.
                mask_torch = self._build_full_acq_mask_3d(
                    full_kspace,
                    acq_start,
                    acq_end,
                )
                masked_kspace = full_kspace

            sample = VolumeVarNetSample(
                full_kspace=full_kspace,
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                max_value=max_value,
                crop_size=crop_size,
                acq_start=acq_start,
                acq_end=acq_end,
            )

        return sample