from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.fft
from torch import Tensor

from data.undersampling_patterns import MaskFunc


# -------------------------------------------#
# --------------- complex mul -------------- #
# -------------------------------------------#
def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


# -------------------------------------------#
# -------------- complex conj -------------- #
# -------------------------------------------#
def complex_conj(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


# -------------------------------------------#
# -------------- complex abs --------------- #
# -------------------------------------------#
def complex_abs(data: torch.Tensor) -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()


# -------------------------------------------#
# ------------ complex abs sq -------------- #
# -------------------------------------------#
def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


# -------------------------------------------#
# -------- tensor to complex numpy --------- #
# -------------------------------------------#
def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    return torch.view_as_complex(data).numpy()


# -------------------------------------------#
# ------------------- rss ------------------ #
# -------------------------------------------#
def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.sqrt((data**2).sum(dim))


# -------------------------------------------#
# -------------- rss complex --------------- #
# -------------------------------------------#
def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.sqrt(complex_abs_sq(data).sum(dim))


# -------------------------------------------#
# ---------------- fft2c new --------------- #
# -------------------------------------------#
def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# -------------------------------------------#
# ---------------- ifft2c new -------------- #
# -------------------------------------------#
def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# -------------------------------------------#
# -------------- roll one dim -------------- #
# -------------------------------------------#
def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


# -------------------------------------------#
# ------------------- roll ----------------- #
# -------------------------------------------#
def roll(x: torch.Tensor,shift: List[int],dim: List[int],) -> torch.Tensor:

    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


# -------------------------------------------#
# ---------------- fftshift ---------------- #
# -------------------------------------------#
def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


# -------------------------------------------#
# --------------- ifftshift ---------------- #
# -------------------------------------------#
def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


# -------------------------------------------#
# ------------- save reconstructions ------- #
# -------------------------------------------#
def save_reconstructions(reconstructions: Dict[str, dict], out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)

    for fname, item in reconstructions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction", data=item["reconstruction"])

            if "mask_1d" in item:
                hf.create_dataset("mask_1d", data=item["mask_1d"])

            if "support_width" in item:
                hf.create_dataset("support_width", data=item["support_width"])

            if "effective_acceleration" in item:
                hf.create_dataset("effective_acceleration", data=item["effective_acceleration"])

            if "acq_start" in item:
                hf.create_dataset("acq_start", data=item["acq_start"])

            if "acq_end" in item:
                hf.create_dataset("acq_end", data=item["acq_end"])


# -------------------------------------------#
# -------- convert fnames to v2 ------------ #
# -------------------------------------------#
def convert_fnames_to_v2(path: Path):
    if not path.exists():
        raise ValueError("Path does not exist")

    for fname in path.glob("*.h5"):
        if not fname.name.endswith("_v2.h5"):
            fname.rename(path / (fname.stem + "_v2.h5"))


# -------------------------------------------#
# --------------- normalize image ---------- #
# -------------------------------------------#
def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    return image


# -------------------------------------------#
# ------------------ to tensor ------------- #
# -------------------------------------------#
def to_tensor(data: np.ndarray) -> torch.Tensor:
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


# -------------------------------------------#
# ------------------ apply mask ------------ #
# -------------------------------------------#
def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:

    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)

    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0

    masked_data = data * mask + 0.0

    return masked_data, mask, num_low_frequencies


# -------------------------------------------#
# ----------------- mask center ------------ #
# -------------------------------------------#
def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
    return mask


# -------------------------------------------#
# ------------- batched mask center -------- #
# -------------------------------------------#
def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:

    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if mask_from.shape[0] != 1:
        if (x.shape[0] != mask_from.shape[0]) or (
            x.shape[0] != mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


# -------------------------------------------#
# ---------------- center crop ------------- #
# -------------------------------------------#
def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:

    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2

    return data[..., w_from:w_from + shape[0], h_from:h_from + shape[1]]


# -------------------------------------------#
# ----------- complex center crop ---------- #
# -------------------------------------------#
def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:

    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2

    return data[..., w_from:w_from + shape[0], h_from:h_from + shape[1], :]


# -------------------------------------------#
# -------- center crop to smallest --------- #
# -------------------------------------------#
def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])

    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


# -------------------------------------------#
# ------------------ normalize ------------- #
# -------------------------------------------#
def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    return (data - mean) / (stddev + eps)


# -------------------------------------------#
# ------------ normalize instance ---------- #
# -------------------------------------------#
def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


# -------------------------------------------#
# ---------------- image crop -------------- #
# -------------------------------------------#
def image_crop(image: Tensor, crop_size: Optional[Tuple[int, int]] = None) -> Tensor:
    if crop_size is None:
        return image
    return center_crop(image, crop_size).contiguous()


# -------------------------------------------#
# --------------- calc uncrop -------------- #
# -------------------------------------------#
def _calc_uncrop(crop_height: int, in_height: int) -> Tuple[int, int]:
    pad_height = (in_height - crop_height) // 2
    if (in_height - crop_height) % 2 != 0:
        pad_height_top = pad_height + 1
    else:
        pad_height_top = pad_height

    pad_height = in_height - pad_height
    return pad_height_top, pad_height


# -------------------------------------------#
# --------------- image uncrop ------------- #
# -------------------------------------------#
def image_uncrop(image: Tensor, original_image: Tensor) -> Tensor:
    in_shape = original_image.shape
    original_image = original_image.clone()

    if in_shape == image.shape:
        return image

    pad_height_top, pad_height = _calc_uncrop(image.shape[-2], in_shape[-2])
    pad_height_left, pad_width = _calc_uncrop(image.shape[-1], in_shape[-1])

    try:
        original_image[
            ..., pad_height_top:pad_height, pad_height_left:pad_width
        ] = image[...]
    except RuntimeError:
        print(f"in_shape: {in_shape}, image shape: {image.shape}")
        raise

    return original_image


# -------------------------------------------#
# ------------------- norm fn -------------- #
# -------------------------------------------#
def norm_fn(image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    means = means.view(1, -1, 1, 1)
    variances = variances.view(1, -1, 1, 1)
    return (image - means) * torch.rsqrt(variances)


# -------------------------------------------#
# ------------------ unnorm fn ------------- #
# -------------------------------------------#
def unnorm_fn(image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    means = means.view(1, -1, 1, 1)
    variances = variances.view(1, -1, 1, 1)
    return image * torch.sqrt(variances) + means


# -------------------------------------------#
# ----------- complex to chan dim ---------- #
# -------------------------------------------#
def complex_to_chan_dim(x: Tensor) -> Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    assert c == 1
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


# -------------------------------------------#
# ------- chan complex to last dim --------- #
# -------------------------------------------#
def chan_complex_to_last_dim(x: Tensor) -> Tensor:
    b, c2, h, w = x.shape
    assert c2 == 2
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


# -------------------------------------------#
# --------------- sens expand -------------- #
# -------------------------------------------#
def sens_expand(x: Tensor, sens_maps: Tensor) -> Tensor:
    return fft2c_new(complex_mul(chan_complex_to_last_dim(x), sens_maps))


# -------------------------------------------#
# --------------- sens reduce -------------- #
# -------------------------------------------#
def sens_reduce(x: Tensor, sens_maps: Tensor) -> Tensor:
    return complex_to_chan_dim(
        complex_mul(ifft2c_new(x), complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
    )
