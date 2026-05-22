import contextlib
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch


# -------------------------------------------#
# ----------------- temp seed -------------- #
# -------------------------------------------#
@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


# -------------------------------------------#
# ---------------- mask func --------------- #
# -------------------------------------------#
class MaskFunc:

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs
        return mask

    def choose_acceleration(self):
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


# -------------------------------------------#
# ------------ equispaced mask ------------- #
# -------------------------------------------#
class EquiSpacedMaskFunc(MaskFunc):

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


# -------------------------------------------#
# ------- equispaced fraction mask --------- #
# -------------------------------------------#
class EquispacedMaskFractionFunc(MaskFunc):

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask

class EdgeDominantMaskFunc(MaskFunc):
    """
    Edge-dominant variable-density mask:
      - center determined by center_fraction
      - edge-biased adaptive sampling outside center
      - exact acceleration preserved
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        total_samples = round(num_cols / acceleration)

        # remaining samples outside center
        remaining = total_samples - num_low_frequencies

        if remaining <= 0:
            return np.zeros(num_cols, dtype=np.float32)

        left_needed = int(np.ceil(remaining / 2))
        right_needed = int(np.floor(remaining / 2))

        mask = np.zeros(num_cols, dtype=np.float32)

        center_start = (num_cols - num_low_frequencies + 1) // 2
        center_end = center_start + num_low_frequencies

        # ---------------- LEFT SIDE ----------------
        left_positions = []
        pos = 0
        gap = 2

        while len(left_positions) < left_needed and pos < center_start:
            left_positions.append(pos)
            pos += gap
            gap = min(gap + 1, acceleration + 2)

        # fill remaining if needed
        candidate = center_start - 1
        while len(left_positions) < left_needed:
            if candidate not in left_positions:
                left_positions.append(candidate)
            candidate -= 1

        left_positions = sorted(left_positions[:left_needed])

        # ---------------- RIGHT SIDE ----------------
        right_positions = []
        pos = num_cols - 1
        gap = 2

        while len(right_positions) < right_needed and pos >= center_end:
            right_positions.append(pos)
            pos -= gap
            gap = min(gap + 1, acceleration + 2)

        candidate = center_end
        while len(right_positions) < right_needed:
            if candidate not in right_positions:
                right_positions.append(candidate)
            candidate += 1

        right_positions = sorted(right_positions[:right_needed])

        mask[left_positions] = 1.0
        mask[right_positions] = 1.0

        return mask

# -------------------------------------------#
# -------- create mask for mask type ------- #
# -------------------------------------------#
def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:

    if mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "edge_dominant":
        return EdgeDominantMaskFunc(center_fractions, accelerations)
    else:
        raise ValueError(f"{mask_type_str} not supported")
