from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCartesianMask(nn.Module):
    """
    Learnable 1D Cartesian phase-encoding mask.

    The mask is learned on a canonical 1D support and interpolated when the
    acquired phase-encoding support has a different width. A contiguous center
    region remains fully sampled. The outer probabilities are renormalized to
    the sampling ratio implied by ``acceleration`` and ``center_fraction``.

    Training uses Bernoulli sampling with a straight-through estimator.
    Evaluation uses a deterministic top-k mask with the exact rounded sampling
    budget. This guarantees that reconstruction testing and uncertainty
    inference use the same learned sampling pattern.
    """

    def __init__(
        self,
        acceleration: int,
        center_fraction: float,
        num_logits: int = 320,
        init_bias: Optional[float] = None,
    ):
        super().__init__()

        if acceleration <= 0:
            raise ValueError("acceleration must be positive")
        if not (0.0 < center_fraction < 1.0):
            raise ValueError("center_fraction must be in (0, 1)")
        if num_logits <= 0:
            raise ValueError("num_logits must be positive")

        self.acceleration = int(acceleration)
        self.center_fraction = float(center_fraction)
        self.num_logits = int(num_logits)
        self.prob_slope = 0.25

        if init_bias is None:
            initial_parameters = torch.empty(num_logits).uniform_(-15.0, 15.0)
        else:
            initial_parameters = torch.full((num_logits,), float(init_bias))

        self.mask_parameters = nn.Parameter(initial_parameters)

    def _resize_parameter_map(self, support_width: int) -> torch.Tensor:
        if support_width <= 0:
            raise ValueError("support_width must be positive")

        if support_width == self.num_logits:
            return self.mask_parameters

        parameters = self.mask_parameters.view(1, 1, -1)
        parameters = F.interpolate(
            parameters,
            size=support_width,
            mode="linear",
            align_corners=True,
        )
        return parameters.view(-1)

    def _build_center_mask(
        self,
        support_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int, int, int]:
        num_low_frequencies = max(1, round(support_width * self.center_fraction))
        num_low_frequencies = min(num_low_frequencies, support_width)

        center_mask = torch.zeros(support_width, device=device, dtype=dtype)
        center_start = (support_width - num_low_frequencies + 1) // 2
        center_end = center_start + num_low_frequencies
        center_mask[center_start:center_end] = 1.0

        return center_mask, num_low_frequencies, center_start, center_end

    @staticmethod
    def _embed_support_mask(
        support_mask: torch.Tensor,
        full_width: int,
        acq_start: int,
        acq_end: int,
    ) -> torch.Tensor:
        full_mask = support_mask.new_zeros(full_width)
        full_mask[acq_start:acq_end] = support_mask
        return full_mask

    @staticmethod
    def _renormalize_probabilities(
        probabilities: torch.Tensor,
        target_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly renormalize probabilities to the requested mean."""
        if probabilities.numel() == 0:
            return probabilities

        eps = torch.finfo(probabilities.dtype).eps
        target_mean = torch.clamp(target_mean, 0.0, 1.0)
        current_mean = probabilities.mean()

        if target_mean <= 0:
            return torch.zeros_like(probabilities)
        if target_mean >= 1:
            return torch.ones_like(probabilities)

        if current_mean > target_mean:
            output = probabilities * target_mean / (current_mean + eps)
        else:
            output = 1.0 - (1.0 - probabilities) * (
                (1.0 - target_mean) / (1.0 - current_mean + eps)
            )

        return torch.clamp(output, 0.0, 1.0)

    @staticmethod
    def _deterministic_outer_mask(
        probabilities: torch.Tensor,
        num_outer_samples: int,
    ) -> torch.Tensor:
        """Select the highest-probability outer lines with an exact budget."""
        hard_mask = torch.zeros_like(probabilities)
        if num_outer_samples <= 0:
            return hard_mask
        if num_outer_samples >= probabilities.numel():
            return torch.ones_like(probabilities)

        selected = torch.topk(
            probabilities,
            k=num_outer_samples,
            largest=True,
            sorted=False,
        ).indices
        hard_mask[selected] = 1.0
        return hard_mask

    def forward(
        self,
        full_kspace: torch.Tensor,
        acq_start: Union[int, torch.Tensor],
        acq_end: Union[int, torch.Tensor],
    ):
        if full_kspace.ndim != 5 or full_kspace.shape[-1] != 2:
            raise ValueError(
                "Expected full_kspace with shape [batch, coils, height, width, 2]"
            )

        device = full_kspace.device
        batch_size = full_kspace.shape[0]
        full_width = full_kspace.shape[-2]

        if isinstance(acq_start, int):
            acq_start = torch.full(
                (batch_size,), acq_start, device=device, dtype=torch.long
            )
        else:
            acq_start = acq_start.to(device=device, dtype=torch.long).view(-1)

        if isinstance(acq_end, int):
            acq_end = torch.full(
                (batch_size,), acq_end, device=device, dtype=torch.long
            )
        else:
            acq_end = acq_end.to(device=device, dtype=torch.long).view(-1)

        if acq_start.numel() != batch_size or acq_end.numel() != batch_size:
            raise ValueError("acq_start and acq_end must have batch_size elements")

        sampled_hard_full = []
        sampled_st_full = []
        probability_full = []
        center_full = []
        outer_probability_means = []
        raw_outer_probability_means = []
        support_widths = []
        sampled_lines = []
        effective_accelerations = []
        num_low_frequencies_per_sample = []
        center_starts = []
        center_ends = []
        target_total_ratios = []
        target_outer_means = []

        for batch_index in range(batch_size):
            start = int(acq_start[batch_index].item())
            end = int(acq_end[batch_index].item())
            support_width = end - start

            if start < 0 or end > full_width or support_width <= 0:
                raise ValueError(
                    f"Invalid acquisition support [{start}, {end}) for width {full_width}"
                )

            resized_parameters = self._resize_parameter_map(support_width)
            center_mask, num_low_frequencies, center_start, center_end = (
                self._build_center_mask(
                    support_width=support_width,
                    device=device,
                    dtype=resized_parameters.dtype,
                )
            )

            target_total_samples = round(support_width / self.acceleration)
            target_total_samples = max(1, min(support_width, target_total_samples))
            if num_low_frequencies > target_total_samples:
                raise ValueError(
                    "The fully sampled center exceeds the total sampling budget: "
                    f"support_width={support_width}, acceleration={self.acceleration}, "
                    f"center_fraction={self.center_fraction}, "
                    f"center_lines={num_low_frequencies}, "
                    f"budget_lines={target_total_samples}."
                )
            target_outer_samples = target_total_samples - num_low_frequencies

            outer_indices = center_mask == 0
            outer_count = int(outer_indices.sum().item())
            target_outer_mean = (
                float(target_outer_samples) / float(outer_count)
                if outer_count > 0
                else 0.0
            )

            probability_support = center_mask.clone()
            hard_support = center_mask.clone()
            straight_through_support = center_mask.clone()

            if outer_count > 0:
                raw_outer_probabilities = torch.sigmoid(
                    self.prob_slope * resized_parameters[outer_indices]
                )
                outer_probabilities = self._renormalize_probabilities(
                    raw_outer_probabilities,
                    raw_outer_probabilities.new_tensor(target_outer_mean),
                )

                if self.training:
                    random_values = torch.rand_like(outer_probabilities)
                    hard_outer = (random_values < outer_probabilities).to(
                        outer_probabilities.dtype
                    )
                else:
                    hard_outer = self._deterministic_outer_mask(
                        outer_probabilities,
                        num_outer_samples=target_outer_samples,
                    )

                straight_through_outer = (
                    hard_outer
                    + outer_probabilities
                    - outer_probabilities.detach()
                )

                probability_support[outer_indices] = outer_probabilities
                hard_support[outer_indices] = hard_outer
                straight_through_support[outer_indices] = straight_through_outer

                raw_outer_probability_means.append(raw_outer_probabilities.mean())
                outer_probability_means.append(outer_probabilities.mean())
            else:
                zero = resized_parameters.new_tensor(0.0)
                raw_outer_probability_means.append(zero)
                outer_probability_means.append(zero)

            sampled_line_count = hard_support.sum()

            sampled_hard_full.append(
                self._embed_support_mask(
                    hard_support,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                )
            )
            sampled_st_full.append(
                self._embed_support_mask(
                    straight_through_support,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                )
            )
            probability_full.append(
                self._embed_support_mask(
                    probability_support,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                )
            )
            center_full.append(
                self._embed_support_mask(
                    center_mask,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                )
            )

            support_widths.append(float(support_width))
            sampled_lines.append(sampled_line_count)
            effective_accelerations.append(
                resized_parameters.new_tensor(float(support_width))
                / sampled_line_count.clamp_min(1.0)
            )
            num_low_frequencies_per_sample.append(num_low_frequencies)
            center_starts.append(center_start)
            center_ends.append(center_end)
            target_total_ratios.append(float(target_total_samples) / support_width)
            target_outer_means.append(target_outer_mean)

        unique_center_sizes = sorted(set(num_low_frequencies_per_sample))
        if len(unique_center_sizes) != 1:
            raise ValueError(
                "Learnable-mask batches must have one common center size. "
                "Use batch_size=1 when acquisition support widths vary."
            )
        num_low_frequencies = unique_center_sizes[0]

        sampled_hard_full = torch.stack(sampled_hard_full, dim=0)
        sampled_st_full = torch.stack(sampled_st_full, dim=0)
        probability_full = torch.stack(probability_full, dim=0)
        center_full = torch.stack(center_full, dim=0)

        mask_st = sampled_st_full[:, None, None, :, None].to(full_kspace.dtype)
        mask_bool = sampled_hard_full[:, None, None, :, None].to(torch.bool)
        masked_kspace = full_kspace * mask_st

        extras = {
            "prob_mask_1d": probability_full,
            "hard_mask_1d": sampled_hard_full,
            "mask_st_1d": sampled_st_full,
            "center_mask_1d": center_full,
            "support_width": torch.tensor(
                support_widths, device=device, dtype=torch.float32
            ),
            "sampled_lines": torch.stack(sampled_lines).to(
                device=device, dtype=torch.float32
            ),
            "effective_acceleration": torch.stack(effective_accelerations).to(
                device=device, dtype=torch.float32
            ),
            "outer_prob_mean": torch.stack(outer_probability_means).to(
                device=device, dtype=torch.float32
            ),
            "outer_prob_raw_mean": torch.stack(raw_outer_probability_means).to(
                device=device, dtype=torch.float32
            ),
            "target_total_ratio": torch.tensor(
                target_total_ratios, device=device, dtype=torch.float32
            ),
            "target_outer_mean": torch.tensor(
                target_outer_means, device=device, dtype=torch.float32
            ),
            "num_low_frequencies": num_low_frequencies,
            "num_low_frequencies_per_sample": torch.tensor(
                num_low_frequencies_per_sample,
                device=device,
                dtype=torch.float32,
            ),
            "center_start_per_sample": torch.tensor(
                center_starts, device=device, dtype=torch.float32
            ),
            "center_end_per_sample": torch.tensor(
                center_ends, device=device, dtype=torch.float32
            ),
            "mask_dim": "1d",
            "sampling_mode": "stochastic" if self.training else "deterministic",
        }

        return masked_kspace, mask_bool, num_low_frequencies, extras
