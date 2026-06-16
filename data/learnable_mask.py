from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCartesianMask(nn.Module):
    """
    MoDL+BS style learnable 1D Cartesian mask.

    What this implements:
      - learn unconstrained 1D mask parameters on a canonical support axis
      - map parameters to probabilities with a sigmoid (slope a = 0.25)
      - renormalize the OUTER probabilities so their mean matches the desired
        outer sampling ratio implied by the target acceleration and fixed center
      - sample a binary outer mask with Bernoulli sampling during training
      - select the highest-probability outer lines during validation and testing
      - use a straight-through estimator for backprop through the binary sample
      - keep the center region fully sampled and fixed
      - adapt one learned canonical profile to variable support widths via 1D interpolation

    Notes
    -----
    - This follows the MoDL+BS logic from Zhang et al. (2020), adapted to 1D.
    - `num_logits` is the canonical 1D support resolution.
    - The 2D/multislice version is intentionally implemented as a separate class
      below: Learnable2DCartesianMask.
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

        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.num_logits = num_logits

        # Sigmoid slope a used in LOUPE / LOUPE-ST papers.
        self.prob_slope = 0.25

        # Unconstrained learnable parameter map over canonical support axis.
        if init_bias is None:
            init = torch.empty(num_logits).uniform_(-15.0, 15.0)
        else:
            init = torch.full((num_logits,), float(init_bias))
        self.mask_parameters = nn.Parameter(init)

    # -------------------------------------------#
    # -------- resize canonical params --------- #
    # -------------------------------------------#
    def _resize_parameter_map_1d(self, support_width: int) -> torch.Tensor:
        if support_width <= 0:
            raise ValueError("support_width must be positive")

        if support_width == self.num_logits:
            return self.mask_parameters

        x = self.mask_parameters.view(1, 1, -1)
        x = F.interpolate(x, size=support_width, mode="linear", align_corners=True)
        return x.view(-1)

    # -------------------------------------------#
    # ------- contiguous fixed center mask ----- #
    # -------------------------------------------#
    def _build_center_mask_1d(
        self,
        support_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int, int, int]:
        num_low_frequencies = max(1, round(support_width * self.center_fraction))
        center_mask = torch.zeros(support_width, device=device, dtype=dtype)
        pad = (support_width - num_low_frequencies + 1) // 2
        center_mask[pad : pad + num_low_frequencies] = 1.0
        return center_mask, num_low_frequencies, pad, pad + num_low_frequencies

    # -------------------------------------------#
    # -------- embed support mask in W --------- #
    # -------------------------------------------#
    def _embed_support_mask(
        self,
        support_mask: torch.Tensor,
        full_width: int,
        acq_start: int,
        acq_end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        full_mask = torch.zeros(full_width, device=device, dtype=dtype)
        full_mask[acq_start:acq_end] = support_mask
        return full_mask

    # -------------------------------------------#
    # -------- linear mean renormalization ----- #
    # -------------------------------------------#
    def _renormalize_probabilities(
        self,
        probs: torch.Tensor,
        target_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear renormalization so mean(probs_out) == target_mean while keeping
        values in [0, 1].

        This is the 1D analogue of the Renormalize(.) operation described in the
        MoDL+BS LOUPE-ST paper.
        """
        eps = torch.finfo(probs.dtype).eps
        target_mean = torch.clamp(target_mean, 0.0, 1.0)

        if probs.numel() == 0:
            return probs

        current_mean = probs.mean()

        # Degenerate exact targets.
        if target_mean <= 0:
            return torch.zeros_like(probs)
        if target_mean >= 1:
            return torch.ones_like(probs)

        # If current mean is larger than target, scale down linearly.
        if current_mean > target_mean:
            scale = target_mean / (current_mean + eps)
            out = probs * scale
        else:
            # Otherwise scale the complement linearly.
            scale = (1.0 - target_mean) / (1.0 - current_mean + eps)
            out = 1.0 - (1.0 - probs) * scale

        return torch.clamp(out, 0.0, 1.0)

    # -------------------------------------------#
    # ----------- MoDL+BS 1D forward ----------- #
    # -------------------------------------------#
    def forward(
        self,
        full_kspace: torch.Tensor,
        acq_start: Union[int, torch.Tensor],
        acq_end: Union[int, torch.Tensor],
    ):
        if full_kspace.ndim != 5 or full_kspace.shape[-1] != 2:
            raise ValueError("Expected full_kspace with shape [B, coils, H, W, 2]")

        device = full_kspace.device
        batch_size = full_kspace.shape[0]
        full_width = full_kspace.shape[-2]

        if isinstance(acq_start, int):
            acq_start = torch.full((batch_size,), acq_start, device=device, dtype=torch.long)
        else:
            acq_start = acq_start.to(device=device, dtype=torch.long).view(-1)

        if isinstance(acq_end, int):
            acq_end = torch.full((batch_size,), acq_end, device=device, dtype=torch.long)
        else:
            acq_end = acq_end.to(device=device, dtype=torch.long).view(-1)

        if acq_start.numel() != batch_size or acq_end.numel() != batch_size:
            raise ValueError("acq_start and acq_end must have batch_size elements")

        sampled_hard_full = []
        sampled_st_full = []
        prob_full_batch = []
        center_full_batch = []
        outer_prob_means = []
        outer_prob_raw_means = []
        support_widths = []
        sampled_lines = []
        effective_accelerations = []
        num_low_freqs_list = []
        center_start_list = []
        center_end_list = []
        target_total_ratio_list = []
        target_outer_mean_list = []

        for b in range(batch_size):
            start = int(acq_start[b].item())
            end = int(acq_end[b].item())
            support_width = end - start
            if support_width <= 0:
                raise ValueError("acq_end must be larger than acq_start")

            support_widths.append(float(support_width))

            resized_params = self._resize_parameter_map_1d(support_width)

            center_mask_support, num_low_frequencies, center_start, center_end = self._build_center_mask_1d(
                support_width=support_width,
                device=device,
                dtype=full_kspace.dtype,
            )
            num_low_freqs_list.append(num_low_frequencies)
            center_start_list.append(center_start)
            center_end_list.append(center_end)

            outer_indicator = (1.0 - center_mask_support).to(resized_params.dtype)
            outer_idx = outer_indicator.bool()
            outer_count = int(outer_idx.sum().item())

            target_total_ratio = 1.0 / float(self.acceleration)
            target_total_ratio_list.append(target_total_ratio)

            if outer_count > 0:
                target_outer_mean = (
                    target_total_ratio * support_width - num_low_frequencies
                ) / float(outer_count)
                target_outer_mean = float(max(0.0, min(1.0, target_outer_mean)))

                outer_probs_raw = torch.sigmoid(self.prob_slope * resized_params[outer_idx])
                outer_prob_raw_means.append(outer_probs_raw.mean())

                outer_probs = self._renormalize_probabilities(
                    outer_probs_raw,
                    target_mean=outer_probs_raw.new_tensor(target_outer_mean),
                )
                outer_prob_means.append(outer_probs.mean())
                target_outer_mean_list.append(target_outer_mean)

                if self.training:
                    u_outer = torch.rand_like(outer_probs)
                    hard_outer = (u_outer < outer_probs).to(outer_probs.dtype)
                else:
                    num_outer_samples = int(round(target_outer_mean * outer_count))
                
                    hard_outer = torch.zeros_like(outer_probs)
                
                    if num_outer_samples > 0:
                        topk_indices = torch.topk(
                            outer_probs,
                            k=num_outer_samples,
                        ).indices
                
                        hard_outer[topk_indices] = 1.0
                
                # Straight-through estimator for the binary sampling block:
                # forward -> binary mask, backward -> identity wrt probabilities.
                st_outer = hard_outer + outer_probs - outer_probs.detach()

                prob_support = center_mask_support.clone().to(resized_params.dtype)
                hard_support = center_mask_support.clone().to(resized_params.dtype)
                st_support = center_mask_support.clone().to(resized_params.dtype)

                prob_support[outer_idx] = outer_probs
                hard_support[outer_idx] = hard_outer
                st_support[outer_idx] = st_outer
            else:
                # Entire support is center (degenerate case).
                prob_support = center_mask_support.clone().to(resized_params.dtype)
                hard_support = center_mask_support.clone().to(resized_params.dtype)
                st_support = center_mask_support.clone().to(resized_params.dtype)
                outer_prob_raw_means.append(resized_params.new_tensor(0.0))
                outer_prob_means.append(resized_params.new_tensor(0.0))
                target_outer_mean_list.append(0.0)

            sampled_lines_count = hard_support.sum()
            sampled_lines.append(sampled_lines_count)
            effective_accelerations.append(
                resized_params.new_tensor(float(support_width)) / sampled_lines_count.clamp_min(1.0)
            )

            prob_full_batch.append(
                self._embed_support_mask(
                    prob_support,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                    device=device,
                    dtype=prob_support.dtype,
                )
            )
            center_full_batch.append(
                self._embed_support_mask(
                    center_mask_support,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                    device=device,
                    dtype=prob_support.dtype,
                )
            )
            sampled_hard_full.append(
                self._embed_support_mask(
                    hard_support,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                    device=device,
                    dtype=prob_support.dtype,
                )
            )
            sampled_st_full.append(
                self._embed_support_mask(
                    st_support,
                    full_width=full_width,
                    acq_start=start,
                    acq_end=end,
                    device=device,
                    dtype=prob_support.dtype,
                )
            )

        unique_num_low = sorted(set(num_low_freqs_list))
        if len(unique_num_low) != 1:
            raise ValueError(
                "MoDL+BS 1D mask produced different center sizes inside the same batch. "
                "Use batch_size=1 or update the VarNet sensitivity path to accept per-sample center sizes."
            )
        num_low_frequencies = unique_num_low[0]

        sampled_hard_full = torch.stack(sampled_hard_full, dim=0)
        sampled_st_full = torch.stack(sampled_st_full, dim=0)
        prob_full_batch = torch.stack(prob_full_batch, dim=0)
        center_full_batch = torch.stack(center_full_batch, dim=0)

        mask_st = sampled_st_full[:, None, None, :, None].to(dtype=full_kspace.dtype)
        mask_hard = sampled_hard_full[:, None, None, :, None].to(dtype=full_kspace.dtype)

        mask_bool = mask_hard.to(dtype=torch.bool)
        masked_kspace = full_kspace * mask_st

        extras = {
            "prob_mask_1d": prob_full_batch,
            "hard_mask_1d": sampled_hard_full,
            "mask_st_1d": sampled_st_full,
            "center_mask_1d": center_full_batch,
            "support_width": torch.tensor(support_widths, device=device, dtype=torch.float32),
            "sampled_lines": torch.stack(sampled_lines).to(device=device, dtype=torch.float32),
            "effective_acceleration": torch.stack(effective_accelerations).to(device=device, dtype=torch.float32),
            "outer_prob_mean": torch.stack(outer_prob_means).to(device=device, dtype=torch.float32),
            "outer_prob_raw_mean": torch.stack(outer_prob_raw_means).to(device=device, dtype=torch.float32),
            "target_total_ratio": torch.tensor(target_total_ratio_list, device=device, dtype=torch.float32),
            "target_outer_mean": torch.tensor(target_outer_mean_list, device=device, dtype=torch.float32),
            # Kept for compatibility with the existing module API; not used in the MoDL+BS loss.
            "prob_reg_loss": torch.tensor(0.0, device=device, dtype=torch.float32),
            "num_low_frequencies": num_low_frequencies,
            "num_low_frequencies_per_sample": torch.tensor(num_low_freqs_list, device=device, dtype=torch.float32),
            "center_start_per_sample": torch.tensor(center_start_list, device=device, dtype=torch.float32),
            "center_end_per_sample": torch.tensor(center_end_list, device=device, dtype=torch.float32),
            "mask_dim": "1d",
        }

        return masked_kspace, mask_bool, num_low_frequencies, extras
