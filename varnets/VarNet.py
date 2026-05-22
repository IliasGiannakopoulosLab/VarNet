from typing import NamedTuple, Optional, Tuple, List
from torch import Tensor
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
torch.set_float32_matmul_precision("high")
from utilities.functions import ifft2c_new as ifft2c
from utilities.functions import (
    batched_mask_center,
    rss_complex,
    rss,
    complex_abs,
    image_crop,
    image_uncrop,
    sens_expand,
    sens_reduce,
)

# -------------------------------------------#
# ---------------- norm stats -------------- #
# -------------------------------------------#
class NormStats(nn.Module):
    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:

        batch, chans, _, _ = data.shape

        if batch != 1:
            raise ValueError("Unexpected input dimensions.")

        data = data.view(chans, -1)

        mean = data.mean(dim=1)
        variance = data.var(dim=1, unbiased=False)

        assert mean.ndim == 1
        assert variance.ndim == 1
        assert mean.shape[0] == chans
        assert variance.shape[0] == chans

        return mean, variance


# -------------------------------------------#
# --------------- feature image ------------ #
# -------------------------------------------#
class FeatureImage(NamedTuple):
    features: Tensor
    sens_maps: Tensor = None
    crop_size: Optional[Tuple[int, int]] = None
    means: Tensor = None
    variances: Tensor = None
    mask: Tensor = None
    ref_kspace: Tensor = None
    beta: Optional[Tensor] = None
    gamma: Optional[Tensor] = None


# -------------------------------------------#
# --------------- crop helpers ------------- #
# -------------------------------------------#
def _safe_crop_size(
    image: Tensor,
    crop_size: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Return a valid center-crop size for the H/W image dimensions.

    Supports both channel-first tensors and complex-last tensors:
        [B, C, H, W]
        [B, C, H, W, 2]
        [B, C, D, H, W]
        [B, C, D, H, W, 2]
    """

    if crop_size is None:
        return None

    if len(crop_size) != 2:
        raise ValueError(f"Expected crop_size=(H, W), got {crop_size}")

    if image.ndim in (5, 6) and image.shape[-1] == 2:
        h_size = image.shape[-3]
        w_size = image.shape[-2]
    else:
        h_size = image.shape[-2]
        w_size = image.shape[-1]

    crop_h = min(int(crop_size[0]), int(h_size))
    crop_w = min(int(crop_size[1]), int(w_size))

    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Invalid crop_size={crop_size} for image shape {image.shape}")

    return crop_h, crop_w


def _apply_model_with_optional_crop(
    model: nn.Module,
    image: Tensor,
    crop_size: Optional[Tuple[int, int]],
) -> Tensor:
    """Apply an image-domain model on a center crop and uncrop back.

    The crop is applied only over the last two dimensions. For 3D tensors this
    keeps the slice/depth dimension intact and crops only H/W.
    """

    crop_size = _safe_crop_size(image, crop_size)

    if crop_size is None:
        return model(image)

    return image_uncrop(
        model(image_crop(image, crop_size)),
        image.clone(),
    )


# -------------------------------------------#
# --------------- feature encoder ---------- #
# -------------------------------------------#
class FeatureEncoder(nn.Module):
    def __init__(self, in_chans: int, feature_chans: int = 32, drop_prob: float = 0.0):
        super().__init__()
        self.feature_chans = feature_chans

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=feature_chans,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
        )

    def forward(self, image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.encoder((image - means) * torch.rsqrt(variances))


# -------------------------------------------#
# --------------- feature decoder ---------- #
# -------------------------------------------#
class FeatureDecoder(nn.Module):
    def __init__(self, feature_chans: int = 32, out_chans: int = 2):
        super().__init__()
        self.feature_chans = feature_chans

        self.decoder = nn.Conv2d(
            in_channels=feature_chans,
            out_channels=out_chans,
            kernel_size=5,
            padding=2,
            bias=True,
        )

    def forward(self, features: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.decoder(features) * torch.sqrt(variances) + means


# -------------------------------------------#
# ---------------- attention pe ------------ #
# -------------------------------------------#
class AttentionPE(nn.Module):
    def __init__(self, in_chans: int):
        super().__init__()
        self.in_chans = in_chans

        self.norm = nn.InstanceNorm2d(in_chans)
        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.dilated_conv = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=2, dilation=2)

    def reshape_to_blocks(self, x: Tensor, accel: int) -> Tensor:
        chans = x.shape[1]
        pad_total = (accel - (x.shape[3] - accel)) % accel
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        x = F.pad(x, (pad_left, pad_right, 0, 0), "reflect")
        return (
            torch.stack(x.chunk(chunks=accel, dim=3), dim=-1)
            .view(chans, -1, accel)
            .permute(1, 0, 2)
            .contiguous()
        )

    def reshape_from_blocks(self, x: Tensor, image_size: Tuple[int, int], accel: int) -> Tensor:
        chans = x.shape[1]
        num_freq, num_phase = image_size
        x = (
            x.permute(1, 0, 2)
            .reshape(1, chans, num_freq, -1, accel)
            .permute(0, 1, 2, 4, 3)
            .reshape(1, chans, num_freq, -1)
        )
        padded_phase = x.shape[3]
        pad_total = padded_phase - num_phase
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        return x[:, :, :, pad_left : padded_phase - pad_right]

    def get_positional_encodings(self, seq_len: int, embed_dim: int, device: str) -> Tensor:
        freqs = torch.tensor(
            [1 / (10000 ** (2 * (i // 2) / embed_dim)) for i in range(embed_dim)],
            device=device,
        )
        freqs = freqs.unsqueeze(0)
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        scaled = positions * freqs
        sin_encodings = torch.sin(scaled)
        cos_encodings = torch.cos(scaled)
        encodings = torch.cat([sin_encodings, cos_encodings], dim=1)[:, :embed_dim]
        return encodings

    def forward(self, x: Tensor, accel: int) -> Tensor:
        im_size = (x.shape[2], x.shape[3])
        h_ = x
        h_ = self.norm(h_)

        pos_enc = self.get_positional_encodings(x.shape[2], x.shape[3], h_.device)
        h_ = h_ + pos_enc

        q = self.dilated_conv(self.q(h_))
        k = self.dilated_conv(self.k(h_))
        v = self.dilated_conv(self.v(h_))

        c = q.shape[1]
        q = self.reshape_to_blocks(q, accel)
        k = self.reshape_to_blocks(k, accel)
        q = q.permute(0, 2, 1)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = self.reshape_to_blocks(v, accel)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = self.reshape_from_blocks(h_, im_size, accel)

        h_ = self.proj_out(h_)

        return x + h_


# -------------------------------------------#
# ------------------- unet ----------------- #
# -------------------------------------------#
class UNet(nn.Module):

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        # ---------------- encoder ---------------- #
        self.down = nn.ModuleList()
        ch = chans
        self.down.append(ConvBlock(in_chans, ch, drop_prob))

        for _ in range(num_pool_layers - 1):
            self.down.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        # bottleneck
        self.mid = ConvBlock(ch, ch * 2, drop_prob)
        ch = ch * 2

        # ---------------- decoder ---------------- #
        self.up_transpose = nn.ModuleList()
        self.up_conv = nn.ModuleList()

        for _ in range(num_pool_layers - 1):
            self.up_transpose.append(TransposeConvBlock(ch, ch // 2))
            self.up_conv.append(ConvBlock(ch, ch // 2, drop_prob))
            ch //= 2

        self.up_transpose.append(TransposeConvBlock(ch, ch // 2))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch, ch // 2, drop_prob),
                nn.Conv2d(ch // 2, out_chans, kernel_size=1, stride=1),
            )
        )

    # -------------------------------------------#
    # ------------------ forward --------------- #
    # -------------------------------------------#
    def forward(self, x: Tensor) -> Tensor:

        stack = []
        out = x

        # encoder
        for layer in self.down:
            out = layer(out)
            stack.append(out)
            out = F.avg_pool2d(out, 2)

        out = self.mid(out)

        # decoder
        for tconv, conv in zip(self.up_transpose, self.up_conv):
            skip = stack.pop()
            out = tconv(out)

            # pad if needed
            diffY = skip.shape[-2] - out.shape[-2]
            diffX = skip.shape[-1] - out.shape[-1]
            if diffY != 0 or diffX != 0:
                out = F.pad(out, [0, diffX, 0, diffY], mode="reflect")

            out = torch.cat([out, skip], dim=1)
            out = conv(out)

        return out


# -------------------------------------------#
# ---------------- norm unet --------------- #
# -------------------------------------------#
class NormUNet(nn.Module):

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        use_complex: bool = True,
    ):
        super().__init__()

        self.use_complex = use_complex

        self.unet = UNet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    # -------------------------------------------#
    # -------- complex → channel dim ----------- #
    # -------------------------------------------#
    def complex_to_chan_dim(self, x: Tensor) -> Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    # -------------------------------------------#
    # -------- channel → complex dim ----------- #
    # -------------------------------------------#
    def chan_complex_to_last_dim(self, x: Tensor) -> Tensor:
        b, c2, h, w = x.shape
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    # -------------------------------------------#
    # ------------------ norm ------------------ #
    # -------------------------------------------#
    def norm(self, x: Tensor):
        b, c, h, w = x.shape
        x_ = x.view(b, c, -1)

        mean = x_.mean(dim=2).view(b, c, 1, 1)
        std = x_.std(dim=2).view(b, c, 1, 1)

        return (x - mean) / (std + 1e-8), mean, std

    # -------------------------------------------#
    # ---------------- unnorm ------------------ #
    # -------------------------------------------#
    def unnorm(self, x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return x * std + mean

    # -------------------------------------------#
    # ------------------- pad ------------------ #
    # -------------------------------------------#
    def pad(self, x: Tensor):
        _, _, h, w = x.shape
        h_mult = ((h - 1) | 15) + 1
        w_mult = ((w - 1) | 15) + 1

        pad_h = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        pad_w = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]

        x = F.pad(x, pad_w + pad_h)
        return x, (pad_h, pad_w, h_mult, w_mult)

    # -------------------------------------------#
    # ------------------ unpad ----------------- #
    # -------------------------------------------#
    def unpad(self, x: Tensor, pad_h, pad_w, h_mult, w_mult):
        return x[..., pad_h[0] : h_mult - pad_h[1], pad_w[0] : w_mult - pad_w[1]]

    # -------------------------------------------#
    # ---------------- forward ----------------- #
    # -------------------------------------------#
    def forward(self, x: Tensor) -> Tensor:

        is_complex = self.use_complex and x.shape[-1] == 2

        if is_complex:
            x = self.complex_to_chan_dim(x)

        x, mean, std = self.norm(x)
        x, pad_info = self.pad(x)

        x = self.unet(x)

        x = self.unpad(x, *pad_info)
        x = self.unnorm(x, mean, std)

        if is_complex:
            x = self.chan_complex_to_last_dim(x)

        return x


# -------------------------------------------#
# ---------------- conv block -------------- #
# -------------------------------------------#
class ConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)



# -------------------------------------------#
# ------------ transpose conv block -------- #
# -------------------------------------------#
class TransposeConvBlock(nn.Module):

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)



# -------------------------------------------#
# ------------ sensitivity model ----------- #
# -------------------------------------------#
class SensitivityModel(nn.Module):

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        super().__init__()

        self.mask_center = mask_center

        self.norm_unet = NormUNet(
            chans=chans,
            num_pools=num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
            use_complex=True,
        )

    # -------------------------------------------#
    # --------- chans → batch dim -------------- #
    # -------------------------------------------#
    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    # -------------------------------------------#
    # -------- batch → chans dim --------------- #
    # -------------------------------------------#
    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    # -------------------------------------------#
    # -------- divide rss ---------------------- #
    # -------------------------------------------#
    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    # -------------------------------------------#
    # ---- get pad + num low freqs ------------- #
    # -------------------------------------------#
    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if num_low_frequencies is None or num_low_frequencies == 0:

            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2

            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)

            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right),
                torch.ones_like(left),
            )
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = torch.div(
            mask.shape[-2] - num_low_frequencies_tensor + 1,
            2,
            rounding_mode="trunc",
        )

        return pad, num_low_frequencies_tensor

    # -------------------------------------------#
    # ---------------- forward ----------------- #
    # -------------------------------------------#
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:

        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))

        sens = self.norm_unet(images)

        sens = self.batch_chans_to_chan_dim(sens, batches)

        return self.divide_root_sum_of_squares(sens)


# -------------------------------------------#
# ------------------ fi varnet ------------- #
# -------------------------------------------#
class FIVarNet(nn.Module):

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        acceleration: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()

        if image_conv_cascades is None:
            image_conv_cascades = [i for i in range(num_cascades) if i % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor

        # ---------------- sensitivity net ---------------- #
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )

        # ---------------- feature encoder/decoder -------- #
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        self.decode_norm = nn.InstanceNorm2d(chans)
        self.norm_fn = NormStats()

        # ---------------- feature cascades ---------------- #
        cascades = []
        for i in range(num_cascades):

            use_image_conv = i in self.image_conv_cascades

            cascades.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    acceleration=acceleration,
                    feature_processor=NormUNet(
                        chans=chans,
                        num_pools=pools,
                        in_chans=chans,
                        out_chans=chans,
                        use_complex=False,
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.cascades = nn.Sequential(*cascades)

        # ---------------- image cascades ------------------ #
        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUNet(chans, pools)) for _ in range(num_cascades)]
        )

    # -------------------------------------------#
    # --------------- decode output ------------ #
    # -------------------------------------------#
    def _decode_output(self, feature_image: FeatureImage) -> Tuple[Tensor, Tensor]:

        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )

        return sens_expand(image, feature_image.sens_maps)

    # -------------------------------------------#
    # --------------- encode input ------------- #
    # -------------------------------------------#
    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:

        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)

        image = sens_reduce(masked_kspace, sens_maps)

        crop_size = _safe_crop_size(image, crop_size)

        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    # -------------------------------------------#
    # ---------------- forward ----------------- #
    # -------------------------------------------#
    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:

        masked_kspace = masked_kspace * self.kspace_mult_factor

        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )

        feature_image = self.cascades(feature_image)

        kspace_pred = self._decode_output(feature_image)

        for cascade in self.image_cascades:
            kspace_pred = cascade(
                kspace_pred,
                feature_image.ref_kspace,
                mask,
                feature_image.sens_maps,
                crop_size=feature_image.crop_size,
            )

        kspace_pred = kspace_pred / self.kspace_mult_factor

        return rss(complex_abs(ifft2c(kspace_pred)), dim=1)


# -------------------------------------------#
# ---------------- e2e varnet -------------- #
# -------------------------------------------#
class E2EVarNet(nn.Module):

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUNet(chans, pools)) for _ in range(num_cascades)]
        )

    # -------------------------------------------#
    # ---------------- forward ----------------- #
    # -------------------------------------------#
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:

        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(
                kspace_pred,
                masked_kspace,
                mask,
                sens_maps,
                crop_size=crop_size,
            )

        return rss(complex_abs(ifft2c(kspace_pred)), dim=1)


# -------------------------------------------#
# ----------- uncertainty network ---------- #
# -------------------------------------------#
class UncertaintyNetwork(nn.Module):
    """
    Image-only uncertainty model.

    Input:
        reconstruction image [B, H, W]

    Output:
        lower_pred [B, H, W]
        upper_pred [B, H, W]
    """

    def __init__(
        self,
        head_chans: int = 32,
        head_pools: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.lower_head = UNet(
            in_chans=1,
            out_chans=1,
            chans=head_chans,
            num_pool_layers=head_pools,
            drop_prob=drop_prob,
        )
        self.upper_head = UNet(
            in_chans=1,
            out_chans=1,
            chans=head_chans,
            num_pool_layers=head_pools,
            drop_prob=drop_prob,
        )

    def forward(
        self,
        recon: torch.Tensor,
        crop_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            recon: [B, H, W]
            crop_size: optional center crop over H/W for the uncertainty heads

        Returns:
            lower_pred: [B, H, W]
            upper_pred: [B, H, W]
        """
        if recon.ndim != 3:
            raise ValueError(f"Expected recon with shape [B, H, W], got {recon.shape}")

        recon_in = recon.unsqueeze(1)  # [B, 1, H, W]

        lower_raw = _apply_model_with_optional_crop(
            self.lower_head,
            recon_in,
            crop_size,
        ).squeeze(1)
        upper_raw = _apply_model_with_optional_crop(
            self.upper_head,
            recon_in,
            crop_size,
        ).squeeze(1)

        lower_scale = torch.sigmoid(lower_raw)
        upper_scale = torch.sigmoid(upper_raw)

        lower_pred = torch.clamp(recon - lower_scale * recon, min=0.0)
        upper_pred = recon + upper_scale * recon

        return lower_pred, upper_pred


# -------------------------------------------#
# ------- attention feature varnet --------- #
# -------------------------------------------#
class AttentionFeatureVarNetBlock(nn.Module):

    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        acceleration: int,
        feature_processor: nn.Module,
        attention_layer: AttentionPE,
        use_extra_feature_conv: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.attention_layer = attention_layer
        self.use_image_conv = use_extra_feature_conv
        self.acceleration = acceleration

        self.dc_weight = nn.Parameter(torch.ones(1))

        feature_chans = self.encoder.feature_chans

        self.input_norm = nn.InstanceNorm2d(feature_chans)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if self.use_image_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(feature_chans, feature_chans, 5, padding=2, bias=False),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_chans, feature_chans, 5, padding=2, bias=False),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    # -------------------------------------------#
    # -------- encode from kspace -------------- #
    # -------------------------------------------#
    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:

        image = sens_reduce(kspace, feature_image.sens_maps)

        return self.encoder(
            image,
            means=feature_image.means,
            variances=feature_image.variances,
        )

    # -------------------------------------------#
    # -------- decode to kspace ---------------- #
    # -------------------------------------------#
    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:

        image = self.decoder(
            feature_image.features,
            means=feature_image.means,
            variances=feature_image.variances,
        )

        return sens_expand(image, feature_image.sens_maps)

    # -------------------------------------------#
    # -------------- dc term ------------------- #
    # -------------------------------------------#
    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:

        est_kspace = self.decode_to_kspace(feature_image)

        dc_residual = torch.where(
            feature_image.mask,
            est_kspace - feature_image.ref_kspace,
            self.zero,
        )

        return self.dc_weight * self.encode_from_kspace(dc_residual, feature_image)

    # -------------------------------------------#
    # ---------- feature processor ------------- #
    # -------------------------------------------#
    def apply_model_with_crop(self, feature_image: FeatureImage) -> Tensor:

        if feature_image.crop_size is not None:
            return image_uncrop(
                self.feature_processor(
                    image_crop(feature_image.features, feature_image.crop_size)
                ),
                feature_image.features.clone(),
            )
        else:
            return self.feature_processor(feature_image.features)

    # -------------------------------------------#
    # ---------------- forward ----------------- #
    # -------------------------------------------#
    def forward(self, feature_image: FeatureImage) -> FeatureImage:

        feature_image = feature_image._replace(
            features=self.input_norm(feature_image.features)
        )

        new_features = feature_image.features - self.compute_dc_term(feature_image)

        attended = self.attention_layer(
            feature_image.features,
            self.acceleration,
        )

        feature_image = feature_image._replace(features=attended)

        new_features = new_features - self.apply_model_with_crop(feature_image)

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)


# -------------------------------------------#
# ---------------- varnet block ------------ #
# -------------------------------------------#
class VarNetBlock(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    # -------------------------------------------#
    # ---------------- forward ----------------- #
    # -------------------------------------------#
    def forward(
        self,
        current_kspace: Tensor,
        ref_kspace: Tensor,
        mask: Tensor,
        sens_maps: Tensor,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:

        zero = torch.zeros(1, 1, 1, 1, 1, device=current_kspace.device)

        soft_dc = torch.where(
            mask,
            current_kspace - ref_kspace,
            zero,
        ) * self.dc_weight

        image = sens_reduce(current_kspace, sens_maps)

        model_image = _apply_model_with_optional_crop(
            self.model,
            image,
            crop_size,
        )

        model_term = sens_expand(model_image, sens_maps)

        return current_kspace - soft_dc - model_term    
