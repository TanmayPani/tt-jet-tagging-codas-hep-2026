import numpy as np
import torch
from functools import partial

from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin

_KERNEL_LENGTHS: tuple[int, ...] = (7, 9, 11)
# 2-D image ROCKET: small square kernels. Top-jet prong separation is dR ~ 0.3-0.8,
# i.e. 3-8 pixels at 0.1 px, so the physically relevant receptive fields are small.
_KERNEL_LENGTHS_2D: tuple[int, ...] = (3, 5, 7)


@torch.no_grad()
def _get_kernel_inputs(
    num_channels, num_steps, num_kernels, kernel_lengths=None, generator=None
):
    channels = torch.randint(0, num_channels, (num_kernels,), generator=generator)

    _kernel_lengths = torch.as_tensor(
        kernel_lengths if kernel_lengths is not None else _KERNEL_LENGTHS,
        dtype=torch.long,
    )
    _len_choices = torch.randint(
        0, len(_kernel_lengths), (num_kernels,), generator=generator
    )
    lengths = _kernel_lengths[_len_choices]

    _max_exp = ((num_steps - 1) / (lengths - 1)).log2_().floor_().clamp_(min=0).long()
    _exps = torch.rand(num_kernels, generator=generator).mul_(_max_exp + 1).long()
    dilations = torch.pow(2, _exps)  # long: integer power of two

    paddings = (lengths - 1).mul_(dilations).floor_divide_(2)
    _keep = torch.bernoulli(torch.full((num_kernels,), 0.5), generator=generator).long()
    paddings.mul_(_keep)

    return channels, torch.stack([lengths, dilations, paddings], dim=1)


@torch.no_grad()
def _init_one_rocket_kernel(conv, generator=None):
    conv.weight.normal_(generator=generator)
    conv.weight -= conv.weight.mean(dim=-1, keepdim=True)  # ... mean-centred per kernel
    conv.bias.uniform_(-1, 1, generator=generator)  # bias ~ U(-1, 1)


# ---------------------------------------------------------------------------
# ROCKET (Dempster et al. 2020) — random convolutional kernel transform for the DL
# track. Kernel sampling stays in numpy; the transform is a torch/GPU reimplementation
# of the original numba loops (identical features: RAW input, max + ppv pooling, one
# channel per kernel). Torch expresses ROCKET as what it is — a bank of dilated 1-D
# convolutions — so the whole 10k-kernel transform is a handful of grouped `conv1d`
# calls
# ---------------------------------------------------------------------------


class RocketEncoder(nn.Module):
    """Frozen-by-default ROCKET featurizer, sampled natively in torch. forward() is
    gradient-transparent; freezing is parameter state, so unfreeze() -> Tier 2 (learnable)."""

    def __init__(
        self,
        n_channels,
        n_steps,
        n_kernels=10_000,
        generator=None,
        kernel_length_opts=None,
    ):
        super().__init__()
        self.n_kernels = n_kernels
        # --- sample per-kernel specs ---
        channels, specs = _get_kernel_inputs(
            n_channels,
            n_steps,
            n_kernels,
            generator=generator,
            kernel_lengths=kernel_length_opts,
        )

        group_specs, group_indices, group_counts = specs.unique(
            dim=0, sorted=False, return_inverse=True, return_counts=True
        )

        # --- one grouped Conv1d per (L, d, p) bucket; weights sampled STRAIGHT into the params ---
        self.convs = nn.ModuleList()

        _kernel_idx = torch.arange(n_kernels)
        for _gr, _gr_spec in enumerate(group_specs):
            L, d, p = _gr_spec.tolist()
            _nch = int(group_counts[_gr])
            _conv1d = nn.Conv1d(_nch, _nch, L, dilation=d, padding=p, groups=_nch)
            _conv1d.apply(partial(_init_one_rocket_kernel, generator=generator))
            self.convs.append(_conv1d)
            _gr_kernel_idx = _kernel_idx[group_indices == _gr]
            self.register_buffer(f"ch_{_gr}", channels[_gr_kernel_idx])
            self.register_buffer(f"slot_{_gr}", _gr_kernel_idx)

        self.requires_grad_(False)  # frozen by default; call unfreeze() for Tier 2

    def freeze(self):
        self.requires_grad_(False)

    def unfreeze(self):  # Tier 2: kernels become learnable
        self.requires_grad_(True)

    @torch.no_grad()
    def kernel_specs(self):
        """Per-kernel (channel, length, dilation, padding) arrays indexed by kernel slot,
        for feature-importance interpretation (feature 2k -> max, 2k+1 -> ppv of kernel k)."""
        specs = {
            k: torch.zeros(self.n_kernels, dtype=torch.long, requires_grad=False)
            for k in ("channel", "length", "dilation", "padding")
        }
        for _i, _conv in enumerate(self.convs):
            _slot = getattr(self, f"slot_{_i}").detach().cpu()
            specs["channel"][_slot] = getattr(self, f"ch_{_i}").detach().cpu()
            specs["length"][_slot] = _conv.kernel_size[0]
            specs["dilation"][_slot] = _conv.dilation[0]
            specs["padding"][_slot] = _conv.padding[0]
        return specs

    def forward(self, x):  # x: (N, C, T) -> (N, 2*n_kernels)
        x = torch.nan_to_num(x)
        out = x.new_zeros(x.shape[0], 2 * self.n_kernels)
        for _kr_idx, _kr in enumerate(self.convs):
            _ch, _slot = (
                getattr(self, f"ch_{_kr_idx}"),
                getattr(self, f"slot_{_kr_idx}"),
            )
            _c = _kr(x[:, _ch, :])
            out[:, _slot * 2] = _c.amax(-1)  # max
            out[:, _slot * 2 + 1] = (
                (_c > 0).float().mean(-1)
            )  # ppv (soft-surrogate if trainable)
        return out


# ---------------------------------------------------------------------------
# 2-D image ROCKET for jet images. Same philosophy as the 1-D encoder
# ---------------------------------------------------------------------------


@torch.no_grad()
def _get_kernel_inputs_2d(img_size, num_kernels, kernel_lengths=None, generator=None):
    H, W = img_size
    _lengths_menu = torch.as_tensor(
        kernel_lengths if kernel_lengths is not None else _KERNEL_LENGTHS_2D,
        dtype=torch.long,
    )
    _choice = torch.randint(0, len(_lengths_menu), (num_kernels,), generator=generator)
    lengths = _lengths_menu[_choice]

    def _sample_dilation(axis_size):
        # dilation is a power of two, capped so the receptive field fits the axis.
        _max_exp = (
            ((axis_size - 1) / (lengths - 1)).log2_().floor_().clamp_(min=0).long()
        )
        _exps = torch.rand(num_kernels, generator=generator).mul_(_max_exp + 1).long()
        return torch.pow(2, _exps)

    d_h, d_w = _sample_dilation(H), _sample_dilation(W)
    # "same" padding per axis so output stays H x W (odd lengths -> exact).
    p_h = (lengths - 1).mul(d_h).floor_divide(2)
    p_w = (lengths - 1).mul(d_w).floor_divide(2)
    return torch.stack([lengths, d_h, d_w, p_h, p_w], dim=1)


@torch.no_grad()
def _init_one_rocket_kernel_2d(conv, generator=None):
    conv.weight.normal_(generator=generator)
    # mean-centre each kernel over its L*L taps -> zero response on any flat patch.
    conv.weight -= conv.weight.mean(dim=(-1, -2), keepdim=True)


class RocketEncoder2D(nn.Module):
    """Frozen 2-D ROCKET featurizer for jet images. forward() -> (N, n_kernels*(1+n_biases)):
    one max feature + n_biases ppv features per kernel. The ppv `bias` buffer starts at 0
    (ppv then falls back to plain proportion-positive); RocketTransform2D.fit() calibrates it
    from data. Pure Module -- no data-dependent state is learned here, so it stays usable in a
    training loop if unfrozen."""

    def __init__(
        self,
        img_size=(30, 30),
        n_kernels=5_000,
        n_ppv_biases=3,
        generator=None,
        kernel_length_opts=None,
    ):
        super().__init__()
        self.n_kernels = n_kernels
        self.n_biases = n_ppv_biases
        self.block = 1 + n_ppv_biases  # features per kernel: [max, ppv_0, ...]

        # ppv thresholds; zeros -> plain >0 fallback. RocketTransform2D.fit() fills these
        # (active-position quantiles). Kept as a buffer so forward() is self-contained.
        self.register_buffer("bias", torch.zeros(n_kernels, n_ppv_biases))

        specs = _get_kernel_inputs_2d(
            img_size, n_kernels, kernel_lengths=kernel_length_opts, generator=generator
        )
        group_specs, group_indices, group_counts = specs.unique(
            dim=0, sorted=False, return_inverse=True, return_counts=True
        )

        # one Conv2d(1 -> n_in_group) per (L, d_h, d_w, p_h, p_w) bucket; every kernel
        # reads the single image channel, so out_channels == kernels in the group.
        self.convs = nn.ModuleList()
        _kernel_idx = torch.arange(n_kernels)
        for _gr, _gr_spec in enumerate(group_specs):
            L, d_h, d_w, p_h, p_w = _gr_spec.tolist()
            _nch = int(group_counts[_gr])
            _conv2d = nn.Conv2d(
                1, _nch, L, dilation=(d_h, d_w), padding=(p_h, p_w), bias=False
            )
            _conv2d.apply(partial(_init_one_rocket_kernel_2d, generator=generator))
            self.convs.append(_conv2d)
            self.register_buffer(f"slot_{_gr}", _kernel_idx[group_indices == _gr])

        self.requires_grad_(False)  # frozen; biases come from fit(), not backprop

    def freeze(self):
        self.requires_grad_(False)

    def unfreeze(self):
        self.requires_grad_(True)

    def forward(self, x):  # x: (N, 1, H, W) -> (N, n_kernels * (1 + n_biases))
        x = torch.nan_to_num(x)
        out = x.new_zeros(x.shape[0], self.n_kernels * self.block)
        for _gr, _conv in enumerate(self.convs):
            _slot = getattr(self, f"slot_{_gr}")
            _c = _conv(x).flatten(2)  # (N, nch, P)
            out[:, _slot * self.block] = _c.amax(-1)  # max
            _b = self.bias[_slot]  # (nch, n_biases)
            for _j in range(self.n_biases):
                # ppv at bias j: fraction of positions exceeding the calibrated threshold
                out[:, _slot * self.block + 1 + _j] = (
                    (_c > _b[:, _j].view(1, -1, 1)).float().mean(-1)
                )
        return out


class RocketTransform(BaseEstimator, TransformerMixin):
    """sklearn-style featurizer: sample a frozen ROCKET bank in fit(), return the
    (N, 2*n_kernels) [max, ppv] matrix in transform(). The inference context lives HERE,
    not in the Module, so the same Module stays usable in a training loop if unfrozen."""

    def __init__(self, n_kernels=10_000, seed=42, device=None):
        self.n_kernels, self.seed, self.device = n_kernels, seed, device

    def fit(self, X, y=None):
        X = np.asarray(X)
        _, C, T = X.shape
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        _g = torch.Generator().manual_seed(self.seed)
        self.module_ = (
            RocketEncoder(C, T, self.n_kernels, generator=_g).to(self.device_).eval()
        )  # .eval(): hygiene
        return self

    @torch.no_grad()
    def transform(self, X):
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device_)
        out = self.module_(x)
        return np.nan_to_num(out.detach().cpu().numpy())


class RocketTransform2D(BaseEstimator, TransformerMixin):
    """sklearn-style featurizer for jet images. fit() samples a frozen 2-D ROCKET bank AND
    calibrates its ppv biases from the training images -- per kernel, quantiles taken over
    ACTIVE positions only (|response| > eps), so the ~85% background-zero mass never drags
    the thresholds to 0. transform() returns the (N, n_kernels*(1+n_ppv_biases)) matrix.
    The data-dependent calibration lives HERE, so RocketEncoder2D stays a pure featurizer."""

    def __init__(
        self,
        n_kernels=5_000,
        n_ppv_biases=3,
        ppv_quantile_range=(0.5, 0.9),
        img_size=(30, 30),
        seed=42,
        device=None,
        fit_chunk=256,
        transform_batch=512,
        max_active=1_000_000,
        eps=1e-8,
    ):
        self.n_kernels = n_kernels
        self.n_ppv_biases = n_ppv_biases
        self.ppv_quantile_range = ppv_quantile_range
        self.img_size = img_size
        self.seed = seed
        self.device = device
        self.fit_chunk = fit_chunk
        self.transform_batch = transform_batch
        self.max_active = max_active
        self.eps = eps

    @staticmethod
    def _as_nchw(X):
        X = np.asarray(X, dtype=np.float32)
        return X[:, None] if X.ndim == 3 else X  # (N, H, W) -> (N, 1, H, W)

    def fit(self, X, y=None):
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.nan_to_num(torch.as_tensor(self._as_nchw(X))).to(self.device_)
        _g = torch.Generator().manual_seed(self.seed)
        self.module_ = (
            RocketEncoder2D(
                img_size=self.img_size,
                n_kernels=self.n_kernels,
                n_ppv_biases=self.n_ppv_biases,
                generator=_g,
            )
            .to(self.device_)
            .eval()
        )
        self._calibrate_bias(x)
        return self

    @torch.no_grad()
    def _calibrate_bias(self, images):
        _lo, _hi = self.ppv_quantile_range
        _levels = torch.linspace(_lo, _hi, self.n_ppv_biases, device=images.device)
        _mod = self.module_
        for _gr, _conv in enumerate(_mod.convs):
            _slot = getattr(_mod, f"slot_{_gr}")
            _nch = _slot.numel()
            _active = [[] for _ in range(_nch)]
            for _start in range(0, images.shape[0], self.fit_chunk):
                _c = _conv(images[_start : _start + self.fit_chunk]).flatten(2)
                for _k in range(_nch):
                    _r = _c[:, _k].reshape(-1)
                    _active[_k].append(_r[_r.abs() > self.eps])
            for _k in range(_nch):
                _r = torch.cat(_active[_k]) if _active[_k] else images.new_empty(0)
                if _r.numel() == 0:
                    _qs = torch.zeros_like(_levels)
                else:
                    if _r.numel() > self.max_active:  # torch.quantile caps input size
                        _sel = torch.randint(
                            _r.numel(), (self.max_active,), device=_r.device
                        )
                        _r = _r[_sel]
                    _qs = torch.quantile(_r, _levels)
                _mod.bias[_slot[_k]] = _qs

    @torch.no_grad()
    def transform(self, X):
        # Batch rows through the frozen module: forward() materializes a
        # (batch, nch, P) tensor per kernel group, so a single pass over all N
        # images would blow up memory (OOM on GPU for large N).
        x = torch.nan_to_num(torch.as_tensor(self._as_nchw(X))).to(self.device_)
        out = []
        for _start in range(0, x.shape[0], self.transform_batch):
            _f = self.module_(x[_start : _start + self.transform_batch])
            out.append(_f.detach().cpu().numpy())
        return np.nan_to_num(np.concatenate(out, axis=0))
