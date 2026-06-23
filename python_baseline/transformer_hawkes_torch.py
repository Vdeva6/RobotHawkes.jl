import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerHawkesConfig:
    num_events: int
    embed_dim: int
    num_heads: int
    init_scale: float = 0.02
    eps: float = 1.0e-6

    @property
    def head_dim(self) -> int:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        return self.embed_dim // self.num_heads


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, init_scale: float = 0.02):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for sin/cos pairs")

        half_dim = embed_dim // 2

        self.embed_dim = embed_dim
        self.logfreq = nn.Parameter(init_scale * torch.randn(half_dim))
        self.phase = nn.Parameter(init_scale * torch.randn(half_dim))
        self.scale = nn.Parameter(torch.ones(half_dim))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: (T, B)
        # output:  (E, T, B)
        half_dim = self.embed_dim // 2

        dt = delta_t.unsqueeze(0)  # (1, T, B)

        freq = torch.exp(self.logfreq).view(half_dim, 1, 1)
        phase = self.phase.view(half_dim, 1, 1)
        scale = self.scale.view(half_dim, 1, 1)

        angles = freq * dt + phase

        y_sin = scale * torch.sin(angles)
        y_cos = scale * torch.cos(angles)

        return torch.cat([y_sin, y_cos], dim=0)


class EventEmbedding(nn.Module):
    def __init__(self, num_events: int, embed_dim: int, init_scale: float = 0.02):
        super().__init__()

        if num_events <= 0:
            raise ValueError("num_events must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.num_events = num_events
        self.embed_dim = embed_dim

        # PyTorch embedding uses 0-based ids. Our Julia model uses 1-based ids.
        self.table = nn.Embedding(num_events, embed_dim)
        with torch.no_grad():
            self.table.weight.normal_(mean=0.0, std=init_scale)

    def forward(self, event_ids: torch.Tensor) -> torch.Tensor:
        # event_ids: Julia-compatible 1-based ids, shape (T, B)
        # embedded:  (T, B, E)
        # output:    (E, T, B)
        if torch.min(event_ids) < 1 or torch.max(event_ids) > self.num_events:
            raise ValueError(f"event ids must be in 1:{self.num_events}")

        embedded = self.table(event_ids.long() - 1)
        return embedded.permute(2, 0, 1).contiguous()


class TransformerHawkesCell(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, init_scale: float = 0.02):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        self.decay = nn.Parameter(init_scale * torch.randn(num_heads))

        for layer in (self.Wq, self.Wk, self.Wv, self.Wo):
            with torch.no_grad():
                layer.weight.normal_(mean=0.0, std=init_scale)
                layer.bias.zero_()

    def _project(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        # x:      (E, T, B)
        # PyTorch Linear expects last dimension features.
        # output: (E, T, B)
        xt = x.permute(1, 2, 0)          # (T, B, E)
        yt = layer(xt)                   # (T, B, E)
        return yt.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # x:     (E, T, B)
        # times: (T, B)
        E, T, B = x.shape
        H = self.num_heads
        D = self.head_dim

        if E != self.embed_dim:
            raise ValueError("x first dimension must equal embed_dim")
        if times.shape != (T, B):
            raise ValueError("times must have shape (T, B) matching x")

        q = self._project(self.Wq, x)
        k = self._project(self.Wk, x)
        v = self._project(self.Wv, x)

        # (E, T, B) -> (D, H, T, B) -> (D, T, H, B)
        qh = q.reshape(D, H, T, B).permute(0, 2, 1, 3).contiguous()
        kh = k.reshape(D, H, T, B).permute(0, 2, 1, 3).contiguous()
        vh = v.reshape(D, H, T, B).permute(0, 2, 1, 3).contiguous()

        # Collapse heads and batch: (D, T, H * B)
        qn = qh.reshape(D, T, H * B)
        kn = kh.reshape(D, T, H * B)
        vn = vh.reshape(D, T, H * B)

        # KᵀQ: (T, D, HB) batch-matmul (D, T, HB) -> (T, T, HB)
        content_scores = torch.bmm(
            kn.permute(2, 1, 0),   # (HB, T, D)
            qn.permute(2, 0, 1),   # (HB, D, T)
        ).permute(1, 2, 0).contiguous()

        content_scores = content_scores * (1.0 / math.sqrt(D))

        # Δ[s, t, b] = max(times[t, b] - times[s, b], 0)
        source_times = times.unsqueeze(1)  # (T, 1, B), source s on dim 0
        target_times = times.unsqueeze(0)  # (1, T, B), target t on dim 1
        delta = torch.clamp(target_times - source_times, min=0.0)  # (T, T, B)

        delta4 = delta.reshape(T, T, 1, B)
        decay4 = torch.abs(self.decay).reshape(1, 1, H, 1)
        hawkes_bias = (decay4 * delta4).reshape(T, T, H * B)

        causal = torch.triu(
            torch.full((T, T), float("-inf"), dtype=x.dtype, device=x.device),
            diagonal=1,
        ).reshape(T, T, 1)

        scores = content_scores - hawkes_bias + causal
        weights = torch.softmax(scores, dim=0)

        yn = torch.bmm(
            vn.permute(2, 0, 1),       # (HB, D, T)
            weights.permute(2, 0, 1),  # (HB, T, T)
        ).permute(1, 2, 0).contiguous()

        yh = yn.reshape(D, T, H, B)
        y_heads = yh.permute(0, 2, 1, 3).contiguous().reshape(E, T, B)

        return self._project(self.Wo, y_heads)


class TransformerHawkesModel(nn.Module):
    def __init__(self, config: TransformerHawkesConfig):
        super().__init__()

        self.config = config

        self.event = EventEmbedding(
            config.num_events,
            config.embed_dim,
            config.init_scale,
        )
        self.time = TemporalEmbedding(
            config.embed_dim,
            config.init_scale,
        )
        self.cell = TransformerHawkesCell(
            config.embed_dim,
            config.num_heads,
            config.init_scale,
        )

        self.intensity = nn.Linear(config.embed_dim, config.num_events)
        with torch.no_grad():
            self.intensity.weight.normal_(mean=0.0, std=config.init_scale)
            self.intensity.bias.zero_()

    def forward(self, event_ids: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # event_ids: (T, B), 1-based event ids
        # delta_t:   (T, B)
        x_event = self.event(event_ids)
        x_time = self.time(delta_t)

        x = x_event + x_time
        times = torch.cumsum(delta_t, dim=0)

        h = self.cell(x, times)

        # h:      (E, T, B)
        # logits: (T, B, K)
        ht = h.permute(1, 2, 0)
        logits = self.intensity(ht)

        lambdas = F.softplus(logits) + self.config.eps

        return lambdas.permute(2, 0, 1).contiguous()


def observed_loglikelihood(lambdas: torch.Tensor, event_ids: torch.Tensor) -> torch.Tensor:
    # lambdas:   (K, T, B)
    # event_ids: (T, B), 1-based
    K, T, B = lambdas.shape

    gather_ids = event_ids.long().unsqueeze(0) - 1
    observed = torch.gather(lambdas, dim=0, index=gather_ids).squeeze(0)

    return torch.log(observed).sum()


def observed_nll(
    lambdas: torch.Tensor,
    event_ids: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    nll = -observed_loglikelihood(lambdas, event_ids)

    if normalize:
        return nll / event_ids.numel()

    return nll


def total_intensity_integral(
    lambdas: torch.Tensor,
    delta_t: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    total = (lambdas.sum(dim=0) * delta_t).sum()

    if normalize:
        return total / delta_t.numel()

    return total


def full_hawkes_nll(
    lambdas: torch.Tensor,
    event_ids: torch.Tensor,
    delta_t: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    return observed_nll(lambdas, event_ids, normalize) + total_intensity_integral(
        lambdas,
        delta_t,
        normalize,
    )


def model_observed_nll(
    model: TransformerHawkesModel,
    event_ids: torch.Tensor,
    delta_t: torch.Tensor,
) -> torch.Tensor:
    lambdas = model(event_ids, delta_t)
    return observed_nll(lambdas, event_ids)


def model_full_hawkes_nll(
    model: TransformerHawkesModel,
    event_ids: torch.Tensor,
    delta_t: torch.Tensor,
) -> torch.Tensor:
    lambdas = model(event_ids, delta_t)
    return full_hawkes_nll(lambdas, event_ids, delta_t)