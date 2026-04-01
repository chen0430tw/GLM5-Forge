from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class FlashDSABridge:
    """
    Extracted FlashDSA backend logic.

    This keeps the DSA-specific INT4 -> BF16 bridge, index construction,
    and sparse attention path outside the model definition so the model file
    only owns wiring and tensor projection.
    """

    def __init__(
        self,
        *,
        config,
        num_kv_heads: int,
        num_key_value_groups: int,
        head_dim: int,
        value_head_dim: int,
        scaling: float,
        dropout,
        o_proj,
        sparse_mask_builder,
        repeat_kv,
    ) -> None:
        self.config = config
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.scaling = scaling
        self.dropout = dropout
        self.o_proj = o_proj
        self.sparse_mask_builder = sparse_mask_builder
        self.repeat_kv = repeat_kv

    def int4_roundtrip(self, kv_latent: torch.Tensor) -> torch.Tensor:
        group = self.config.int4_group_size
        last = kv_latent.shape[-1]
        if last % group != 0:
            raise RuntimeError(
                f"flash_dsa requires latent dim divisible by int4_group_size, got {last} vs {group}"
            )
        x = kv_latent.float()
        xg = x.view(*x.shape[:-1], last // group, group)
        scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 7.0
        q = torch.round(xg / scale).clamp(-8, 7)
        deq = (q * scale).view_as(x)
        return deq.to(dtype=kv_latent.dtype)

    def build_indices(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.matmul(index_q, index_k.transpose(-2, -1))
        sparse_mask = self.sparse_mask_builder(scores, attention_mask)
        scores = scores.masked_fill(sparse_mask, float("-inf"))
        topk = self.config.dsa_topk
        valid_topk = min(topk, scores.shape[-1])
        top_idx = scores.topk(k=valid_topk, dim=-1).indices
        bsz, _, s_q, _ = scores.shape
        indices = torch.full(
            (bsz, self.num_kv_heads, s_q, topk),
            -1,
            device=scores.device,
            dtype=torch.int32,
        )
        reduced_idx = top_idx[:, :1]
        indices[:, :, :, :valid_topk] = reduced_idx.to(torch.int32)
        return indices

    def sparse_attention(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        kv_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_bf16 = self.int4_roundtrip(kv_latent)
        k = self.repeat_kv(kv_bf16[..., : self.head_dim], self.num_key_value_groups)
        v = self.repeat_kv(
            kv_bf16[..., self.head_dim : self.head_dim + self.value_head_dim],
            self.num_key_value_groups,
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        dsa_indices = self.build_indices(index_q, index_k, attention_mask)
        sparse_mask = torch.ones_like(scores, dtype=torch.bool)
        topk = dsa_indices.shape[-1]
        expanded = dsa_indices[:, :, :, :topk].clamp_min(0).to(torch.int64)
        sparse_mask[:, :1].scatter_(-1, expanded, False)
        sparse_mask = sparse_mask.expand_as(scores)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                sparse_mask |= attention_mask.expand_as(sparse_mask)
            else:
                sparse_mask |= attention_mask[:, None, :, :].expand_as(sparse_mask)
        all_masked = sparse_mask.all(dim=-1)
        if all_masked.any():
            diag = torch.arange(scores.shape[-2], device=scores.device)
            diag = diag.clamp_max(scores.shape[-1] - 1)
            sparse_mask[..., diag, diag] = False
        scores = scores.masked_fill(sparse_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[-2], -1)
        return self.o_proj(out), attn
