"""
GLM-5 reconstructed modeling file.

This is a research reconstruction, not the official source release.

What is grounded by public evidence:
- GLM-4.5 / 4.6 / 4.7 rely on the GLM4-MoE family
- GLM-5 scales the GLM-4.5 line from 355B/32B-active to 744B/40B-active
- GLM-5 adopts DSA (DeepSeek Sparse Attention)
- GLM-5 keeps strong long-context / agentic emphasis

What is inferred here:
- a practical decoder-only implementation that combines:
  - GLM4-MoE style block structure
  - GQA + RoPE + QK norm
  - sparse attention mask as a DSA-style approximation
  - routed MoE + shared experts

This file is intended for study, adaptation, and constrained experimentation.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:
    from flash_mla import flash_mla_sparse_fwd

    FLASH_MLA_AVAILABLE = True
    FLASH_MLA_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - optional dependency
    flash_mla_sparse_fwd = None
    FLASH_MLA_AVAILABLE = False
    FLASH_MLA_IMPORT_ERROR = e


class Glm5ReconstructedConfig(PreTrainedConfig):
    model_type = "glm5_reconstructed"

    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 8.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        hidden_act: str = "silu",
        use_qk_norm: bool = True,
        use_latent_kv: bool = True,
        attention_backend: str = "reference",
        tie_word_embeddings: bool = True,
        pad_token_id: int = 0,
        n_routed_experts: int = 128,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 8,
        first_k_dense_replace: int = 1,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        dsa_topk: int = 256,
        dsa_sink_tokens: int = 16,
        dsa_local_window: int = 256,
        original_context_length: int = 4096,
        midtrain_target_context_length: int = 200000,
        use_mtp_head: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act
        self.use_qk_norm = use_qk_norm
        self.use_latent_kv = use_latent_kv
        self.attention_backend = attention_backend
        self.tie_word_embeddings = tie_word_embeddings
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.dsa_topk = dsa_topk
        self.dsa_sink_tokens = dsa_sink_tokens
        self.dsa_local_window = dsa_local_window
        self.original_context_length = original_context_length
        self.midtrain_target_context_length = midtrain_target_context_length
        self.use_mtp_head = use_mtp_head
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        out = x_f * torch.rsqrt(var + self.eps)
        return self.weight * out.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Glm5ReconstructedConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.max_seq_len_cached = config.max_position_embeddings
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # freqs[b, t, i] = inv_freq[i] * position_ids[b, t]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.float()  # (B, seqlen, half_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, t, d)
    return x.reshape(b, h * n_rep, t, d)


class SparseDSAAttention(nn.Module):
    """
    DSA-style sparse attention approximation.

    This is not DeepSeek's official exact implementation.
    It approximates the paper-level constraint:
    - sparse attention
    - long-context friendliness
    - token-importance-aware pruning
    """

    def __init__(self, config: Glm5ReconstructedConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.use_latent_kv = config.use_latent_kv
        self.attention_backend = config.attention_backend
        self._flash_warned = False

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        if self.use_latent_kv:
            self.kv_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
            self.k_proj = None
            self.v_proj = None
        else:
            self.kv_proj = None
            self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dropout = nn.Dropout(config.attention_dropout)

    def _build_sparse_mask(
        self,
        scores: torch.Tensor,
        base_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Keep:
        - sink tokens
        - local window
        - per-query top-k by score
        """
        b, h, t, s = scores.shape
        device = scores.device
        keep = torch.zeros((b, h, t, s), dtype=torch.bool, device=device)

        sink = min(self.config.dsa_sink_tokens, s)
        if sink > 0:
            keep[..., :sink] = True

        w = min(self.config.dsa_local_window, s)
        if w > 0:
            q_idx = torch.arange(t, device=device)
            k_idx = torch.arange(s, device=device)
            local = (q_idx[:, None] - k_idx[None, :]).abs() <= w
            keep |= local.view(1, 1, t, s)

        k = min(self.config.dsa_topk, s)
        if k > 0:
            top_idx = scores.topk(k=k, dim=-1).indices
            keep.scatter_(-1, top_idx, True)

        if base_mask is not None:
            keep &= ~base_mask

        return ~keep

    def _warn_flash_fallback(self, reason: str) -> None:
        if self._flash_warned:
            return
        warnings.warn(f"Falling back from FlashMLA to reference attention: {reason}", stacklevel=2)
        self._flash_warned = True

    def _can_use_flash_mla_prefill(
        self,
        q: torch.Tensor,
        kv_latent: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> bool:
        if self.attention_backend != "flash_mla":
            return False
        if not FLASH_MLA_AVAILABLE:
            self._warn_flash_fallback(f"flash_mla import failed: {FLASH_MLA_IMPORT_ERROR}")
            return False
        if not self.use_latent_kv:
            self._warn_flash_fallback("FlashMLA path requires latent KV mode")
            return False
        if kv_latent is None:
            self._warn_flash_fallback("latent KV tensor is missing")
            return False
        if past_key_value is not None or use_cache:
            self._warn_flash_fallback("this integration supports FlashMLA prefill only")
            return False
        if not q.is_cuda:
            self._warn_flash_fallback("FlashMLA requires CUDA tensors")
            return False
        if q.shape[0] != 1:
            self._warn_flash_fallback("current FlashMLA integration only supports batch_size=1")
            return False
        if self.num_kv_heads != 1:
            self._warn_flash_fallback("current FlashMLA integration only supports num_key_value_heads=1")
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            self._warn_flash_fallback("FlashMLA expects fp16/bf16 activations")
            return False
        return True

    def _build_flash_mla_indices(
        self,
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seed_scores = torch.matmul(q[0].mean(dim=0), kv_latent[0, 0].transpose(0, 1))
        seed_scores = seed_scores.unsqueeze(0).unsqueeze(0)
        sparse_mask = self._build_sparse_mask(seed_scores, attention_mask)
        seed_scores = seed_scores.masked_fill(sparse_mask, float("-inf"))
        topk = min(self.config.dsa_topk, seed_scores.shape[-1])
        top_idx = seed_scores.topk(k=topk, dim=-1).indices[0, 0]
        indices = torch.full(
            (seed_scores.shape[-2], self.num_kv_heads, topk),
            -1,
            device=q.device,
            dtype=torch.int32,
        )
        indices[:, 0, :] = top_idx.to(torch.int32)
        return indices

    def _flash_mla_prefill(
        self,
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        indices = self._build_flash_mla_indices(q, kv_latent, attention_mask)
        q_fm = q[0].transpose(0, 1).contiguous()
        kv_fm = kv_latent[0].transpose(0, 1).contiguous()
        out, *_ = flash_mla_sparse_fwd(
            q=q_fm,
            kv=kv_fm,
            indices=indices,
            sm_scale=self.scaling,
        )
        out = out.transpose(0, 1).unsqueeze(0).contiguous().view(1, q.shape[-2], -1)
        return self.o_proj(out), None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seqlen, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_latent_kv:
            kv_latent = self.kv_proj(hidden_states).view(
                bsz, seqlen, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            k = kv_latent
            v = kv_latent
        else:
            kv_latent = None
            k = self.k_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_key_value = (k, v) if use_cache else None

        if self._can_use_flash_mla_prefill(q, kv_latent, past_key_value, use_cache):
            flash_out, flash_attn = self._flash_mla_prefill(q, kv_latent, attention_mask)
            return flash_out, flash_attn, present_key_value

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # attention_mask is already incorporated inside _build_sparse_mask
        sparse_mask = self._build_sparse_mask(scores, attention_mask)
        scores = scores.masked_fill(sparse_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(out), attn, present_key_value


class Glm5MLP(nn.Module):
    def __init__(self, config: Glm5ReconstructedConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Glm5TopkRouter(nn.Module):
    def __init__(self, config: Glm5ReconstructedConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(flat.float(), self.weight.float())
        probs = logits.sigmoid()
        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return logits, topk_indices, topk_weights


class Glm5NaiveExperts(nn.Module):
    def __init__(self, config: Glm5ReconstructedConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
        hit_experts = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=True)[0].tolist()

        for idx in hit_experts:
            topk_pos, token_idx = torch.where(expert_mask[idx])
            current = hidden_states[token_idx]
            gate_up = F.linear(current, self.gate_up_proj[idx])
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            hidden = F.linear(hidden, self.down_proj[idx])
            hidden = hidden * topk_weights[token_idx, topk_pos, None]
            final.index_add_(0, token_idx, hidden.to(final.dtype))
        return final


class Glm5MoE(nn.Module):
    def __init__(self, config: Glm5ReconstructedConfig):
        super().__init__()
        self.router = Glm5TopkRouter(config)
        self.experts = Glm5NaiveExperts(config)
        self.shared_experts = Glm5MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        orig_shape = hidden_states.shape
        router_logits, topk_indices, topk_weights = self.router(hidden_states)
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        mixed = self.experts(flat, topk_indices, topk_weights).view(*orig_shape)
        mixed = mixed + self.shared_experts(residual)
        return mixed, router_logits


class Glm5DecoderLayer(nn.Module):
    def __init__(self, config: Glm5ReconstructedConfig, layer_idx: int):
        super().__init__()
        self.self_attn = SparseDSAAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm5MoE(config)
        else:
            self.mlp = Glm5MLP(config)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, attn_weights, present_key_value = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        router_logits = None
        if isinstance(self.mlp, Glm5MoE):
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states, attn_weights, router_logits, present_key_value


class Glm5ReconstructedPreTrainedModel(PreTrainedModel):
    config_class = Glm5ReconstructedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Glm5TopkRouter):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Glm5ReconstructedModel(Glm5ReconstructedPreTrainedModel):
    def __init__(self, config: Glm5ReconstructedConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Glm5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.post_init()

    def _make_causal_mask(
        self,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
        past_seen_tokens: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        causal = torch.triu(
            torch.ones(query_length, key_length, device=device, dtype=torch.bool),
            diagonal=1 + past_seen_tokens,
        ).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            causal = causal | pad_mask
        return causal.expand(batch_size, -1, query_length, key_length)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        bsz, seqlen = hidden_states.shape[:2]
        use_cache = False if use_cache is None else use_cache
        past_seen_tokens = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_seen_tokens = past_key_values[0][0].shape[-2]
        position_ids = (
            torch.arange(
                past_seen_tokens,
                past_seen_tokens + seqlen,
                device=hidden_states.device,
            )
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        total_key_length = past_seen_tokens + seqlen
        if attention_mask is not None and attention_mask.shape[-1] != total_key_length:
            raise ValueError(
                f"attention_mask last dim must match total key length {total_key_length}, "
                f"got {attention_mask.shape[-1]}"
            )
        causal_mask = self._make_causal_mask(
            batch_size=bsz,
            query_length=seqlen,
            key_length=total_key_length,
            device=hidden_states.device,
            past_seen_tokens=past_seen_tokens,
            attention_mask=attention_mask,
        )

        all_attn = []
        all_router_logits = []
        next_past_key_values = []
        for layer_idx, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            hidden_states, attn, router_logits, present_key_value = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            all_attn.append(attn)
            if router_logits is not None:
                all_router_logits.append(router_logits)
            if use_cache:
                next_past_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(next_past_key_values) if use_cache else None,
            attentions=tuple(all_attn),
        )
        out.router_logits = tuple(all_router_logits)
        return out


class Glm5ReconstructedForCausalLM(Glm5ReconstructedPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Glm5ReconstructedConfig):
        super().__init__(config)
        self.model = Glm5ReconstructedModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=outputs.attentions,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None
        reordered = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered.append(None)
                continue
            reordered.append(tuple(p.index_select(0, beam_idx) for p in layer_past))
        return tuple(reordered)


if __name__ == "__main__":
    import shutil
    import tempfile

    cfg = Glm5ReconstructedConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1536,
        moe_intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        dsa_topk=32,
        dsa_local_window=32,
        max_position_embeddings=2048,
    )
    model = Glm5ReconstructedForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    out = model(x, labels=x)
    print("logits:", tuple(out.logits.shape))
    print("loss:", float(out.loss.detach()))
    cached = model(input_ids=x[:, :63], use_cache=True)
    print("cached layers:", len(cached.past_key_values or ()))
    tmpdir = tempfile.mkdtemp(prefix="glm5_reconstructed_")
    try:
        model.save_pretrained(tmpdir)
        cfg.save_pretrained(tmpdir)
        reloaded = Glm5ReconstructedForCausalLM.from_pretrained(tmpdir)
        probe = reloaded(x[:, :4], use_cache=True)
        print("reload logits:", tuple(probe.logits.shape))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
