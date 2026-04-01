import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

import quant  # noqa: E402


def slow_sparse_decode_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    topk_length: torch.Tensor | None,
    head_dim_v: int,
    softmax_scale: float,
):
    b, s_q, h_q, d_qk = q.shape
    topk = indices_in_kvcache.size(-1)

    dequant = quant.dequantize_k_cache(k_cache, quant.FP8KVCacheLayout.V32_FP8Sparse)
    blocked_k = dequant.view(-1, d_qk)

    fixed_indices = torch.clamp_min(indices_in_kvcache, 0)
    gathered_kv = blocked_k.index_select(0, fixed_indices.view(-1)).view(b, s_q, topk, d_qk)

    invalid_mask = indices_in_kvcache == -1
    if topk_length is not None:
        invalid_mask |= torch.arange(0, topk, device=indices_in_kvcache.device).view(1, 1, topk) >= topk_length.view(b, 1, 1)

    gathered_kv = gathered_kv.view(b * s_q, topk, d_qk).float()
    qf = q.float().view(b * s_q, h_q, d_qk)
    attn = qf @ gathered_kv.transpose(-1, -2)
    attn *= softmax_scale
    attn[invalid_mask.view(b * s_q, 1, topk).expand_as(attn)] = float("-inf")

    lse = attn.logsumexp(dim=-1)
    probs = torch.exp(attn - lse.unsqueeze(-1))
    out = probs @ gathered_kv[..., :head_dim_v]
    return out.view(b, s_q, h_q, head_dim_v).to(torch.bfloat16), lse.view(b, s_q, h_q).transpose(1, 2)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    b, s_q, h_q, d_qk, d_v = 2, 3, 8, 576, 512
    s_kv, topk, block_size = 64, 7, 64

    q = torch.randn((b, s_q, h_q, d_qk), dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn((1, block_size, 1, d_qk), dtype=torch.bfloat16, device=device) / 10
    q = q.clamp(-1, 1)
    kv = kv.clamp(-1, 1)

    quant_kv = quant.quantize_k_cache(kv, quant.FP8KVCacheLayout.V32_FP8Sparse)
    indices = torch.randint(0, s_kv, (b, s_q, topk), dtype=torch.int32, device=device)
    indices[0, 0, -1] = -1
    topk_length = torch.tensor([topk - 1, topk], dtype=torch.int32, device=device)
    sm_scale = 0.5

    out, lse = slow_sparse_decode_fallback(
        q=q,
        k_cache=quant_kv,
        indices_in_kvcache=indices,
        topk_length=topk_length,
        head_dim_v=d_v,
        softmax_scale=sm_scale,
    )

    assert out.shape == (b, s_q, h_q, d_v)
    assert lse.shape == (b, h_q, s_q)
    assert torch.isfinite(out.float()).all()
    finite_lse = lse[torch.isfinite(lse)]
    assert finite_lse.numel() > 0

    print("CU126_SPARSE_DECODE_FALLBACK_OK")
    print(tuple(out.shape), tuple(lse.shape))


if __name__ == "__main__":
    main()
