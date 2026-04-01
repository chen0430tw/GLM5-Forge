import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_interface_module():
    repo_root = Path(__file__).resolve().parents[1]
    interface_path = repo_root / "flash_mla" / "flash_mla_interface.py"

    fake_pkg = types.ModuleType("flash_mla")
    fake_cuda = types.ModuleType("flash_mla.cuda")

    def sparse_decode_fwd(*args, **kwargs):
        raise RuntimeError(
            "flash_mla reduced cu126 build does not include native sparse decode kernels; "
            "Python fallback should handle this path"
        )

    def _unused(*args, **kwargs):
        raise RuntimeError("This bypass script only supports sparse decode fallback testing")

    fake_cuda.sparse_decode_fwd = sparse_decode_fwd
    fake_cuda.dense_decode_fwd = _unused
    fake_cuda.sparse_prefill_fwd = _unused
    fake_cuda.dense_prefill_fwd = _unused
    fake_cuda.dense_prefill_bwd = _unused

    fake_pkg.cuda = fake_cuda

    sys.modules["flash_mla"] = fake_pkg
    sys.modules["flash_mla.cuda"] = fake_cuda

    spec = importlib.util.spec_from_file_location("flash_mla.flash_mla_interface", interface_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["flash_mla.flash_mla_interface"] = module
    spec.loader.exec_module(module)
    return module


def build_v32_sparse_k_cache(k_bf16: torch.Tensor) -> torch.Tensor:
    # V32 layout: 512 e4m3 + 4 fp32 scales + 64 bf16 rope = 656 bytes/token
    num_blocks, block_size, h_k, d_qk = k_bf16.shape
    assert h_k == 1 and d_qk == 576
    d_nope, tile_size, num_tiles = 512, 128, 4
    rope = k_bf16[..., d_nope:]

    out = torch.empty((num_blocks, block_size, 1, 656), dtype=torch.float8_e4m3fn, device=k_bf16.device)
    packed = out.view(num_blocks, block_size, 656)

    for tile_idx in range(num_tiles):
        start = tile_idx * tile_size
        end = start + tile_size
        chunk = k_bf16[..., start:end].float().squeeze(2)
        scale = torch.abs(chunk).amax(dim=-1).clamp_min(1e-4) / 448.0
        q = (chunk / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        packed[..., start:end] = q
        packed[..., d_nope + tile_idx * 4 : d_nope + (tile_idx + 1) * 4].view(torch.float32)[:] = scale.unsqueeze(-1)

    packed[..., d_nope + num_tiles * 4 :].view(torch.bfloat16)[:] = rope.squeeze(2)
    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bypass test")

    iface = load_interface_module()

    b, s_q, h_q, d_qk, d_v = 1, 2, 8, 576, 512
    block_size = 64
    topk = 4

    q = torch.randn((b, s_q, h_q, d_qk), dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn((1, block_size, 1, d_qk), dtype=torch.bfloat16, device="cuda") / 10
    k_cache = build_v32_sparse_k_cache(kv)
    indices = torch.tensor([[[0, 1, 2, -1], [3, 4, 5, 6]]], dtype=torch.int32, device="cuda")
    meta, _ = iface.get_mla_metadata()

    out, lse = iface.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=meta,
        num_splits=None,
        softmax_scale=0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        topk_length=None,
        extra_topk_length=None,
    )

    assert out.shape == (b, s_q, h_q, d_v)
    assert lse.shape == (b, h_q, s_q)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse[torch.isfinite(lse)]).all()

    print("CU126_BYPASS_OK")
    print(tuple(out.shape), tuple(lse.shape))


if __name__ == "__main__":
    main()
