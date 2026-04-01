import importlib.util
from pathlib import Path
import sys

import torch


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("glm5_flashdsa_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    model_path = Path("/work/twsuday816/glm5_reconstruction_test/modeling_glm5_flashdsa_reconstructed.py")
    mod = load_module(str(model_path))
    cfg = mod.Glm5ReconstructedConfig(
        vocab_size=1024,
        hidden_size=768,
        intermediate_size=1536,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=64,
        num_key_value_heads=1,
        max_position_embeddings=2048,
        attention_backend="flash_dsa",
        flashdsa_qk_dim=192,
        flashdsa_v_dim=256,
        flashdsa_kv_lora_dim=512,
        flashdsa_indexer_heads=32,
        flashdsa_indexer_head_dim=128,
        int4_group_size=32,
        dsa_topk=64,
        dsa_sink_tokens=4,
        dsa_local_window=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )

    model = mod.Glm5ReconstructedForCausalLM(cfg).cuda().to(dtype=torch.bfloat16).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 32), device="cuda")

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    print("FLASH_DSA_OK")
    print("LOGITS", tuple(out.logits.shape))
    print("LOGITS_FINITE", bool(torch.isfinite(out.logits).all().item()))


if __name__ == "__main__":
    main()
