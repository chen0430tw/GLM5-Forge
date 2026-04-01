import argparse
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from modeling_glm5_mla_reconstructed import (
    Glm5ReconstructedConfig as MlaConfig,
    Glm5ReconstructedForCausalLM as MlaModel,
)
from modeling_glm5_flashdsa_reconstructed import (
    Glm5ReconstructedConfig as DsaConfig,
    Glm5ReconstructedForCausalLM as DsaModel,
)


CORPUS = (
    "GLM-5 reconstructs sparse reasoning paths.\n"
    "MLA compresses latent KV while DSA retrieves important tokens.\n"
    "FlashMLA accelerates compatible MLA prefill kernels on H100.\n"
    "FlashDSA uses deterministic top-k routing with an indexer.\n"
    "We compare MLA and DSA on the same tiny corpus.\n"
    "The goal is to observe loss, throughput, and stability.\n"
)


def encode_text(text: str) -> list[int]:
    return [b for b in text.encode("utf-8")]


@dataclass
class BatchSource:
    tokens: torch.Tensor
    seq_len: int
    device: torch.device

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - self.seq_len - 1
        start = torch.randint(0, max_start + 1, (1,), device=self.device).item()
        window = self.tokens[start : start + self.seq_len + 1]
        x = window[:-1].unsqueeze(0)
        y = window[1:].unsqueeze(0)
        return x, y


def build_model(args: argparse.Namespace, device: torch.device, train_dtype: torch.dtype) -> torch.nn.Module:
    common = dict(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.seq_len,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
        use_cache=False,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        num_experts_per_tok=args.num_experts_per_tok,
    )
    if args.variant == "mla":
        cfg = MlaConfig(
            **common,
            num_attention_heads=64,
            num_key_value_heads=1,
            use_latent_kv=True,
            attention_backend="flash_mla",
            mla_qk_dim=576,
            mla_v_dim=512,
            dsa_topk=128,
        )
        model = MlaModel(cfg)
    else:
        cfg = DsaConfig(
            **common,
            num_attention_heads=64,
            num_key_value_heads=1,
            use_latent_kv=True,
            attention_backend="flash_dsa",
            flashdsa_qk_dim=192,
            flashdsa_v_dim=256,
            flashdsa_kv_lora_dim=512,
            flashdsa_indexer_heads=32,
            flashdsa_indexer_head_dim=128,
            int4_group_size=32,
            dsa_topk=64,
        )
        model = DsaModel(cfg)
    return model.to(device=device, dtype=train_dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLA and FlashDSA GLM-5 variants on a shared tiny corpus.")
    parser.add_argument("--variant", choices=["mla", "dsa"], required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--moe-intermediate-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-routed-experts", type=int, default=8)
    parser.add_argument("--n-shared-experts", type=int, default=1)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    train_dtype = dtype_map[args.dtype]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device, train_dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    token_ids = encode_text(CORPUS * 128)
    data = torch.tensor(token_ids, dtype=torch.long, device=device)
    batch_source = BatchSource(tokens=data, seq_len=args.seq_len, device=device)

    history: list[dict[str, float]] = []
    start_wall = time.perf_counter()
    total_tokens = 0

    amp_enabled = device.type == "cuda" and train_dtype != torch.float32
    autocast_dtype = train_dtype if amp_enabled else None

    for step in range(1, args.steps + 1):
        x, y = batch_source.get_batch()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            out = model(input_ids=x, labels=y)
            loss = out.loss
        loss.backward()
        optimizer.step()

        total_tokens += x.numel()
        elapsed = max(time.perf_counter() - start_wall, 1e-6)
        tok_s = total_tokens / elapsed
        rec = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "tok_s": tok_s,
        }
        history.append(rec)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"[{args.variant}] step={step} loss={rec['loss']:.4f} tok/s={tok_s:.1f}", flush=True)

    result = {
        "variant": args.variant,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_layers": args.num_layers,
        "dtype": args.dtype,
        "device": str(device),
        "final_loss": history[-1]["loss"],
        "best_loss": min(item["loss"] for item in history),
        "avg_last5_loss": sum(item["loss"] for item in history[-min(5, len(history)):]) / min(5, len(history)),
        "final_tok_s": history[-1]["tok_s"],
        "history": history,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"RESULT_JSON {out_path}", flush=True)
    print(
        f"RESULT variant={args.variant} final_loss={result['final_loss']:.4f} "
        f"best_loss={result['best_loss']:.4f} tok/s={result['final_tok_s']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
