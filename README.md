# GLM5-Forge

`GLM5-Forge` is the parent project for the reconstructed GLM-5 research stack.

It groups three things under one roof:

- `glm5/`
  - GLM-5 reconstruction variants
  - baseline reconstruction
  - MLA-compatible reconstruction
  - FlashDSA-oriented reconstruction
  - training and remote test scripts
- `flashmla_cu126_patch/`
  - the reduced `FlashMLA` compatibility patch set for `CUDA 12.6 + H100`
  - source patches, bridge helpers, and fallback test scripts

Project hierarchy:

- `GLM5-Forge` is the top-level project
- `glm5` is the model-reconstruction core
- `flashmla_cu126_patch` is supporting backend infrastructure

Current result:

- `MLA` path reconstructed and tested against native `FlashMLA`
- `FlashDSA` path reconstructed and trained
- training tradeoff observed:
  - `MLA` is faster
  - `DSA` is stronger

Real profiling and Tensorearch result:

- real 40-step profiling on Nano5:
  - `MLA`
    - final loss: `3.0897`
    - throughput: `1377.4 tok/s`
  - `DSA`
    - final loss: `2.2174`
    - throughput: `458.4 tok/s`
- `Tensorearch` on the real exported traces:
  - `MLA`
    - bottleneck: `blk0.q_proj`
    - obedience: `0.4619`
    - intelligence: `0.0808`
    - coupling: `0.0839`
  - `DSA`
    - bottleneck: `blk1.flash_dsa`
    - obedience: `0.5008`
    - intelligence: `0.0190`
    - coupling: `0.0578`

Interpretation:

- `DSA` is more target-aligned and converges better, but its pressure concentrates inside the sparse attention core
- `MLA` is faster, more flexible, and more strongly coupled around the latent attention path
- strongest `DSA` attribution: `indexer -> flash_dsa`
- strongest `MLA` attributions: `kv_latent -> flash_mla` and `flash_mla -> ffn`

So the current reconstruction is now supported by both:

- training behavior
- real architecture trace analysis
