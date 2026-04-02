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
- `FLASHMLA_CUDA_MATRIX.md` records the observed `FlashMLA` behavior across `cu126/cu128/cu129/cu130`

FlashMLA backend status:

- `cu126`
  - reduced compatibility build
- `cu128`
  - native container path works with `SM100` disabled
- `cu129`
  - full-ish upstream path validated with `sm100/sm90`, import, and `GLM5 MLA` smoke
- `cu130`
  - build and import now work after a CUDA 13 `cccl` include-path fix
  - runtime on the tested RunPod host is still blocked by driver compatibility
  - on Vast `B200` (`CUDA 13.2`, driver `595.45.04`), `FlashMLA` now builds, imports, and runs `GLM5 MLA` smoke successfully

Current result:

- `MLA` path reconstructed and tested against native `FlashMLA`
- `FlashDSA` path reconstructed and trained
- training tradeoff observed:
  - `MLA` is faster
  - `DSA` is stronger

Real B200 result:

- Vast `B200` / `CUDA 13.2`
  - `FlashMLA` built successfully with the CUDA 13 `cccl` include-path fix
  - `flash_mla` import passed
  - `GLM5 MLA` native `FlashMLA` smoke passed
- B200 `100`-step comparison:
  - `MLA + FlashMLA`
    - final loss: `3.4788`
    - best loss: `2.8717`
    - throughput: `3832.0 tok/s`
  - `FlashDSA`
    - final loss: `2.3179`
    - best loss: `1.6720`
    - throughput: `2574.7 tok/s`

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
