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
