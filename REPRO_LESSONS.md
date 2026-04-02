# GLM5-Forge Migration And Reproduction Notes

This document records the practical lessons from rebuilding the GLM-5 research stack from public materials, porting it onto the local/cluster environment, and validating MLA vs DSA behavior.

## 1. What Was Actually Reconstructed

The reconstruction was not a direct copy of an official full release. It was assembled from:

- GLM-5 paper constraints
- GLM-4.x / GLM-4.5 public implementation references
- Hugging Face `glm4_moe` references
- DeepSeek-V3 / FlashMLA implementation details
- observed runtime constraints from real kernels

The final project contains:

- baseline GLM-5-style reconstruction
- MLA-compatible reconstruction
- FlashDSA-oriented reconstruction
- reduced `FlashMLA` compatibility patches for `CUDA 12.6 + H100`

## 2. Core Reconstruction Strategy

The effective strategy was:

1. Reconstruct a runnable Transformer/MoE skeleton first.
2. Make MLA run natively against real FlashMLA constraints.
3. Use the MLA-compatible path as the substrate for DSA reconstruction.
4. Validate behavior with real forward passes before claiming architectural alignment.
5. Use small training runs to check whether the reconstructed tradeoff matches the paper direction.

This avoided a common failure mode: trying to reproduce the final paper claim before building a stable intermediate base.

## 3. Why MLA Had To Come First

GLM-5 is not just "DSA from scratch". The public evidence indicates:

- GLM-5 adopts DSA as the main sparse attention direction.
- But that DSA sits on top of an MLA-origin infrastructure/history.

That meant the shortest working path was:

- first align to MLA-compatible latent KV semantics
- then adapt that into DSA-style indexing and sparse routing

Trying to skip MLA and jump directly to DSA would have left too many hidden interface assumptions unresolved.

## 4. What FlashMLA Taught Us

FlashMLA exposed real architectural constraints that were not obvious from high-level descriptions.

For native sparse FlashMLA paths, the important constraints were:

- `h_q` must be `64` or `128`
- `d_qk` is not arbitrary
- `d_v` is fixed by kernel contract
- `topk` must obey kernel block assumptions
- KV tensors must follow MLA latent semantics, not generic packed `K||V`

This was important because runtime errors became protocol discovery signals, not just failures.

## 5. Reduced FlashMLA On CUDA 12.6

The upstream FlashMLA path was not cleanly reduced-build safe.

Problems discovered:

- `setup.py` feature gating did not fully match source inclusion
- header/API gating did not fully match source gating
- stale build artifacts could make header fixes appear ineffective
- multiple SM100 and sparse FP8 paths remained referenced even after partial disable flags

What made `cu126 + H100` work:

- remove `sm100` source paths from build
- remove `sm90 sparse_fp8` source paths from build
- patch `helpers.cuh` to bridge inline assembly constraints
- add API/header stubs for disabled bindings
- clean build artifacts before reinstalling

Result:

- reduced FlashMLA successfully built and imported on Nano5 under `CUDA 12.6`

## 6. Why LECAC Mattered

LECAC was useful because it provided an elegant intermediate language for DSA-style behavior when the exact internal formulas were not public.

Its value was not "copy code blindly". Its value was:

- offering a compact latent representation idea
- enabling a clean `INT4 -> BF16` bridge
- letting sparse routing behavior remain structurally meaningful
- avoiding a large amount of ad hoc fake quantization code

Without LECAC, the likely fallback would have been either:

- trivial stubs that break the architecture path at runtime
- or a large amount of arbitrary compression logic with weak interpretability

LECAC made it possible to express the intended DSA behavior in a principled approximation.

## 7. MLA vs DSA Reproduction Outcome

The reconstructed comparison now shows a stable tradeoff:

- MLA is faster
- DSA is stronger

Observed trend across `100 / 500 / 1000 / 3000 / 10000` training steps:

- MLA consistently delivered higher throughput
- DSA consistently delivered lower loss

By `10000` steps:

- MLA final loss: about `2.44`
- DSA final loss: about `0.53`
- MLA throughput: about `4543 tok/s`
- DSA throughput: about `2546 tok/s`

This is not proof of official exact reproduction, but it is strong evidence that the reconstructed architecture direction is aligned with the intended paper-level tradeoff.

## 8. What This Project Demonstrated

This work demonstrated that:

- incomplete public releases can still be reconstructed into a runnable research stack
- kernel/runtime constraints can be used to reverse-engineer hidden protocol assumptions
- paper reproduction sometimes requires infrastructure reconstruction, not just model code
- architecture reverse engineering benefits from building an instrumented intermediate base instead of aiming at the final target immediately

## 9. Practical Rules Learned

- Treat runtime errors as protocol hints.
- Reconstruct the base path before reconstructing the final sparse path.
- For third-party kernel libraries, reduced-build support must be verified, not assumed.
- Clean rebuilds matter whenever header-only changes are involved.
- When the exact math is not public, prefer a principled intermediate representation over random placeholders.
- Validate with real training behavior, not just successful forward passes.

## 10. Current Position

`GLM5-Forge` now contains:

- a runnable GLM-5 reconstruction base
- a native FlashMLA-backed MLA path
- a FlashDSA-oriented path aligned to the paper direction
- a working cu126 FlashMLA reduced patch set

That means the project has moved beyond "source reconstruction" and into "architecture study platform".

## 11. Real Profiling And Trace Analysis

After the model paths were stable, we exported real profiling traces from short Nano5 runs and fed them into `Tensorearch`.

Real 40-step profiling:

- `MLA`
  - final loss: `3.0897`
  - throughput: `1377.4 tok/s`
- `DSA`
  - final loss: `2.2174`
  - throughput: `458.4 tok/s`

`Tensorearch` on the real traces reported:

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

This gives a stronger conclusion than loss alone:

- `DSA` is more target-aligned and structurally tighter
- `MLA` is faster, more flexible, and more strongly coupled
- `DSA` pressure concentrates around `indexer -> flash_dsa`
- `MLA` pressure concentrates around `kv_latent -> flash_mla`

Another useful interpretation is behavioral:

- `DSA` shows higher obedience but lower structural intelligence than `MLA`
- this makes `DSA` feel more rigid, more disciplined, and more target-aligned
- `MLA` feels more fluid, more adaptive, and more structurally flexible

In practice, `DSA` behaves more like a disciplined sparse retrieval system, while `MLA` behaves more like a more free-flowing latent attention system.

This is why real trace export matters. Without it, the reconstruction only tells us what optimization does. With it, we also get a structural explanation for the observed tradeoff.

## 12. Real B200 Validation

The reconstruction was also pushed onto a real `B200` (`Blackwell`, `sm100`) machine on Vast.ai.

This mattered because it moved the project beyond:

- reduced `cu126` survival patches
- `cu128` container validation
- `cu129` full-ish upstream validation

and into an actual `sm100` environment.

Observed environment:

- GPU: `NVIDIA B200`
- driver: `595.45.04`
- CUDA: `13.2`
- torch: `2.10.0a0+a36e1d39eb.nv26.01.42222806`

The main build issue was again the CUDA 13 header layout:

- `cutlass.h` wanted `cuda/std/utility`
- CUDA 13 placed the needed headers under:
  - `/usr/local/cuda/targets/x86_64-linux/include/cccl`

Once that include path was added to `setup.py`, `FlashMLA`:

- built successfully
- imported successfully
- and ran native `GLM5 MLA` smoke successfully on `B200`

That gives a stronger conclusion than earlier stages:

- the reconstruction is no longer only validated on Hopper (`H100/H200`)
- it is now also validated on real `Blackwell/B200`

Short-run `100`-step result on `B200`:

- `MLA + FlashMLA`
  - final loss: `3.4788`
  - best loss: `2.8717`
  - throughput: `3832.0 tok/s`
- `FlashDSA`
  - final loss: `2.3179`
  - best loss: `1.6720`
  - throughput: `2574.7 tok/s`

So the high-level tradeoff remained stable even after moving to real `Blackwell`:

- `MLA` stayed faster
- `DSA` stayed stronger
