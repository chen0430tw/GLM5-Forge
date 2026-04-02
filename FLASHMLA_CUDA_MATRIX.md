# FlashMLA CUDA Matrix

This note records the actual `FlashMLA` status we observed across different CUDA toolchain versions on Nano5 and RunPod.

Test environment baselines:

- Nano5:
  - GPU: `H100`
  - Host PyTorch baseline: `2.6.0+cu124`
  - Cluster default toolkit line before containers: `cuda/12.6`
  - Container path:
    - `CUDA 12.8`: `/work/twsuday816/containers/ubuntu2404_cuda12.8.sif`
    - `CUDA 12.9`: `/work/twsuday816/containers/ubuntu2404_cuda12.9.sif`
    - `CUDA 13.0`: `/work/twsuday816/containers/ubuntu2404_cuda13.0.sif`
- RunPod:
  - GPU: `H200 NVL`
  - Driver: `570.195.03`
  - host-visible CUDA compatibility: `12.8`
  - container/toolkit line tested: `CUDA 13.0`
- Vast.ai:
  - GPU: `B200`
  - Driver: `595.45.04`
  - host-visible CUDA compatibility: `CUDA 13.2`
  - torch line tested: `2.10.0a0+a36e1d39eb.nv26.01.42222806`

## Summary

| CUDA line | Status | Notes |
| --- | --- | --- |
| `cu126` | `working (reduced build)` | required local compatibility patch set |
| `cu128` | `working (upstream-ish)` | container build works with `SM100` disabled |
| `cu129` | `working (full-ish upstream path)` | native `sm100/sm90` build, import, and `GLM5 MLA` smoke all passed |
| `cu130` | `mixed` | RunPod `H200` is runtime-blocked by driver compatibility, but Vast `B200` builds, imports, and runs `GLM5 MLA` successfully |

## CUDA 12.6

Status:

- `working`, but only after a reduced-build compatibility patch

What was needed:

- remove `sm100` compilation path
- remove `sm90 sparse_fp8` path
- patch `helpers.cuh`
- patch `api.cpp`
- patch `sparse_decode.h`
- patch `sparse_fwd.h`
- patch `flash_mla_interface.py`

Result:

- `FlashMLA` can `build + import` on Nano5 under the reduced `cu126` path
- this patch set is preserved in:
  - `flashmla_cu126_patch/`

Interpretation:

- `cu126` is the survival path
- valuable because cluster-wide `12.8+` was not immediately available

## CUDA 12.8

Status:

- `working` in container
- not full upstream happy path

What we observed:

- container toolkit is valid
- upstream build works after:
  - ensuring `cutlass` submodule exists
  - skipping redundant `git submodule` calls in `setup.py`
  - disabling `SM100`

Why `SM100` still needed to be disabled:

- upstream documentation suggests `12.8+`
- but upstream source still enforces `12.9+` for the `sm100` path

Result:

- `FlashMLA` imports successfully inside the `cu128` container
- `GLM5 MLA + native FlashMLA` short training was successfully run

Observed short-run result (`100` steps):

- `MLA + FlashMLA`
  - final loss: `3.1835`
  - best loss: `2.9980`
  - throughput: `2325.5 tok/s`
- `DSA`
  - final loss: `2.2580`
  - best loss: `1.7505`
  - throughput: `1865.8 tok/s`

Interpretation:

- `cu128` is the first clean container line that let us run native `FlashMLA`
- still not the full `sm100` upstream path

## CUDA 12.9

Status:

- `working`
- currently the strongest full-ish upstream path we have validated

What is confirmed:

- container exists and works
- `nvcc` is valid:
  - `Cuda compilation tools, release 12.9, V12.9.41`
- the `FlashMLA_cuda129_test` tree compiled and installed successfully
- native `sm100/sm90` kernel compilation completed
- `import flash_mla` succeeded
- `GLM5 MLA + cu129 FlashMLA` smoke succeeded

Important difference from `cu128`:

- `cu129` is the first line where we are letting the real `sm100` kernels compile instead of disabling them upfront

Observed smoke result:

- `FLASH_MLA_AVAILABLE True`
- `FLASH_MLA_CALL`
- `NUM_HEADS 64`
- `LOGITS (1, 32, 1024)`
- `LOGITS_FINITE True`

Current interpretation:

- `cu129` is the point where upstream `FlashMLA` becomes much closer to its intended full build path on Nano5

## CUDA 13.0

Status:

- toolchain is valid
- build path is now valid
- runtime remains blocked on the tested RunPod `H200` machine
- runtime is valid on the tested Vast `B200` machine

Confirmed:

- RunPod `H200 NVL` machine used for the cleanest `cu130` attempt
- toolkit is valid:
  - `Cuda compilation tools, release 13.0, V13.0.88`
- dedicated runtime created:
  - `torch 2.9.1+cu130`

What failed first:

- `FlashMLA` build under `torch 2.6.0+cu124`

Failure reason:

- PyTorch extension CUDA version guard:
  - detected CUDA `13.0`
  - PyTorch built against CUDA `12.4`
  - build rejected before real kernel compilation could proceed

What we fixed:

- created a dedicated `torch+cu130` environment
- disabled build isolation so `setup.py` could see installed `torch`
- installed `wheel` and `ninja`
- patched `setup.py` to add the CUDA 13 CCCL include path:
  - `/usr/local/cuda/targets/x86_64-linux/include/cccl`

What works now:

- `FlashMLA` builds successfully
- `FlashMLA` imports successfully
- on Vast `B200`, native runtime is also valid
- `GLM5 MLA + FlashMLA` smoke passes on Vast `B200`

What still does not work:

- `torch.cuda.is_available() == False`
- GPU tensors cannot be created in the `torch+cu130` venv
- the runtime warning says the effective driver compatibility is still too old for `cu130`

What works on Vast `B200`:

- `nvidia-smi`
  - `NVIDIA B200`
  - driver `595.45.04`
  - `CUDA 13.2`
- `flash_mla` builds successfully after adding:
  - `/usr/local/cuda/targets/x86_64-linux/include/cccl`
- `flash_mla` imports successfully
- `GLM5 MLA` native `FlashMLA` smoke succeeded with:
  - `FLASH_MLA_AVAILABLE True`
  - `FLASH_MLA_CALL`
  - `LOGITS_FINITE True`

Observed short-run result on Vast `B200` (`100` steps):

- `MLA + FlashMLA`
  - final loss: `3.4788`
  - best loss: `2.8717`
  - throughput: `3832.0 tok/s`
- `FlashDSA`
  - final loss: `2.3179`
  - best loss: `1.6720`
  - throughput: `2574.7 tok/s`

Interpretation:

- `cu130` is no longer blocked by `FlashMLA` source or build scripts
- the blocking factor on RunPod was host driver/runtime compatibility
- Vast `B200` provides the first confirmed `sm100` path where `cu130+`-class `FlashMLA` actually builds and runs end-to-end

## Practical Takeaways

- `cu126` proved that a reduced, compatibility-oriented path is possible and useful
- `cu128` proved that containerized native `FlashMLA` use is practical
- `cu129` is the most promising line for a fuller upstream build with `sm100`
- `cu130` requires a proper `torch+cu130` runtime, not a mixed old-host wheel stack

## Current Ranking

From most immediately useful to most ambitious:

1. `cu126 reduced build`
   - stable fallback
   - already validated
2. `cu128 container`
   - practical native `FlashMLA` path
   - already validated with real model training
3. `cu129 container`
   - fuller upstream path
   - already validated with import and `GLM5 MLA` smoke
4. `cu130+ on real B200`
   - highest ceiling on paper
   - now validated on Vast `B200` with build, import, and `GLM5 MLA`
5. `cu130 container on RunPod H200`
   - build/import validated
   - still runtime-blocked on that host
