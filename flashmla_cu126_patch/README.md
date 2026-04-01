# cu126 FlashMLA Patch Bundle

This bundle contains the final reduced-build patch set that made FlashMLA import successfully on Nano5 with CUDA 12.6 and H100.

Included files:
- setup.py
- api.cpp
- sparse_decode.h
- sparse_fwd.h
- helpers.cuh
- flash_mla_interface.py
- check_cu126_bridge.py
- test_cu126_sparse_decode_fallback.py
- run_cu126_sparse_bypass.py

Patch intent:
- disable SM100 sources and bindings
- disable SM90 sparse_fp8 sources and bindings
- keep SM90 dense decode and sparse prefill paths
- bridge cu126 shared-memory asm incompatibility in helpers.cuh
- provide Python sparse decode fallback via BF16 path when native sparse decode is unavailable

Verified result on Nano5:
- wheel builds on CUDA 12.6 + gcc 12.5.0 + H100
- import flash_mla succeeds
