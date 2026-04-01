from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
HELPERS = ROOT / "csrc" / "kerutils" / "include" / "kerutils" / "device" / "sm80" / "helpers.cuh"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"forbidden {label}: {needle}")


def main() -> int:
    setup_text = SETUP.read_text(encoding="utf-8")
    helpers_text = HELPERS.read_text(encoding="utf-8")

    # cu126 path must automatically disable fragile newer sparse-fp8 sources.
    require(setup_text, 'DISABLE_SM90_SPARSE_FP8 = is_flag_set("FLASH_MLA_DISABLE_SM90_SPARSE_FP8")', "sparse-fp8 flag")
    require(setup_text, "if nvcc_major < 12 or (nvcc_major == 12 and nvcc_minor <= 6):", "cu126 gate")
    require(setup_text, "DISABLE_SM90_SPARSE_FP8 = True", "automatic cu126 sparse-fp8 disable")
    require(setup_text, "if not DISABLE_SM90 and not DISABLE_SM90_SPARSE_FP8:", "conditional sparse-fp8 sources")

    # shared-memory 128b helpers should avoid __int128_t + q bindings on cu126.
    require(helpers_text, "st.shared.v4.u32", "vectorized shared store bridge")
    require(helpers_text, "ld.shared.v4.u32", "vectorized shared load bridge")
    forbid(helpers_text, '"q"(val)', "legacy 128-bit q store operand")
    forbid(helpers_text, '"=q"(val)', "legacy 128-bit q load operand")
    forbid(helpers_text, "st.shared.b128 [%0], %1;", "legacy b128 shared store")
    forbid(helpers_text, "ld.shared.b128 %0, [%1];", "legacy b128 shared load")

    print("CU126_BRIDGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
