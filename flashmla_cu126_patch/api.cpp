#include <pybind11/pybind11.h>
#include <stdexcept>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

#ifdef FLASH_MLA_DISABLE_SM100_BINDINGS
void FMHACutlassSM100FwdRun(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                            at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                            int, float, int, int, bool) {
    throw std::runtime_error("flash_mla reduced cu126 build does not include SM100 dense prefill forward kernels");
}

void FMHACutlassSM100BwdRun(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                            at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                            at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                            int, float, int, int, bool) {
    throw std::runtime_error("flash_mla reduced cu126 build does not include SM100 dense prefill backward kernels");
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
