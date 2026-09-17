// Optional NVTX ranges so Nsight Systems shows named render/denoise stages.
#pragma once

#if defined(PT_USE_NVTX) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
namespace pt {
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};
}  // namespace pt
#define PT_NVTX_RANGE(name) ::pt::NvtxRange pt_nvtx_range_##__LINE__(name)
#else
#define PT_NVTX_RANGE(name) ((void)0)
#endif
