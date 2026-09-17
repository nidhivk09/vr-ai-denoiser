// Portable per-pixel random number generation.
// PcgRng runs on CPU and GPU; the CUDA backend can alternatively use curand
// (see pathtrace_kernel.cu), which is what the project spec asks for.
#pragma once

#include "common.h"

namespace pt {

// 64-bit mix (splitmix64 finalizer) used to derive decorrelated seeds.
DEVFN inline uint64_t hash64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Seed for pixel `pixel` of eye `eye` in frame `frame`.
DEVFN inline uint64_t pixel_seed(uint64_t base, uint32_t frame, uint32_t eye, uint32_t pixel) {
    return hash64(hash64(hash64(base) ^ frame) ^ ((uint64_t)eye << 32 | pixel));
}

// PCG32 (O'Neill, pcg-random.org).
struct PcgRng {
    uint64_t state;
    uint64_t inc;

    DEVFN void seed(uint64_t s, uint64_t seq) {
        state = 0u;
        inc = (seq << 1u) | 1u;
        next_u32();
        state += s;
        next_u32();
    }
    DEVFN uint32_t next_u32() {
        uint64_t old = state;
        state = old * 6364136223846793005ull + inc;
        uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }
    // Uniform float in [0, 1).
    DEVFN float uniform() { return (float)(next_u32() >> 8) * (1.0f / 16777216.0f); }
};

}  // namespace pt
