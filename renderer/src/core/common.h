// Shared definitions for code that compiles both as plain C++ (CPU backend)
// and as CUDA (GPU backend). Everything tagged DEVFN runs on host and device.
#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define DEVFN __host__ __device__
#else
#define DEVFN
#endif

namespace pt {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 1.0f / kPi;
constexpr float kInf = 1e30f;
constexpr float kRayEps = 1e-4f;   // self-intersection offset (scene is ~1 unit)
constexpr int kBvhStackSize = 64;

DEVFN inline float minf(float a, float b) { return a < b ? a : b; }
DEVFN inline float maxf(float a, float b) { return a > b ? a : b; }
DEVFN inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
DEVFN inline bool finitef(float x) { return (x - x) == 0.0f; }  // false for NaN/Inf

}  // namespace pt
