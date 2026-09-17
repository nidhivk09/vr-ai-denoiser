// Host-side image buffers (interleaved HWC float32).
#pragma once

#include <string>
#include <vector>

#include "core/scene.h"

namespace pt {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;

    void resize(int w, int h, int c) {
        width = w;
        height = h;
        channels = c;
        data.assign((size_t)w * h * c, 0.0f);
    }
    float& at(int x, int y, int c) { return data[((size_t)y * width + x) * channels + c]; }
    float at(int x, int y, int c) const { return data[((size_t)y * width + x) * channels + c]; }
};

// All buffers produced for one eye.
struct FrameBuffers {
    Image color;   // linear HDR radiance
    Image normal;  // world-space normal encoded to [0,1]
    Image albedo;  // first-hit base color
    Image depth;   // planar depth (1 channel)
    std::vector<PathStats> stats;  // per-pixel work counters (optional)

    void resize(int w, int h, bool with_stats) {
        color.resize(w, h, 3);
        normal.resize(w, h, 3);
        albedo.resize(w, h, 3);
        depth.resize(w, h, 1);
        stats.assign(with_stats ? (size_t)w * h : 0, PathStats{});
    }
};

}  // namespace pt
