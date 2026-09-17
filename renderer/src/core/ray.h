#pragma once

#include "vec3.h"

namespace pt {

struct Ray {
    Vec3 origin;
    Vec3 dir;
    DEVFN Ray() {}
    DEVFN Ray(const Vec3& o, const Vec3& d) : origin(o), dir(d) {}
    DEVFN Vec3 at(float t) const { return origin + t * dir; }
};

// Axis-aligned bounding box.
struct Aabb {
    Vec3 lo;
    Vec3 hi;

    DEVFN Aabb() : lo(kInf), hi(-kInf) {}
    DEVFN Aabb(const Vec3& a, const Vec3& b) : lo(a), hi(b) {}

    DEVFN void expand(const Vec3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    DEVFN void expand(const Aabb& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    DEVFN Vec3 centroid() const { return 0.5f * (lo + hi); }
    DEVFN int longest_axis() const {
        Vec3 e = hi - lo;
        return (e.x > e.y && e.x > e.z) ? 0 : (e.y > e.z ? 1 : 2);
    }
};

// Slab test. inv_dir = 1/dir (component-wise). Returns true if the ray
// overlaps the box somewhere in [t_min, t_max].
DEVFN inline bool hit_aabb(const Vec3& lo, const Vec3& hi, const Vec3& origin,
                           const Vec3& inv_dir, float t_min, float t_max) {
    for (int a = 0; a < 3; ++a) {
        float t0 = (lo[a] - origin[a]) * inv_dir[a];
        float t1 = (hi[a] - origin[a]) * inv_dir[a];
        if (inv_dir[a] < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max < t_min) return false;
    }
    return true;
}

}  // namespace pt
