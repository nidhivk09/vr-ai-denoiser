// Bounding volume hierarchy: host-side build (bvh.cpp) and device-side
// stack-based traversal.
#pragma once

#include <vector>

#include "scene.h"

namespace pt {

// Builds a binary BVH with median splits along the longest centroid axis.
// Reorders `prims` in place so leaves reference contiguous ranges.
std::vector<BvhNode> build_bvh(std::vector<Primitive>& prims, int max_leaf_size = 2);

// Closest hit. If `stats` is non-null, work counters are accumulated.
DEVFN inline bool intersect(const SceneView& s, const Ray& r, float t_min, float t_max,
                            HitRecord& rec, PathStats* stats) {
    Vec3 inv_dir(1.0f / r.dir.x, 1.0f / r.dir.y, 1.0f / r.dir.z);
    int dir_neg[3] = {inv_dir.x < 0.0f, inv_dir.y < 0.0f, inv_dir.z < 0.0f};
    int stack[kBvhStackSize];
    int sp = 0;
    int node_idx = 0;
    bool hit_any = false;
    float closest = t_max;
    uint32_t visited = 0, tested = 0;

    while (true) {
        const BvhNode& node = s.nodes[node_idx];
        ++visited;
        if (hit_aabb(node.lo, node.hi, r.origin, inv_dir, t_min, closest)) {
            if (node.count > 0) {
                for (int i = 0; i < node.count; ++i) {
                    int pi = node.offset + i;
                    ++tested;
                    if (hit_primitive(s.prims[pi], r, t_min, closest, rec)) {
                        hit_any = true;
                        closest = rec.t;
                        rec.prim = pi;
                    }
                }
                if (sp == 0) break;
                node_idx = stack[--sp];
            } else {
                // Visit the near child first, push the far one.
                int axis = -node.count - 1;
                if (dir_neg[axis]) {
                    stack[sp++] = node_idx + 1;
                    node_idx = node.offset;
                } else {
                    stack[sp++] = node.offset;
                    node_idx = node_idx + 1;
                }
            }
        } else {
            if (sp == 0) break;
            node_idx = stack[--sp];
        }
    }
    if (stats) {
        stats->nodes_visited += visited;
        stats->prims_tested += tested;
    }
    return hit_any;
}

// Any-hit query for shadow rays: true if something blocks (t_min, t_max).
DEVFN inline bool occluded(const SceneView& s, const Ray& r, float t_min, float t_max, PathStats* stats) {
    Vec3 inv_dir(1.0f / r.dir.x, 1.0f / r.dir.y, 1.0f / r.dir.z);
    int stack[kBvhStackSize];
    int sp = 0;
    int node_idx = 0;
    uint32_t visited = 0, tested = 0;
    bool blocked = false;
    HitRecord tmp;

    while (true) {
        const BvhNode& node = s.nodes[node_idx];
        ++visited;
        if (hit_aabb(node.lo, node.hi, r.origin, inv_dir, t_min, t_max)) {
            if (node.count > 0) {
                for (int i = 0; i < node.count; ++i) {
                    ++tested;
                    if (hit_primitive(s.prims[node.offset + i], r, t_min, t_max, tmp)) {
                        blocked = true;
                        break;
                    }
                }
                if (blocked || sp == 0) break;
                node_idx = stack[--sp];
            } else {
                stack[sp++] = node.offset;
                node_idx = node_idx + 1;
            }
        } else {
            if (sp == 0) break;
            node_idx = stack[--sp];
        }
    }
    if (stats) {
        stats->nodes_visited += visited;
        stats->prims_tested += tested;
    }
    return blocked;
}

}  // namespace pt
