#include "core/bvh.h"

#include <algorithm>

namespace pt {

namespace {

struct BuildItem {
    Aabb box;
    Vec3 centroid;
    int index;
};

int build_recursive(std::vector<BuildItem>& items, int start, int end, int max_leaf,
                    std::vector<BvhNode>& nodes) {
    int node_idx = (int)nodes.size();
    nodes.push_back(BvhNode{});

    Aabb bounds, cbounds;
    for (int i = start; i < end; ++i) {
        bounds.expand(items[i].box);
        cbounds.expand(items[i].centroid);
    }
    nodes[node_idx].lo = bounds.lo;
    nodes[node_idx].hi = bounds.hi;

    int count = end - start;
    int axis = cbounds.longest_axis();
    bool degenerate = cbounds.hi[axis] - cbounds.lo[axis] < 1e-7f;
    if (count <= max_leaf || degenerate) {
        nodes[node_idx].offset = start;
        nodes[node_idx].count = count;
        return node_idx;
    }

    int mid = start + count / 2;
    std::nth_element(items.begin() + start, items.begin() + mid, items.begin() + end,
                     [axis](const BuildItem& a, const BuildItem& b) { return a.centroid[axis] < b.centroid[axis]; });

    build_recursive(items, start, mid, max_leaf, nodes);  // left child == node_idx + 1
    int right = build_recursive(items, mid, end, max_leaf, nodes);
    nodes[node_idx].offset = right;
    nodes[node_idx].count = -(axis + 1);
    return node_idx;
}

}  // namespace

std::vector<BvhNode> build_bvh(std::vector<Primitive>& prims, int max_leaf_size) {
    std::vector<BuildItem> items(prims.size());
    for (size_t i = 0; i < prims.size(); ++i) {
        items[i].box = primitive_bounds(prims[i]);
        items[i].centroid = items[i].box.centroid();
        items[i].index = (int)i;
    }
    std::vector<BvhNode> nodes;
    nodes.reserve(2 * prims.size());
    build_recursive(items, 0, (int)items.size(), max_leaf_size, nodes);

    std::vector<Primitive> ordered(prims.size());
    for (size_t i = 0; i < items.size(); ++i) ordered[i] = prims[items[i].index];
    prims.swap(ordered);
    return nodes;
}

}  // namespace pt
