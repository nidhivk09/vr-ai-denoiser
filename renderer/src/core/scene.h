// Scene representation: materials, primitives (spheres, quads) and the
// device-copyable SceneView used by the path tracing kernel.
#pragma once

#include "ray.h"

namespace pt {

enum MaterialType : int {
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,       // glossy reflector, `fuzz` controls roughness
    MAT_DIELECTRIC = 2,  // glass, `ior`
    MAT_LIGHT = 3,       // one-sided area emitter (emits along the quad normal)
};

struct Material {
    int type;
    Vec3 albedo;
    Vec3 emission;
    float fuzz;
    float ior;
};

enum PrimitiveType : int { PRIM_SPHERE = 0, PRIM_QUAD = 1 };

// Flat primitive record (no virtual dispatch so it can live on the GPU).
// Quads are parallelograms Q + a*u + b*v with a,b in [0,1]; boxes and the
// Cornell-box walls are made of quads.
struct Primitive {
    int type;
    int material;
    // sphere
    Vec3 center;
    float radius;
    // quad
    Vec3 q, u, v;
    Vec3 normal;  // unit, along cross(u, v)
    Vec3 w;       // cross(u, v) / |cross(u, v)|^2
    float d;      // plane offset: dot(normal, q)
    float area;
};

struct HitRecord {
    float t;
    Vec3 p;
    Vec3 normal;      // shading normal, always facing against the incoming ray
    Vec3 geo_normal;  // outward geometric normal
    bool front_face;
    int material;
    int prim;
};

DEVFN inline Primitive make_sphere(const Vec3& c, float r, int mat) {
    Primitive p;
    p.type = PRIM_SPHERE;
    p.material = mat;
    p.center = c;
    p.radius = r;
    p.d = 0.0f;
    p.area = 4.0f * kPi * r * r;
    return p;
}

DEVFN inline Primitive make_quad(const Vec3& q, const Vec3& u, const Vec3& v, int mat) {
    Primitive p;
    p.type = PRIM_QUAD;
    p.material = mat;
    p.q = q;
    p.u = u;
    p.v = v;
    Vec3 n = cross(u, v);
    p.area = n.length();
    p.normal = n / p.area;
    p.d = dot(p.normal, q);
    p.w = n / dot(n, n);
    p.radius = 0.0f;
    return p;
}

inline Aabb primitive_bounds(const Primitive& p) {
    if (p.type == PRIM_SPHERE) {
        Vec3 r(p.radius);
        return Aabb(p.center - r, p.center + r);
    }
    Aabb b;
    b.expand(p.q);
    b.expand(p.q + p.u);
    b.expand(p.q + p.v);
    b.expand(p.q + p.u + p.v);
    Vec3 pad(1e-4f);
    b.lo -= pad;
    b.hi += pad;
    return b;
}

DEVFN inline void set_face_normal(HitRecord& h, const Ray& r, const Vec3& outward) {
    h.geo_normal = outward;
    h.front_face = dot(r.dir, outward) < 0.0f;
    h.normal = h.front_face ? outward : -outward;
}

DEVFN inline bool hit_sphere(const Primitive& s, const Ray& r, float t_min, float t_max, HitRecord& h) {
    Vec3 oc = s.center - r.origin;
    float a = r.dir.length_squared();
    float hb = dot(r.dir, oc);
    float c = oc.length_squared() - s.radius * s.radius;
    float disc = hb * hb - a * c;
    if (disc < 0.0f) return false;
    float sq = std::sqrt(disc);
    float root = (hb - sq) / a;
    if (root <= t_min || root >= t_max) {
        root = (hb + sq) / a;
        if (root <= t_min || root >= t_max) return false;
    }
    h.t = root;
    h.p = r.at(root);
    set_face_normal(h, r, (h.p - s.center) / s.radius);
    h.material = s.material;
    return true;
}

DEVFN inline bool hit_quad(const Primitive& qd, const Ray& r, float t_min, float t_max, HitRecord& h) {
    float denom = dot(qd.normal, r.dir);
    if (std::fabs(denom) < 1e-8f) return false;
    float t = (qd.d - dot(qd.normal, r.origin)) / denom;
    if (t <= t_min || t >= t_max) return false;
    Vec3 p = r.at(t);
    Vec3 planar = p - qd.q;
    float alpha = dot(qd.w, cross(planar, qd.v));
    float beta = dot(qd.w, cross(qd.u, planar));
    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) return false;
    h.t = t;
    h.p = p;
    set_face_normal(h, r, qd.normal);
    h.material = qd.material;
    return true;
}

DEVFN inline bool hit_primitive(const Primitive& p, const Ray& r, float t_min, float t_max, HitRecord& h) {
    return p.type == PRIM_SPHERE ? hit_sphere(p, r, t_min, t_max, h) : hit_quad(p, r, t_min, t_max, h);
}

// Flattened BVH node (32 bytes). Leaves: count > 0, primitives
// [offset, offset + count). Interior: count = -(split_axis + 1), the left child
// is the next node in the array and `offset` is the right child index.
struct BvhNode {
    Vec3 lo;
    int32_t offset;
    Vec3 hi;
    int32_t count;
};

// Plain view of all scene arrays. Pointers are host or device memory
// depending on the backend.
struct SceneView {
    const Primitive* prims;
    int num_prims;
    const BvhNode* nodes;
    int num_nodes;
    const Material* materials;
    int num_materials;
    const int* lights;  // indices of emissive quads in `prims`
    int num_lights;
    Vec3 background;
};

// Work counters for the parallel-architecture analysis (per pixel).
struct PathStats {
    uint32_t nodes_visited;   // BVH nodes whose AABB was tested
    uint32_t prims_tested;    // ray-primitive intersection tests
    uint32_t bounces;         // path segments traced
    uint32_t shadow_rays;     // next-event-estimation rays
};

}  // namespace pt
