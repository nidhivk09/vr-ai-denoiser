// The path tracing integrator shared by the CPU and CUDA backends.
//
// Loosely follows "Ray Tracing in One Weekend" (Shirley) but is iterative
// instead of recursive (GPU kernels cannot recurse) and adds next-event
// estimation (explicit light sampling) so low-spp renders are usable.
#pragma once

#include "bvh.h"
#include "camera.h"
#include "rng.h"
#include "scene.h"

namespace pt {

struct RenderParams {
    int spp = 8;
    int max_depth = 8;
    bool nee = true;               // next-event estimation (light sampling)
    bool russian_roulette = true;
    int rr_start_depth = 3;
    bool collect_stats = false;    // per-pixel work counters (for divergence analysis)
    float clamp_indirect = 0.0f;   // >0: clamp indirect contributions (firefly suppression, biased)
};

// Depth written for rays that escape the scene.
constexpr float kFarDepth = 1.0e4f;

DEVFN inline Vec3 sample_cosine_hemisphere(const Vec3& n, float u1, float u2) {
    Vec3 t, b;
    onb(n, t, b);
    float r = std::sqrt(u1);
    float phi = 2.0f * kPi * u2;
    return normalize(r * std::cos(phi) * t + r * std::sin(phi) * b +
                     std::sqrt(maxf(0.0f, 1.0f - u1)) * n);
}

DEVFN inline Vec3 sample_unit_ball(float u1, float u2, float u3) {
    float z = 1.0f - 2.0f * u1;
    float r = std::sqrt(maxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * kPi * u2;
    return std::cbrt(u3) * Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

DEVFN inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    float m = 1.0f - cosine;
    return r0 + (1.0f - r0) * m * m * m * m * m;
}

// Direct lighting at a Lambertian point from one uniformly chosen area light.
DEVFN inline Vec3 sample_direct(const SceneView& s, const HitRecord& h, const Vec3& albedo,
                                float u_light, float u1, float u2, PathStats* stats) {
    if (s.num_lights == 0) return Vec3(0.0f);
    int li = (int)(u_light * (float)s.num_lights);
    if (li >= s.num_lights) li = s.num_lights - 1;
    const Primitive& light = s.prims[s.lights[li]];

    Vec3 lp = light.q + u1 * light.u + u2 * light.v;
    Vec3 to_light = lp - h.p;
    float dist2 = to_light.length_squared();
    float dist = std::sqrt(dist2);
    Vec3 wi = to_light / dist;

    float cos_surf = dot(h.normal, wi);
    if (cos_surf <= 0.0f) return Vec3(0.0f);
    float cos_light = dot(light.normal, -wi);  // lights are one-sided
    if (cos_light <= 0.0f) return Vec3(0.0f);

    if (stats) stats->shadow_rays++;
    Ray shadow(h.p + kRayEps * h.normal, wi);
    if (occluded(s, shadow, 0.0f, dist * (1.0f - 1e-3f), stats)) return Vec3(0.0f);

    // Lambertian BRDF (albedo/pi) over the solid-angle pdf of the light sample.
    const Material& lm = s.materials[light.material];
    float geom = cos_surf * cos_light * light.area * (float)s.num_lights / dist2;
    return lm.emission * albedo * (kInvPi * geom);
}

// Clamps the max component of an indirect contribution (depth >= 1).
DEVFN inline Vec3 clamp_contribution(const Vec3& c, int depth, float limit) {
    if (limit <= 0.0f || depth == 0) return c;
    float m = c.max_component();
    return m > limit ? c * (limit / m) : c;
}

// First-hit feature buffers written alongside the radiance.
struct PathAux {
    Vec3 normal;  // world-space shading normal in [-1,1]
    Vec3 albedo;
};

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <class Rng>
DEVFN inline Vec3 trace_path(const SceneView& s, Ray ray, Rng& rng, const RenderParams& p,
                             PathAux& aux, PathStats* stats) {
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);
    bool count_emission = true;  // false right after a diffuse bounce with NEE
    aux.normal = Vec3(0.0f);
    aux.albedo = Vec3(0.0f);

    for (int depth = 0; depth < p.max_depth; ++depth) {
        HitRecord h;
        if (stats) stats->bounces++;
        if (!intersect(s, ray, kRayEps, kInf, h, stats)) {
            radiance += clamp_contribution(throughput * s.background, depth, p.clamp_indirect);
            if (depth == 0) aux.albedo = vmin(s.background, Vec3(1.0f));
            break;
        }
        const Material& m = s.materials[h.material];

        if (depth == 0) {
            aux.normal = h.normal;
            if (m.type == MAT_LIGHT) aux.albedo = vmin(m.emission, Vec3(1.0f));
            else if (m.type == MAT_DIELECTRIC) aux.albedo = Vec3(1.0f);
            else aux.albedo = m.albedo;
        }

        if (m.type == MAT_LIGHT) {
            if (count_emission && h.front_face)
                radiance += clamp_contribution(throughput * m.emission, depth, p.clamp_indirect);
            break;
        }

        Vec3 origin;
        Vec3 dir;
        if (m.type == MAT_LAMBERTIAN) {
            if (p.nee) {
                float ul = rng.uniform();
                float u1 = rng.uniform();
                float u2 = rng.uniform();
                radiance += clamp_contribution(throughput * sample_direct(s, h, m.albedo, ul, u1, u2, stats),
                                               depth, p.clamp_indirect);
            }
            float u1 = rng.uniform();
            float u2 = rng.uniform();
            dir = sample_cosine_hemisphere(h.normal, u1, u2);
            origin = h.p + kRayEps * h.normal;
            // cosine-weighted sampling: brdf * cos / pdf == albedo
            throughput *= m.albedo;
            count_emission = !p.nee;
        } else if (m.type == MAT_METAL) {
            float u1 = rng.uniform();
            float u2 = rng.uniform();
            float u3 = rng.uniform();
            dir = normalize(reflect(ray.dir, h.normal) + m.fuzz * sample_unit_ball(u1, u2, u3));
            if (dot(dir, h.normal) <= 0.0f) break;  // absorbed
            origin = h.p + kRayEps * h.normal;
            throughput *= m.albedo;
            count_emission = true;
        } else {  // MAT_DIELECTRIC
            float ratio = h.front_face ? (1.0f / m.ior) : m.ior;
            float cos_theta = minf(dot(-ray.dir, h.normal), 1.0f);
            float sin_theta = std::sqrt(maxf(0.0f, 1.0f - cos_theta * cos_theta));
            float u = rng.uniform();
            if (ratio * sin_theta > 1.0f || schlick(cos_theta, ratio) > u) {
                dir = reflect(ray.dir, h.normal);
                origin = h.p + kRayEps * h.normal;
            } else {
                dir = normalize(refract(ray.dir, h.normal, ratio));
                origin = h.p - kRayEps * h.normal;
            }
            count_emission = true;
        }
        ray = Ray(origin, dir);

        if (p.russian_roulette && depth >= p.rr_start_depth) {
            float q = clampf(throughput.max_component(), 0.05f, 0.95f);
            if (rng.uniform() > q) break;
            throughput /= q;
        }
    }
    return radiance;
}

// Renders one pixel with p.spp jittered samples. Writes 3 floats to each of
// color/normal/albedo and 1 float to depth. The normal is encoded to [0,1];
// depth is the planar depth of the (un-jittered) pixel-center ray.
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <class Rng>
DEVFN inline void render_pixel(const SceneView& s, const Camera& cam, int px, int py,
                               const RenderParams& p, Rng& rng, float* color, float* normal,
                               float* albedo, float* depth, PathStats* stats) {
    Vec3 c(0.0f), n(0.0f), a(0.0f);
    int valid = 0;
    for (int i = 0; i < p.spp; ++i) {
        float jx = rng.uniform();
        float jy = rng.uniform();
        Ray r = cam.ray((float)px + jx, (float)py + jy);
        PathAux aux;
        Vec3 li = trace_path(s, r, rng, p, aux, stats);
        if (!li.is_finite()) continue;  // drop NaN/Inf samples
        c += li;
        n += aux.normal;
        a += aux.albedo;
        ++valid;
    }
    if (valid > 0) {
        float inv = 1.0f / (float)valid;
        c *= inv;
        a *= inv;
    }
    float nl = n.length();
    n = nl > 1e-6f ? n / nl : Vec3(0.0f);

    color[0] = c.x; color[1] = c.y; color[2] = c.z;
    normal[0] = 0.5f * (n.x + 1.0f); normal[1] = 0.5f * (n.y + 1.0f); normal[2] = 0.5f * (n.z + 1.0f);
    albedo[0] = a.x; albedo[1] = a.y; albedo[2] = a.z;

    HitRecord h;
    Ray center = cam.ray((float)px + 0.5f, (float)py + 0.5f);
    depth[0] = intersect(s, center, kRayEps, kInf, h, nullptr) ? cam.zdepth(h.p) : kFarDepth;
}

}  // namespace pt
