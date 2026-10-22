// Geometry tests for renderer/src/core/{camera,stereo}.h.
//
// The whole stereo pipeline rests on one identity: for a point at planar depth
// Z, the two eyes see it at columns that differ by exactly
//     d = ipd * focal_px / Z,     x_right = x_left - d,
// and at the *same* row. The dataset stores that d per pixel, the denoiser's
// consistency loss warps by it, and main.cpp computes it from a single depth
// buffer per eye. These tests check the identity against an independent
// projection of world points through each eye's camera, so a change to the
// camera basis, the viewport or focal_px cannot silently break the convention.
//
// Header-only: no CUDA, no scene files. Build with the `tests` CMake target.
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <string>

#include "core/stereo.h"

using pt::Camera;
using pt::CameraDesc;
using pt::StereoRig;
using pt::Vec3;

static int g_failed = 0;
static int g_checks = 0;

static void check(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL (line %d): %s\n", line, expr);
    }
}

static void check_near(double a, double b, double tol, const char* expr, int line) {
    ++g_checks;
    if (!(std::fabs(a - b) <= tol)) {
        ++g_failed;
        std::printf("  FAIL (line %d): %s   (%.6f vs %.6f, tol %.1e)\n", line, expr, a, b, tol);
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_NEAR(a, b, tol) check_near((a), (b), (tol), #a " ~= " #b, __LINE__)

// Inverse of Camera::ray: world point -> continuous pixel coordinates.
// Derived independently from the camera's stored basis, not from ray().
struct Pixel {
    float x, y;
};

static Pixel project(const Camera& c, const Vec3& p) {
    Vec3 v = p - c.origin;
    float z = dot(v, c.forward);
    Vec3 on_plane = v / z;  // image plane sits at distance 1 along forward
    Vec3 offset = on_plane - (c.top_left - c.origin);
    float viewport_w = c.horizontal.length();
    float viewport_h = c.vertical.length();
    Pixel px;
    px.x = dot(offset, c.right) / viewport_w * (float)c.width;
    px.y = -dot(offset, c.up) / viewport_h * (float)c.height;
    return px;
}

// ---------------------------------------------------------------- disparity_px

static void test_disparity_formula() {
    std::printf("disparity_formula\n");
    const float ipd = 0.063f, focal = 700.0f;
    CHECK_NEAR(pt::disparity_px(ipd, focal, 1.0f), ipd * focal, 1e-4);
    CHECK_NEAR(pt::disparity_px(ipd, focal, 2.0f), ipd * focal * 0.5f, 1e-4);

    // Inverse in depth: doubling Z halves the disparity.
    CHECK_NEAR(2.0 * pt::disparity_px(ipd, focal, 4.0f), pt::disparity_px(ipd, focal, 2.0f), 1e-4);
    // Nearer is larger - the sign convention the anaglyph preview relies on.
    CHECK(pt::disparity_px(ipd, focal, 0.5f) > pt::disparity_px(ipd, focal, 5.0f));

    // Background / miss pixels carry depth 0 and must not produce Inf or NaN.
    CHECK_NEAR(pt::disparity_px(ipd, focal, 0.0f), 0.0f, 0.0);
    CHECK_NEAR(pt::disparity_px(ipd, focal, 1e-9f), 0.0f, 0.0);
    CHECK_NEAR(pt::disparity_px(ipd, focal, -1.0f), 0.0f, 0.0);
    // Zero baseline (mono) means zero disparity at any depth.
    CHECK_NEAR(pt::disparity_px(0.0f, focal, 1.5f), 0.0f, 0.0);
}

// -------------------------------------------------------------------- the rig

static void test_rig_layout() {
    std::printf("rig_layout\n");
    CameraDesc d;
    const float ipd = 0.063f;
    const int W = 640, H = 480;
    StereoRig rig = pt::make_stereo_rig(d, W, H, ipd);
    Camera mono = pt::make_camera(d, W, H);

    CHECK_NEAR(rig.ipd, ipd, 0.0);
    // Eyes are ipd apart, centred on the scene camera, displaced along `right`.
    CHECK_NEAR((rig.right.origin - rig.left.origin).length(), ipd, 1e-6);
    CHECK_NEAR(dot(rig.right.origin - rig.left.origin, mono.right), ipd, 1e-6);
    Vec3 mid = 0.5f * (rig.left.origin + rig.right.origin);
    CHECK_NEAR((mid - mono.origin).length(), 0.0f, 1e-6);

    // Parallel axes: both eyes share the orientation and the intrinsics, which
    // is what lets one focal_px and one depth buffer describe both views.
    CHECK_NEAR(dot(rig.left.forward, rig.right.forward), 1.0f, 1e-6);
    CHECK_NEAR(dot(rig.left.up, rig.right.up), 1.0f, 1e-6);
    CHECK_NEAR(dot(rig.left.right, rig.right.right), 1.0f, 1e-6);
    CHECK_NEAR(rig.left.focal_px, rig.right.focal_px, 1e-4);
    CHECK_NEAR(rig.left.focal_px, mono.focal_px, 1e-4);
    CHECK(rig.left.width == W && rig.right.width == W);
    CHECK(rig.left.height == H && rig.right.height == H);

    // Square pixels: one focal length serves both axes.
    CHECK_NEAR(mono.focal_px, (float)H / mono.vertical.length(), 1e-3);
    CHECK_NEAR(mono.focal_px, (float)W / mono.horizontal.length(), 1e-3);

    // Right-handed, orthonormal basis.
    CHECK_NEAR(dot(mono.right, mono.forward), 0.0f, 1e-6);
    CHECK_NEAR(dot(mono.up, mono.forward), 0.0f, 1e-6);
    CHECK_NEAR(dot(mono.right, mono.up), 0.0f, 1e-6);
    CHECK_NEAR(mono.forward.length(), 1.0f, 1e-6);
}

// ------------------------------------------------------- ray() <-> project()

static void test_projection_round_trip() {
    std::printf("projection_round_trip\n");
    CameraDesc d;
    const int W = 320, H = 240;
    Camera c = pt::make_camera(d, W, H);

    const float pts[][2] = {{0.5f, 0.5f}, {160.0f, 120.0f}, {12.5f, 230.0f}, {319.5f, 0.5f}, {80.0f, 200.0f}};
    for (const auto& q : pts) {
        for (float t : {0.5f, 2.0f, 7.0f}) {
            Vec3 p = c.ray(q[0], q[1]).at(t);
            Pixel r = project(c, p);
            CHECK_NEAR(r.x, q[0], 2e-3);
            CHECK_NEAR(r.y, q[1], 2e-3);
        }
    }

    // The centre of the image looks at `lookat`.
    Pixel mid = project(c, d.lookat);
    CHECK_NEAR(mid.x, 0.5f * W, 2e-3);
    CHECK_NEAR(mid.y, 0.5f * H, 2e-3);

    // zdepth is the planar depth along the viewing axis, not the ray length.
    Vec3 p = c.ray(10.0f, 10.0f).at(3.0f);  // off-axis, so the two differ
    CHECK_NEAR(c.zdepth(p), dot(p - c.origin, c.forward), 1e-6);
    CHECK((p - c.origin).length() > c.zdepth(p));
}

// ------------------------------------------- the identity the pipeline uses

static void check_identity_for(const CameraDesc& d, int W, int H, float ipd, const Vec3* pts, int n) {
    StereoRig rig = pt::make_stereo_rig(d, W, H, ipd);
    for (int i = 0; i < n; ++i) {
        const Vec3& p = pts[i];
        Pixel l = project(rig.left, p);
        Pixel r = project(rig.right, p);

        // Both eyes measure the same planar depth: main.cpp derives the
        // disparity map from one depth buffer per eye using the shared focal_px.
        float zl = rig.left.zdepth(p);
        float zr = rig.right.zdepth(p);
        CHECK_NEAR(zl, zr, 1e-5);

        float d_expected = pt::disparity_px(ipd, rig.left.focal_px, zl);
        CHECK_NEAR(l.x - r.x, d_expected, 3e-3);   // x_right = x_left - d
        CHECK_NEAR(l.y, r.y, 1e-3);                // parallel axes: no vertical disparity
        CHECK(d_expected > 0.0f);                  // positive, as stored in the maps
    }
}

static void test_disparity_matches_projection() {
    std::printf("disparity_matches_projection\n");
    const Vec3 pts[] = {
        Vec3(0.0f, 0.5f, 0.0f),      // centre of the box
        Vec3(-0.4f, 0.1f, 0.3f),     // near, left and low
        Vec3(0.45f, 0.95f, -0.9f),   // far corner
        Vec3(0.2f, 0.3f, 0.8f),      // close to the camera
        Vec3(-0.25f, 0.75f, -0.45f),
    };
    const int n = (int)(sizeof(pts) / sizeof(pts[0]));

    // Default Cornell-box camera and IPD, at the dataset resolution.
    check_identity_for(CameraDesc(), 512, 512, 0.063f, pts, n);

    // Non-square aspect: focal_px comes from the vertical field of view, so a
    // wider image must not change the disparity of a given point.
    check_identity_for(CameraDesc(), 1280, 720, 0.063f, pts, n);

    // A rotated, tilted rig with a tilted up-vector: the identity is a property
    // of the parallel-axis construction, not of an axis-aligned camera.
    CameraDesc tilted;
    tilted.lookfrom = Vec3(1.1f, 1.3f, 1.7f);
    tilted.lookat = Vec3(-0.1f, 0.4f, -0.2f);
    tilted.vup = Vec3(0.15f, 0.95f, -0.1f);
    tilted.vfov_deg = 65.0f;
    check_identity_for(tilted, 800, 600, 0.1f, pts, n);

    // A wide VR-like IPD at a narrow field of view.
    CameraDesc narrow;
    narrow.vfov_deg = 20.0f;
    check_identity_for(narrow, 512, 512, 0.12f, pts, n);
}

static void test_disparity_scales_with_resolution() {
    std::printf("disparity_scales_with_resolution\n");
    CameraDesc d;
    const Vec3 p(0.1f, 0.4f, 0.2f);
    StereoRig small = pt::make_stereo_rig(d, 256, 256, 0.063f);
    StereoRig big = pt::make_stereo_rig(d, 512, 512, 0.063f);

    // Doubling the resolution doubles focal_px, so pixel disparity doubles
    // while the geometry is unchanged. Patch extraction depends on this.
    float ds = project(small.left, p).x - project(small.right, p).x;
    float db = project(big.left, p).x - project(big.right, p).x;
    CHECK_NEAR(2.0 * small.left.focal_px, big.left.focal_px, 1e-3);
    CHECK_NEAR(2.0 * ds, db, 5e-3);
}

static void test_eye_names() {
    std::printf("eye_names\n");
    // The names are the filename prefixes the dataset loader expects.
    CHECK(std::string(pt::eye_name(pt::EYE_LEFT)) == "left");
    CHECK(std::string(pt::eye_name(pt::EYE_RIGHT)) == "right");
    CHECK(std::string(pt::eye_name(pt::EYE_MONO)) == "mono");
}

int main() {
    std::printf("test_stereo_geometry\n");
    test_disparity_formula();
    test_rig_layout();
    test_projection_round_trip();
    test_disparity_matches_projection();
    test_disparity_scales_with_resolution();
    test_eye_names();
    std::printf("%s: %d checks, %d failed\n", g_failed ? "FAILED" : "ok", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
