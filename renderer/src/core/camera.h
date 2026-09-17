// Pinhole camera with square pixels. Pixel (0,0) is the top-left corner.
#pragma once

#include "ray.h"

namespace pt {

struct CameraDesc {
    Vec3 lookfrom = Vec3(0.0f, 0.5f, 1.4f);
    Vec3 lookat = Vec3(0.0f, 0.5f, 0.0f);
    Vec3 vup = Vec3(0.0f, 1.0f, 0.0f);
    float vfov_deg = 40.0f;
};

struct Camera {
    Vec3 origin;
    Vec3 top_left;    // point on the image plane (distance 1) at pixel corner (0,0)
    Vec3 horizontal;  // full image width vector on the image plane
    Vec3 vertical;    // full image height vector (points up)
    Vec3 right, up, forward;
    int width, height;
    float focal_px;   // focal length in pixels (same for x and y)

    // (px, py) are continuous pixel coordinates in [0,W) x [0,H).
    DEVFN Ray ray(float px, float py) const {
        Vec3 target = top_left + (px / (float)width) * horizontal - (py / (float)height) * vertical;
        return Ray(origin, normalize(target - origin));
    }

    // Planar depth (distance along the viewing axis), used for disparity.
    DEVFN float zdepth(const Vec3& p) const { return dot(p - origin, forward); }
};

// `eye_offset` shifts the camera along its right vector (used for stereo).
inline Camera make_camera(const CameraDesc& d, int width, int height, float eye_offset = 0.0f) {
    Camera c;
    c.width = width;
    c.height = height;
    float theta = d.vfov_deg * kPi / 180.0f;
    float viewport_h = 2.0f * std::tan(theta * 0.5f);
    float viewport_w = viewport_h * (float)width / (float)height;
    c.forward = normalize(d.lookat - d.lookfrom);
    c.right = normalize(cross(c.forward, d.vup));
    c.up = cross(c.right, c.forward);
    c.origin = d.lookfrom + eye_offset * c.right;
    c.horizontal = viewport_w * c.right;
    c.vertical = viewport_h * c.up;
    c.top_left = c.origin + c.forward - 0.5f * c.horizontal + 0.5f * c.vertical;
    c.focal_px = (float)height / viewport_h;
    return c;
}

}  // namespace pt
