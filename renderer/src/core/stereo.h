// Parallel-axis stereo rig: two identical pinhole cameras separated by the
// interpupillary distance (IPD) along the camera's right vector.
//
// For a point at planar depth Z, the horizontal disparity in pixels is
//     d = ipd * focal_px / Z,   x_right = x_left - d.
#pragma once

#include "camera.h"

namespace pt {

enum Eye : int { EYE_MONO = 0, EYE_LEFT = 1, EYE_RIGHT = 2 };

inline const char* eye_name(int eye) {
    return eye == EYE_LEFT ? "left" : (eye == EYE_RIGHT ? "right" : "mono");
}

struct StereoRig {
    Camera left;
    Camera right;
    float ipd;
};

inline StereoRig make_stereo_rig(const CameraDesc& d, int width, int height, float ipd) {
    StereoRig rig;
    rig.ipd = ipd;
    rig.left = make_camera(d, width, height, -0.5f * ipd);
    rig.right = make_camera(d, width, height, +0.5f * ipd);
    return rig;
}

DEVFN inline float disparity_px(float ipd, float focal_px, float zdepth) {
    return zdepth > 1e-6f ? ipd * focal_px / zdepth : 0.0f;
}

}  // namespace pt
