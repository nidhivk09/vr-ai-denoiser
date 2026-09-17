// Minimal 3-vector, following "Ray Tracing in One Weekend" (Shirley).
#pragma once

#include "common.h"

namespace pt {

struct Vec3 {
    float x, y, z;

    DEVFN Vec3() : x(0), y(0), z(0) {}
    DEVFN Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    DEVFN explicit Vec3(float a) : x(a), y(a), z(a) {}

    DEVFN float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    DEVFN float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

    DEVFN Vec3 operator-() const { return Vec3(-x, -y, -z); }
    DEVFN Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    DEVFN Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    DEVFN Vec3& operator*=(const Vec3& o) { x *= o.x; y *= o.y; z *= o.z; return *this; }
    DEVFN Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    DEVFN Vec3& operator/=(float s) { return *this *= (1.0f / s); }

    DEVFN float length_squared() const { return x * x + y * y + z * z; }
    DEVFN float length() const { return std::sqrt(length_squared()); }
    DEVFN float max_component() const { return maxf(x, maxf(y, z)); }
    DEVFN bool is_finite() const { return finitef(x) && finitef(y) && finitef(z); }
};

DEVFN inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
DEVFN inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
DEVFN inline Vec3 operator*(const Vec3& a, const Vec3& b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
DEVFN inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
DEVFN inline Vec3 operator*(float s, const Vec3& a) { return a * s; }
DEVFN inline Vec3 operator/(const Vec3& a, float s) { return a * (1.0f / s); }

DEVFN inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
DEVFN inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
DEVFN inline Vec3 normalize(const Vec3& v) { return v / v.length(); }
DEVFN inline Vec3 vmin(const Vec3& a, const Vec3& b) { return Vec3(minf(a.x, b.x), minf(a.y, b.y), minf(a.z, b.z)); }
DEVFN inline Vec3 vmax(const Vec3& a, const Vec3& b) { return Vec3(maxf(a.x, b.x), maxf(a.y, b.y), maxf(a.z, b.z)); }
DEVFN inline Vec3 reflect(const Vec3& v, const Vec3& n) { return v - 2.0f * dot(v, n) * n; }

// Snell refraction of unit vector uv about normal n (both facing each other).
DEVFN inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = minf(dot(-uv, n), 1.0f);
    Vec3 r_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_par = -std::sqrt(std::fabs(1.0f - r_perp.length_squared())) * n;
    return r_perp + r_par;
}

// Orthonormal basis around unit vector n (Duff et al. 2017, branchless).
DEVFN inline void onb(const Vec3& n, Vec3& t, Vec3& b) {
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float bb = n.x * n.y * a;
    t = Vec3(1.0f + sign * n.x * n.x * a, sign * bb, -sign * n.x);
    b = Vec3(bb, sign + n.y * n.y * a, -n.y);
}

}  // namespace pt
