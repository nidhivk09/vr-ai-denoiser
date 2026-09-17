#include "scene_loader.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/bvh.h"

namespace pt {

namespace {

const char* kCornell = R"(# Cornell box, 1 unit wide. All-diffuse Phase 1 reference scene.
camera 0 0.5 1.4   0 0.5 0   0 1 0   40
background 0 0 0
material white lambertian 0.73 0.73 0.73
material red   lambertian 0.65 0.05 0.05
material green lambertian 0.12 0.45 0.15
material blue  lambertian 0.15 0.25 0.65
material light light 12 12 12
# walls: floor, ceiling, back, left (red), right (green)
quad -0.5 0 0      1 0 0    0 0 -1   white
quad -0.5 1 0      1 0 0    0 0 -1   white
quad -0.5 0 -1     1 0 0    0 1 0    white
quad -0.5 0 0      0 0 -1   0 1 0    red
quad  0.5 0 0      0 0 -1   0 1 0    green
# ceiling light, normal points down
quad -0.125 0.999 -0.4   0 0 -0.2   0.25 0 0   light
sphere -0.2 0.18 -0.6   0.18   white
sphere  0.22 0.13 -0.35 0.13   blue
box 0.05 0 -0.85  0.3 0.35 -0.6  white 20
)";

const char* kCornellGlossy = R"(# Cornell box with glossy metal and glass spheres.
camera 0 0.5 1.4   0 0.5 0   0 1 0   40
background 0 0 0
material white lambertian 0.73 0.73 0.73
material red   lambertian 0.65 0.05 0.05
material green lambertian 0.12 0.45 0.15
material gold  metal 0.9 0.7 0.3 0.15
material glass dielectric 1.5
material light light 12 12 12
quad -0.5 0 0      1 0 0    0 0 -1   white
quad -0.5 1 0      1 0 0    0 0 -1   white
quad -0.5 0 -1     1 0 0    0 1 0    white
quad -0.5 0 0      0 0 -1   0 1 0    red
quad  0.5 0 0      0 0 -1   0 1 0    green
quad -0.125 0.999 -0.4   0 0 -0.2   0.25 0 0   light
sphere -0.2 0.18 -0.6   0.18   gold
sphere  0.2 0.14 -0.3   0.14   glass
box 0.05 0 -0.85  0.3 0.35 -0.6  white 20
)";

Vec3 read_vec3(std::istringstream& is) {
    float x, y, z;
    is >> x >> y >> z;
    return Vec3(x, y, z);
}

Vec3 rotate_y(const Vec3& p, float c, float s) { return Vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); }

}  // namespace

int Scene::material_index(const std::string& mat_name, int line) const {
    for (size_t i = 0; i < material_names.size(); ++i)
        if (material_names[i] == mat_name) return (int)i;
    throw std::runtime_error("line " + std::to_string(line) + ": unknown material '" + mat_name + "'");
}

void Scene::add_box(const Vec3& a, const Vec3& b, int mat, float rotate_y_deg) {
    Vec3 lo = vmin(a, b), hi = vmax(a, b);
    Vec3 dx(hi.x - lo.x, 0, 0), dy(0, hi.y - lo.y, 0), dz(0, 0, hi.z - lo.z);
    struct Face { Vec3 q, u, v; };
    Face faces[6] = {
        {Vec3(lo.x, lo.y, hi.z), dx, dy},   // front
        {Vec3(hi.x, lo.y, hi.z), -dz, dy},  // right
        {Vec3(hi.x, lo.y, lo.z), -dx, dy},  // back
        {Vec3(lo.x, lo.y, lo.z), dz, dy},   // left
        {Vec3(lo.x, hi.y, hi.z), dx, -dz},  // top
        {Vec3(lo.x, lo.y, lo.z), dx, dz},   // bottom
    };
    float th = rotate_y_deg * kPi / 180.0f;
    float c = std::cos(th), s = std::sin(th);
    Vec3 center = 0.5f * (lo + hi);
    for (const Face& f : faces) {
        Vec3 q = center + rotate_y(f.q - center, c, s);
        prims.push_back(make_quad(q, rotate_y(f.u, c, s), rotate_y(f.v, c, s), mat));
    }
}

Scene Scene::from_string(const std::string& text, const std::string& scene_name) {
    Scene sc;
    sc.name = scene_name;
    std::istringstream in(text);
    std::string raw;
    int line_no = 0;
    while (std::getline(in, raw)) {
        ++line_no;
        size_t hash = raw.find('#');
        if (hash != std::string::npos) raw.resize(hash);
        std::istringstream is(raw);
        std::string cmd;
        if (!(is >> cmd)) continue;

        if (cmd == "camera") {
            sc.camera.lookfrom = read_vec3(is);
            sc.camera.lookat = read_vec3(is);
            sc.camera.vup = read_vec3(is);
            is >> sc.camera.vfov_deg;
        } else if (cmd == "background") {
            sc.background = read_vec3(is);
        } else if (cmd == "material") {
            std::string mname, type;
            is >> mname >> type;
            Material m{};
            m.ior = 1.5f;
            if (type == "lambertian") {
                m.type = MAT_LAMBERTIAN;
                m.albedo = read_vec3(is);
            } else if (type == "metal") {
                m.type = MAT_METAL;
                m.albedo = read_vec3(is);
                is >> m.fuzz;
            } else if (type == "dielectric") {
                m.type = MAT_DIELECTRIC;
                m.albedo = Vec3(1.0f);
                is >> m.ior;
            } else if (type == "light") {
                m.type = MAT_LIGHT;
                m.emission = read_vec3(is);
            } else {
                throw std::runtime_error("line " + std::to_string(line_no) + ": unknown material type '" + type + "'");
            }
            sc.material_names.push_back(mname);
            sc.materials.push_back(m);
        } else if (cmd == "sphere") {
            Vec3 c = read_vec3(is);
            float r;
            std::string mat;
            is >> r >> mat;
            sc.prims.push_back(make_sphere(c, r, sc.material_index(mat, line_no)));
        } else if (cmd == "quad") {
            Vec3 q = read_vec3(is), u = read_vec3(is), v = read_vec3(is);
            std::string mat;
            is >> mat;
            sc.prims.push_back(make_quad(q, u, v, sc.material_index(mat, line_no)));
        } else if (cmd == "box") {
            Vec3 a = read_vec3(is), b = read_vec3(is);
            std::string mat;
            float rot = 0.0f;
            is >> mat;
            if (!(is >> rot)) rot = 0.0f;
            sc.add_box(a, b, sc.material_index(mat, line_no), rot);
            continue;
        } else {
            throw std::runtime_error("line " + std::to_string(line_no) + ": unknown statement '" + cmd + "'");
        }
        if (is.fail()) throw std::runtime_error("line " + std::to_string(line_no) + ": could not parse '" + cmd + "'");
    }
    sc.finalize();
    return sc;
}

Scene Scene::from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open scene file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string name = path;
    size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return from_string(ss.str(), name);
}

Scene Scene::builtin(const std::string& n) {
    if (n == "cornell") return from_string(kCornell, n);
    if (n == "cornell_glossy") return from_string(kCornellGlossy, n);
    throw std::runtime_error("unknown builtin scene '" + n + "' (use cornell or cornell_glossy)");
}

Scene Scene::load(const std::string& spec) {
    const std::string prefix = "builtin:";
    if (spec.compare(0, prefix.size(), prefix) == 0) return builtin(spec.substr(prefix.size()));
    return from_file(spec);
}

void Scene::finalize() {
    if (prims.empty()) throw std::runtime_error("scene '" + name + "' has no primitives");
    nodes = build_bvh(prims);
    lights.clear();
    for (size_t i = 0; i < prims.size(); ++i) {
        const Primitive& p = prims[i];
        if (materials[p.material].type == MAT_LIGHT) {
            if (p.type != PRIM_QUAD) throw std::runtime_error("only quads can be lights");
            lights.push_back((int)i);
        }
    }
}

SceneView Scene::host_view() const {
    SceneView v;
    v.prims = prims.data();
    v.num_prims = (int)prims.size();
    v.nodes = nodes.data();
    v.num_nodes = (int)nodes.size();
    v.materials = materials.data();
    v.num_materials = (int)materials.size();
    v.lights = lights.empty() ? nullptr : lights.data();
    v.num_lights = (int)lights.size();
    v.background = background;
    return v;
}

}  // namespace pt
