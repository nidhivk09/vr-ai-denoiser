// Host-side scene container: parsing of .scene text files, built-in scenes,
// BVH construction and light list.
//
// .scene format (one statement per line, '#' starts a comment):
//   camera <lookfrom x y z> <lookat x y z> <vup x y z> <vfov_deg>
//   background r g b
//   material <name> lambertian r g b
//   material <name> metal r g b fuzz
//   material <name> dielectric ior
//   material <name> light r g b          (one-sided, emits along quad normal)
//   sphere cx cy cz radius <material>
//   quad qx qy qz ux uy uz vx vy vz <material>
//   box x0 y0 z0 x1 y1 z1 <material> [rotate_y_degrees]
#pragma once

#include <string>
#include <vector>

#include "core/camera.h"
#include "core/scene.h"

namespace pt {

class Scene {
public:
    std::string name;
    CameraDesc camera;
    Vec3 background = Vec3(0.0f);
    std::vector<Material> materials;
    std::vector<std::string> material_names;
    std::vector<Primitive> prims;
    std::vector<BvhNode> nodes;
    std::vector<int> lights;

    static Scene from_file(const std::string& path);
    static Scene from_string(const std::string& text, const std::string& name);
    // "cornell" (all diffuse, Phase 1 reference) or "cornell_glossy".
    static Scene builtin(const std::string& name);
    // Loads a builtin if `spec` starts with "builtin:", otherwise a file.
    static Scene load(const std::string& spec);

    // Builds the BVH and light list. Called by the loaders.
    void finalize();
    SceneView host_view() const;

    size_t bvh_bytes() const { return nodes.size() * sizeof(BvhNode); }
    size_t prim_bytes() const { return prims.size() * sizeof(Primitive); }

private:
    int material_index(const std::string& mat_name, int line) const;
    void add_box(const Vec3& a, const Vec3& b, int mat, float rotate_y_deg);
};

}  // namespace pt
