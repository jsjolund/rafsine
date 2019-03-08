#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "BoundaryCondition.hpp"
#include "Primitives.hpp"

namespace NodeMode {
enum Enum { OVERWRITE, INTERSECT, FILL };
}

// A base class only holding a name string
class VoxelObject {
 public:
  std::string m_name;
  explicit VoxelObject(std::string name) : m_name(name) {}
};

// A plane of voxels
class VoxelQuad : public VoxelObject {
 public:
  // World coordinates (in m)
  vec3<real> m_origin;
  vec3<real> m_dir1;
  vec3<real> m_dir2;
  // Discretized coordinates and extents in lattice units
  vec3<int> m_voxOrigin;
  vec3<int> m_voxDir1;
  vec3<int> m_voxDir2;
  // Mode (fill, overwrite etc.)
  NodeMode::Enum m_mode;
  // Common boundary condition for voxels in this quad
  BoundaryCondition m_bc;
  // Intersections with other boundary conditions (can be empty)
  std::unordered_set<BoundaryCondition, std::hash<BoundaryCondition>>
      m_intersectingBcs;

  inline int getAreaVoxel() {
    int d1 = sqrt(m_voxDir1.x * m_voxDir1.x + m_voxDir1.y * m_voxDir1.y +
                  m_voxDir1.z * m_voxDir1.z);
    int d2 = sqrt(m_voxDir2.x * m_voxDir2.x + m_voxDir2.y * m_voxDir2.y +
                  m_voxDir2.z * m_voxDir2.z);
    return d1 * d2;
  }

  inline real getAreaReal() {
    real d1 =
        sqrt(m_dir1.x * m_dir1.x + m_dir1.y * m_dir1.y + m_dir1.z * m_dir1.z);
    real d2 =
        sqrt(m_dir2.x * m_dir2.x + m_dir2.y * m_dir2.y + m_dir2.z * m_dir2.z);
    return d1 * d2;
  }

  VoxelQuad()
      : VoxelObject(std::string()),
        m_origin(NaN, NaN, NaN),
        m_dir1(NaN, NaN, NaN),
        m_dir2(NaN, NaN, NaN),
        m_mode(NodeMode::Enum::FILL),
        m_bc(BoundaryCondition()),
        m_voxOrigin(0, 0, 0),
        m_voxDir1(0, 0, 0),
        m_voxDir2(0, 0, 0) {}

  VoxelQuad(std::string name, NodeMode::Enum mode, vec3<int> voxOrigin,
            vec3<int> voxDir1, vec3<int> voxDir2, vec3<int> normal,
            VoxelType::Enum type = VoxelType::Enum::WALL,
            real temperature = NaN,
            vec3<real> velocity = vec3<real>(NaN, NaN, NaN),
            vec3<int> rel_pos = vec3<int>(0, 0, 0),
            vec3<real> origin = vec3<real>(NaN, NaN, NaN),
            vec3<real> dir1 = vec3<real>(NaN, NaN, NaN),
            vec3<real> dir2 = vec3<real>(NaN, NaN, NaN))
      : VoxelObject(name),
        m_bc(BoundaryCondition(-1, type, temperature, velocity, normal,
                               rel_pos)),
        m_origin(origin),
        m_dir1(dir1),
        m_dir2(dir2),
        m_mode(mode),
        m_voxOrigin(voxOrigin),
        m_voxDir1(voxDir1),
        m_voxDir2(voxDir2) {}
};

namespace std {
template <>
struct hash<VoxelQuad> {
  std::size_t operator()(const VoxelQuad &quad) const {
    using std::hash;
    using std::size_t;
    size_t seed = 0;
    ::hash_combine(seed, quad.m_origin.x);
    ::hash_combine(seed, quad.m_origin.y);
    ::hash_combine(seed, quad.m_origin.z);
    ::hash_combine(seed, quad.m_dir1.x);
    ::hash_combine(seed, quad.m_dir1.y);
    ::hash_combine(seed, quad.m_dir1.z);
    ::hash_combine(seed, quad.m_dir2.x);
    ::hash_combine(seed, quad.m_dir2.y);
    ::hash_combine(seed, quad.m_dir2.z);
    ::hash_combine(seed, quad.m_mode);
    ::hash_combine(seed, quad.m_name);
    return seed;
  }
};
}  // namespace std

// A box of voxels
class VoxelBox : public VoxelObject {
 public:
  // World coordinates min/max (in m)
  vec3<real> m_min;
  vec3<real> m_max;
  // World coordinates in voxel units
  vec3<int> m_voxMin;
  vec3<int> m_voxMax;
  // NaN for no temperature
  real m_temperature;
  // The six quads representing the sides of the box
  std::vector<VoxelQuad> m_quads;

  VoxelBox(std::string name, vec3<int> voxMin, vec3<int> voxMax, vec3<real> min,
           vec3<real> max, real temperature = NaN);
};

class VoxelArea : public VoxelObject {
 public:
  // World coordinates min/max (in m)
  vec3<real> m_min;
  vec3<real> m_max;
  // Coordinates in lattice units
  vec3<int> m_voxMin;
  vec3<int> m_voxMax;

  VoxelArea(std::string name, vec3<int> voxMin, vec3<int> voxMax,
            vec3<real> min, vec3<real> max)
      : VoxelObject(name),
        m_voxMin(voxMin),
        m_voxMax(voxMax),
        m_min(min),
        m_max(max) {}

  glm::ivec3 getMin() { return glm::ivec3(m_voxMin.x, m_voxMin.y, m_voxMin.z); }
  glm::ivec3 getMax() { return glm::ivec3(m_voxMax.x, m_voxMax.y, m_voxMax.z); }
  glm::ivec3 getDims() {
    return glm::ivec3(m_voxMax.x - m_voxMin.x, m_voxMax.y - m_voxMin.y,
                      m_voxMax.z - m_voxMin.z);
  }
  int getNumVoxels() {
    glm::ivec3 n = getDims();
    return n.x * n.y * n.z;
  }
};

namespace std {
template <>
struct hash<VoxelArea> {
  std::size_t operator()(const VoxelArea &area) const {
    using std::hash;
    using std::size_t;
    size_t seed = 0;
    ::hash_combine(seed, area.m_min.x);
    ::hash_combine(seed, area.m_min.y);
    ::hash_combine(seed, area.m_min.z);
    ::hash_combine(seed, area.m_max.x);
    ::hash_combine(seed, area.m_max.y);
    ::hash_combine(seed, area.m_max.z);
    ::hash_combine(seed, area.m_voxMin.x);
    ::hash_combine(seed, area.m_voxMin.y);
    ::hash_combine(seed, area.m_voxMin.z);
    ::hash_combine(seed, area.m_voxMax.x);
    ::hash_combine(seed, area.m_voxMax.y);
    ::hash_combine(seed, area.m_voxMax.z);
    ::hash_combine(seed, area.m_name);
    return seed;
  }
};
}  // namespace std

bool operator==(VoxelQuad const &a, VoxelQuad const &b);
bool operator==(VoxelArea const &a, VoxelArea const &b);
