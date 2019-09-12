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
  std::string getName() { return m_name; }
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

inline std::ostream &operator<<(std::ostream &os, const VoxelQuad &v) {
  os << "name: " << v.m_name << std::endl;
  os << "mode: " << v.m_mode << std::endl;
  os << "origin (m): (" << v.m_origin.x << ", " << v.m_origin.y << ", "
     << v.m_origin.z << ")" << std::endl;
  os << "dir1 (m): (" << v.m_dir1.x << ", " << v.m_dir1.y << ", " << v.m_dir1.z
     << ")" << std::endl;
  os << "dir2 (m): (" << v.m_dir2.x << ", " << v.m_dir2.y << ", " << v.m_dir2.z
     << ")" << std::endl;
  os << "origin (vox): (" << v.m_voxOrigin.x << ", " << v.m_voxOrigin.y << ", "
     << v.m_voxOrigin.z << ")" << std::endl;
  os << "dir1 (vox): (" << v.m_voxDir1.x << ", " << v.m_voxDir1.y << ", "
     << v.m_voxDir1.z << ")" << std::endl;
  os << "dir2 (vox): (" << v.m_voxDir2.x << ", " << v.m_voxDir2.y << ", "
     << v.m_voxDir2.z << ")" << std::endl;
  return os;
}

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

class VoxelVolume : public VoxelObject {
 public:
  // World coordinates min/max (in m)
  vec3<real> m_min;
  vec3<real> m_max;
  // Coordinates in lattice units
  vec3<int> m_voxMin;
  vec3<int> m_voxMax;

  VoxelVolume(std::string name, vec3<int> voxMin, vec3<int> voxMax,
              vec3<real> min = vec3<real>(-1, -1, -1),
              vec3<real> max = vec3<real>(-1, -1, -1))
      : VoxelObject(name),
        m_voxMin(voxMin),
        m_voxMax(voxMax),
        m_min(min),
        m_max(max) {
    assert((voxMin.x < voxMax.x && voxMin.y < voxMax.y && voxMin.z < voxMax.z));
  }

  inline glm::ivec3 getMin() {
    return glm::ivec3(m_voxMin.x, m_voxMin.y, m_voxMin.z);
  }

  inline glm::ivec3 getMax() {
    return glm::ivec3(m_voxMax.x, m_voxMax.y, m_voxMax.z);
  }

  inline glm::ivec3 getDims() {
    return glm::ivec3(max(m_voxMax.x - m_voxMin.x, 1),
                      max(m_voxMax.y - m_voxMin.y, 1),
                      max(m_voxMax.z - m_voxMin.z, 1));
  }

  inline size_t getNumVoxels() {
    glm::ivec3 n = getDims();
    return n.x * n.y * n.z;
  }

  inline int getRank() {
    glm::ivec3 n = getDims();
    int rank = 0;
    rank += n.x > 1 ? 1 : 0;
    rank += n.y > 1 ? 1 : 0;
    rank += n.z > 1 ? 1 : 0;
    rank = rank == 0 ? 1 : rank;
    return rank;
  }
};

// A box of voxels
class VoxelBox : public VoxelVolume {
 public:
  // NaN for no temperature
  real m_temperature;
  // The six quads representing the sides of the box
  std::vector<VoxelQuad> m_quads;

  VoxelBox(std::string name, vec3<int> voxMin, vec3<int> voxMax, vec3<real> min,
           vec3<real> max, real temperature = NaN);
};

typedef std::vector<VoxelVolume> VoxelVolumeArray;

namespace std {
template <>
struct hash<VoxelVolume> {
  std::size_t operator()(const VoxelVolume &area) const {
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
bool operator==(VoxelVolume const &a, VoxelVolume const &b);
