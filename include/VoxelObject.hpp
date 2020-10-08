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

#include "BoundaryCondition.hpp"
#include "StdUtils.hpp"
#include "Vector3.hpp"

namespace NodeMode {
enum Enum { OVERWRITE, INTERSECT, FILL };
}

// A base class only holding a name string
class VoxelObject {
 public:
  std::string m_name;
  std::string getName() const { return m_name; }

  VoxelObject() : m_name("NULL") {}
  explicit VoxelObject(std::string name) : m_name(name) {}
  VoxelObject(const VoxelObject& other) : m_name(other.m_name) {}

  VoxelObject& operator=(const VoxelObject& other) {
    m_name = other.m_name;
    return *this;
  }
};

// A plane of voxels
class VoxelQuad : public VoxelObject {
 public:
  // World coordinates (in m)
  vector3<real_t> m_origin;
  vector3<real_t> m_dir1;
  vector3<real_t> m_dir2;
  // Discretized coordinates and extents in lattice units
  vector3<int> m_voxOrigin;
  vector3<int> m_voxDir1;
  vector3<int> m_voxDir2;
  // Mode (fill, overwrite etc.)
  NodeMode::Enum m_mode;
  // Common boundary condition for voxels in this quad
  BoundaryCondition m_bc;
  // Intersections with other boundary conditions (can be empty)
  std::unordered_set<BoundaryCondition, std::hash<BoundaryCondition>>
      m_intersectingBcs;

  inline int getNumVoxels() const {
    real_t d1 = m_voxDir1.x() * m_voxDir1.x() + m_voxDir1.y() * m_voxDir1.y() +
              m_voxDir1.z() * m_voxDir1.z();
    real_t d2 = m_voxDir2.x() * m_voxDir2.x() + m_voxDir2.y() * m_voxDir2.y() +
              m_voxDir2.z() * m_voxDir2.z();
    return static_cast<int>(sqrt(d1) * sqrt(d2));
  }

  inline real_t getAreaReal() const {
    real_t d1 = sqrt(m_dir1.x() * m_dir1.x() + m_dir1.y() * m_dir1.y() +
                   m_dir1.z() * m_dir1.z());
    real_t d2 = sqrt(m_dir2.x() * m_dir2.x() + m_dir2.y() * m_dir2.y() +
                   m_dir2.z() * m_dir2.z());
    return d1 * d2;
  }

  inline real_t getAreaDiscrete(const UnitConverter& uc) const {
    return getNumVoxels() * uc.C_L() * uc.C_L();
  }

  VoxelQuad()
      : VoxelObject(std::string()),
        m_origin(NaN, NaN, NaN),
        m_dir1(NaN, NaN, NaN),
        m_dir2(NaN, NaN, NaN),
        m_voxOrigin(0, 0, 0),
        m_voxDir1(0, 0, 0),
        m_voxDir2(0, 0, 0),
        m_mode(NodeMode::Enum::FILL),
        m_bc(BoundaryCondition()),
        m_intersectingBcs() {}

  VoxelQuad(std::string name, NodeMode::Enum mode, vector3<int> voxOrigin,
            vector3<int> voxDir1, vector3<int> voxDir2, vector3<int> normal,
            VoxelType::Enum type = VoxelType::Enum::WALL,
            real_t temperature = NaN, real_t tau1 = 0, real_t tau2 = 0,
            real_t lambda = 0,
            vector3<real_t> velocity = vector3<real_t>(NaN, NaN, NaN),
            vector3<int> rel_pos = vector3<int>(0, 0, 0),
            vector3<real_t> origin = vector3<real_t>(NaN, NaN, NaN),
            vector3<real_t> dir1 = vector3<real_t>(NaN, NaN, NaN),
            vector3<real_t> dir2 = vector3<real_t>(NaN, NaN, NaN))
      : VoxelObject(name),
        m_origin(origin),
        m_dir1(dir1),
        m_dir2(dir2),
        m_voxOrigin(voxOrigin),
        m_voxDir1(voxDir1),
        m_voxDir2(voxDir2),
        m_mode(mode),
        m_bc(BoundaryCondition(
            VoxelType::Enum::EMPTY, type, temperature,
            vector3<real_t>(velocity.x(), velocity.y(), velocity.z()),
            vector3<int>(normal.x(), normal.y(), normal.z()),
            vector3<int>(rel_pos.x(), rel_pos.y(), rel_pos.z()), tau1, tau2,
            lambda)),
        m_intersectingBcs() {}
};

inline std::ostream& operator<<(std::ostream& os, const VoxelQuad& v) {
  os << "name: " << v.m_name << std::endl;
  os << "mode: " << v.m_mode << std::endl;
  os << "origin (m): (" << v.m_origin.x() << ", " << v.m_origin.y() << ", "
     << v.m_origin.z() << ")" << std::endl;
  os << "dir1 (m): (" << v.m_dir1.x() << ", " << v.m_dir1.y() << ", "
     << v.m_dir1.z() << ")" << std::endl;
  os << "dir2 (m): (" << v.m_dir2.x() << ", " << v.m_dir2.y() << ", "
     << v.m_dir2.z() << ")" << std::endl;
  os << "origin (vox): (" << v.m_voxOrigin.x() << ", " << v.m_voxOrigin.y()
     << ", " << v.m_voxOrigin.z() << ")" << std::endl;
  os << "dir1 (vox): (" << v.m_voxDir1.x() << ", " << v.m_voxDir1.y() << ", "
     << v.m_voxDir1.z() << ")" << std::endl;
  os << "dir2 (vox): (" << v.m_voxDir2.x() << ", " << v.m_voxDir2.y() << ", "
     << v.m_voxDir2.z() << ")" << std::endl;
  return os;
}

namespace std {
template <>
struct hash<VoxelQuad> {
  std::size_t operator()(const VoxelQuad& quad) const {
    using std::hash;
    using std::size_t;
    size_t seed = 0;
    ::hash_combine(&seed, quad.m_origin.x(), quad.m_origin.y(),
                   quad.m_origin.z(), quad.m_dir1.x(), quad.m_dir1.y(),
                   quad.m_dir1.z(), quad.m_dir2.x(), quad.m_dir2.y(),
                   quad.m_dir2.z(), quad.m_mode, quad.m_name);
    return seed;
  }
};
}  // namespace std

class VoxelVolume : public VoxelObject {
 public:
  // World coordinates min/max (in m)
  vector3<real_t> m_min;
  vector3<real_t> m_max;
  // Coordinates in lattice units
  vector3<int> m_voxMin;
  vector3<int> m_voxMax;

  VoxelVolume()
      : VoxelObject(),
        m_min(-1, -1, -1),
        m_max(-1, -1, -1),
        m_voxMin(-1, -1, -1),
        m_voxMax(-1, -1, -1) {}

  VoxelVolume(const VoxelVolume& other)
      : VoxelObject(other.m_name),
        m_min(other.m_min),
        m_max(other.m_max),
        m_voxMin(other.m_voxMin),
        m_voxMax(other.m_voxMax) {}

  VoxelVolume(std::string name, vector3<int> voxMin, vector3<int> voxMax,
              vector3<real_t> min = vector3<real_t>(-1, -1, -1),
              vector3<real_t> max = vector3<real_t>(-1, -1, -1))
      : VoxelObject(name),
        m_min(min),
        m_max(max),
        m_voxMin(voxMin),
        m_voxMax(voxMax) {
    assert((voxMin.x() < voxMax.x() && voxMin.y() < voxMax.y() &&
            voxMin.z() < voxMax.z()));
  }

  inline vector3<int> getMin() const {
    return vector3<int>(m_voxMin.x(), m_voxMin.y(), m_voxMin.z());
  }

  inline vector3<int> getMax() const {
    return vector3<int>(m_voxMax.x(), m_voxMax.y(), m_voxMax.z());
  }

  inline vector3<int> getExtents() const {
    return vector3<int>(max(m_voxMax.x() - m_voxMin.x(), 1),
                        max(m_voxMax.y() - m_voxMin.y(), 1),
                        max(m_voxMax.z() - m_voxMin.z(), 1));
  }

  inline size_t getNumVoxels() const {
    vector3<int> n = getExtents();
    return n.x() * n.y() * n.z();
  }

  inline int getRank() const {
    vector3<int> n =
        vector3<int>(m_voxMax.x() - m_voxMin.x(), m_voxMax.y() - m_voxMin.y(),
                     m_voxMax.z() - m_voxMin.z());
    int rank = 0;
    rank += n.x() > 1 ? 1 : 0;
    rank += n.y() > 1 ? 1 : 0;
    rank += n.z() > 1 ? 1 : 0;
    rank = rank == 0 ? 1 : rank;
    return rank;
  }

  VoxelVolume& operator=(const VoxelVolume& other) {
    VoxelObject::operator=(other);
    m_min = other.m_min;
    m_max = other.m_max;
    m_voxMin = other.m_voxMin;
    m_voxMax = other.m_voxMax;
    return *this;
  }
};

// A box of voxels
class VoxelBox : public VoxelVolume {
 public:
  // NaN for no temperature
  real_t m_temperature;
  // The six quads representing the sides of the box
  std::vector<VoxelQuad> m_quads;

  VoxelBox(std::string name, vector3<int> voxMin, vector3<int> voxMax,
           vector3<real_t> min, vector3<real_t> max, real_t temperature = NaN);
};

typedef std::vector<VoxelVolume> VoxelVolumeArray;

namespace std {
template <>
struct hash<VoxelVolume> {
  std::size_t operator()(const VoxelVolume& area) const {
    using std::hash;
    using std::size_t;
    size_t seed = 0;
    ::hash_combine(&seed, area.m_min.x(), area.m_min.y(), area.m_min.z(),
                   area.m_max.x(), area.m_max.y(), area.m_max.z(),
                   area.m_voxMin.x(), area.m_voxMin.y(), area.m_voxMin.z(),
                   area.m_voxMax.x(), area.m_voxMax.y(), area.m_voxMax.z(),
                   area.m_name);
    return seed;
  }
};
}  // namespace std

bool operator==(VoxelQuad const& a, VoxelQuad const& b);
bool operator==(VoxelVolume const& a, VoxelVolume const& b);
