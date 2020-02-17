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

  VoxelObject& operator=(const VoxelObject& other) {
    m_name = other.m_name;
    return *this;
  }
};

// A plane of voxels
class VoxelQuad : public VoxelObject {
 public:
  // World coordinates (in m)
  Eigen::Vector3f m_origin;
  Eigen::Vector3f m_dir1;
  Eigen::Vector3f m_dir2;
  // Discretized coordinates and extents in lattice units
  Eigen::Vector3i m_voxOrigin;
  Eigen::Vector3i m_voxDir1;
  Eigen::Vector3i m_voxDir2;
  // Mode (fill, overwrite etc.)
  NodeMode::Enum m_mode;
  // Common boundary condition for voxels in this quad
  BoundaryCondition m_bc;
  // Intersections with other boundary conditions (can be empty)
  std::unordered_set<BoundaryCondition, std::hash<BoundaryCondition>>
      m_intersectingBcs;

  inline int getNumVoxels() const {
    real d1 = m_voxDir1.x() * m_voxDir1.x() + m_voxDir1.y() * m_voxDir1.y() +
              m_voxDir1.z() * m_voxDir1.z();
    real d2 = m_voxDir2.x() * m_voxDir2.x() + m_voxDir2.y() * m_voxDir2.y() +
              m_voxDir2.z() * m_voxDir2.z();
    return static_cast<int>(sqrt(d1) * sqrt(d2));
  }

  inline real getAreaReal() const {
    real d1 = sqrt(m_dir1.x() * m_dir1.x() + m_dir1.y() * m_dir1.y() +
                   m_dir1.z() * m_dir1.z());
    real d2 = sqrt(m_dir2.x() * m_dir2.x() + m_dir2.y() * m_dir2.y() +
                   m_dir2.z() * m_dir2.z());
    return d1 * d2;
  }

  inline real getAreaDiscrete(const UnitConverter& uc) const {
    return getNumVoxels() * uc.C_L() * uc.C_L();
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

  VoxelQuad(std::string name,
            NodeMode::Enum mode,
            Eigen::Vector3i voxOrigin,
            Eigen::Vector3i voxDir1,
            Eigen::Vector3i voxDir2,
            Eigen::Vector3i normal,
            VoxelType::Enum type = VoxelType::Enum::WALL,
            real temperature = NaN,
            real tau1 = 0,
            real tau2 = 0,
            real lambda = 0,
            Eigen::Vector3f velocity = Eigen::Vector3f(NaN, NaN, NaN),
            Eigen::Vector3i rel_pos = Eigen::Vector3i(0, 0, 0),
            Eigen::Vector3f origin = Eigen::Vector3f(NaN, NaN, NaN),
            Eigen::Vector3f dir1 = Eigen::Vector3f(NaN, NaN, NaN),
            Eigen::Vector3f dir2 = Eigen::Vector3f(NaN, NaN, NaN))
      : VoxelObject(name),
        m_bc(BoundaryCondition(
            -1,
            type,
            temperature,
            Eigen::Vector3f(velocity.x(), velocity.y(), velocity.z()),
            Eigen::Vector3i(normal.x(), normal.y(), normal.z()),
            Eigen::Vector3i(rel_pos.x(), rel_pos.y(), rel_pos.z()),
            tau1,
            tau2,
            lambda)),
        m_origin(origin),
        m_dir1(dir1),
        m_dir2(dir2),
        m_mode(mode),
        m_voxOrigin(voxOrigin),
        m_voxDir1(voxDir1),
        m_voxDir2(voxDir2) {}
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
  Eigen::Vector3f m_min;
  Eigen::Vector3f m_max;
  // Coordinates in lattice units
  Eigen::Vector3i m_voxMin;
  Eigen::Vector3i m_voxMax;

  VoxelVolume()
      : VoxelObject(),
        m_min(-1, -1, -1),
        m_max(-1, -1, -1),
        m_voxMin(-1, -1, -1),
        m_voxMax(-1, -1, -1) {}

  VoxelVolume(std::string name,
              Eigen::Vector3i voxMin,
              Eigen::Vector3i voxMax,
              Eigen::Vector3f min = Eigen::Vector3f(-1, -1, -1),
              Eigen::Vector3f max = Eigen::Vector3f(-1, -1, -1))
      : VoxelObject(name),
        m_voxMin(voxMin),
        m_voxMax(voxMax),
        m_min(min),
        m_max(max) {
    assert((voxMin.x() < voxMax.x() && voxMin.y() < voxMax.y() &&
            voxMin.z() < voxMax.z()));
  }

  inline Eigen::Vector3i getMin() const {
    return Eigen::Vector3i(m_voxMin.x(), m_voxMin.y(), m_voxMin.z());
  }

  inline Eigen::Vector3i getMax() const {
    return Eigen::Vector3i(m_voxMax.x(), m_voxMax.y(), m_voxMax.z());
  }

  inline Eigen::Vector3i getExtents() const {
    return Eigen::Vector3i(max(m_voxMax.x() - m_voxMin.x(), 1),
                           max(m_voxMax.y() - m_voxMin.y(), 1),
                           max(m_voxMax.z() - m_voxMin.z(), 1));
  }

  inline size_t getNumVoxels() const {
    Eigen::Vector3i n = getExtents();
    return n.x() * n.y() * n.z();
  }

  inline int getRank() const {
    Eigen::Vector3i n = Eigen::Vector3i(m_voxMax.x() - m_voxMin.x(),
                                        m_voxMax.y() - m_voxMin.y(),
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
  real m_temperature;
  // The six quads representing the sides of the box
  std::vector<VoxelQuad> m_quads;

  VoxelBox(std::string name,
           Eigen::Vector3i voxMin,
           Eigen::Vector3i voxMax,
           Eigen::Vector3f min,
           Eigen::Vector3f max,
           real temperature = NaN);
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
