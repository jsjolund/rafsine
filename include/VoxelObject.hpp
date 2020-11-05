#pragma once

#include <algorithm>
#include <cmath>
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

/**
 * @brief  Voxel object base class only holding a name string
 */
class VoxelObject {
 public:
  //! Name of object
  std::string m_name;
  /**
   * @return std::string Name of object
   */
  std::string getName() const { return m_name; }
  /**
   * @brief Construct a new empty voxel object
   */
  VoxelObject() : m_name("NULL") {}
  /**
   * @brief Construct a new named voxel object
   * @param name
   */
  explicit VoxelObject(std::string name) : m_name(name) {}
  /**
   * @brief Copy constructor
   * @param other
   */
  VoxelObject(const VoxelObject& other) : m_name(other.m_name) {}

  VoxelObject& operator=(const VoxelObject& other) {
    m_name = other.m_name;
    return *this;
  }
};

/**
 * @brief Quad plane of voxels
 */
class VoxelQuad : public VoxelObject {
 public:
  //! World coordinate origin (in m)
  Vector3<real_t> m_origin;
  //! Real direction 1
  Vector3<real_t> m_dir1;
  //! Real direction 2
  Vector3<real_t> m_dir2;
  //! Discretized coordinate origin in lattice units
  Vector3<int> m_voxOrigin;
  //! Discretized direction 1
  Vector3<int> m_voxDir1;
  //! Discretized direction 2
  Vector3<int> m_voxDir2;
  //! Mode (fill, overwrite etc.)
  NodeMode::Enum m_mode;
  //! Common boundary condition for voxels in this quad
  BoundaryCondition m_bc;
  //! Intersections with other boundary conditions (can be empty)
  std::unordered_set<BoundaryCondition, std::hash<BoundaryCondition>>
      m_intersectingBcs;

  /**
   * @return int Total number of voxels
   */
  inline size_t getNumVoxels() const {
    real_t d1 = m_voxDir1.x() * m_voxDir1.x() + m_voxDir1.y() * m_voxDir1.y() +
                m_voxDir1.z() * m_voxDir1.z();
    real_t d2 = m_voxDir2.x() * m_voxDir2.x() + m_voxDir2.y() * m_voxDir2.y() +
                m_voxDir2.z() * m_voxDir2.z();
    return static_cast<int>(sqrt(d1) * sqrt(d2));
  }
  /**
   * @return real_t Real area of quad (in m^2)
   */
  inline real_t getAreaReal() const {
    real_t d1 = sqrt(m_dir1.x() * m_dir1.x() + m_dir1.y() * m_dir1.y() +
                     m_dir1.z() * m_dir1.z());
    real_t d2 = sqrt(m_dir2.x() * m_dir2.x() + m_dir2.y() * m_dir2.y() +
                     m_dir2.z() * m_dir2.z());
    return d1 * d2;
  }

  /**
   * @brief Get the discretized area of quad (in m^2)
   *
   * @param uc
   * @return real_t
   */
  inline real_t getAreaDiscrete(const UnitConverter& uc) const {
    return getNumVoxels() * uc.C_L() * uc.C_L();
  }

  /**
   * @brief Construct zero size quad
   */
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

  /**
   * @brief Construct a new voxel quad plane
   *
   * @param name Name of geometry
   * @param mode Type of operation (e.g. overwrite, fill)
   * @param voxOrigin Discretized origin coordinates
   * @param voxDir1 Discretized direction 1
   * @param voxDir2 Discretized direction 2
   * @param normal Plane normal
   * @param type Type of boundary condition (e.g. wall, relative condition)
   * @param temperature Temperature to emit
   * @param tau1 Temperature change time constant 1
   * @param tau2 Temperature change time constant 2
   * @param lambda Temperature change time constant 3
   * @param velocity Velocity of output flow if any
   * @param rel_pos Position of relative boundary condition
   * @param origin Real origin
   * @param dir1 Real direction 1
   * @param dir2 Real direction 2
   */
  VoxelQuad(std::string name,
            NodeMode::Enum mode,
            Vector3<int> voxOrigin,
            Vector3<int> voxDir1,
            Vector3<int> voxDir2,
            Vector3<int> normal,
            VoxelType::Enum type = VoxelType::Enum::WALL,
            real_t temperature = NaN,
            real_t tau1 = 0,
            real_t tau2 = 0,
            real_t lambda = 0,
            Vector3<real_t> velocity = Vector3<real_t>(NaN, NaN, NaN),
            Vector3<int> rel_pos = Vector3<int>(0, 0, 0),
            Vector3<real_t> origin = Vector3<real_t>(NaN, NaN, NaN),
            Vector3<real_t> dir1 = Vector3<real_t>(NaN, NaN, NaN),
            Vector3<real_t> dir2 = Vector3<real_t>(NaN, NaN, NaN))
      : VoxelObject(name),
        m_origin(origin),
        m_dir1(dir1),
        m_dir2(dir2),
        m_voxOrigin(voxOrigin),
        m_voxDir1(voxDir1),
        m_voxDir2(voxDir2),
        m_mode(mode),
        m_bc(BoundaryCondition(
            VoxelType::Enum::EMPTY,
            type,
            temperature,
            Vector3<real_t>(velocity.x(), velocity.y(), velocity.z()),
            Vector3<int>(normal.x(), normal.y(), normal.z()),
            Vector3<int>(rel_pos.x(), rel_pos.y(), rel_pos.z()),
            tau1,
            tau2,
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

/**
 * @brief Cuboid box of voxels
 */
class VoxelCuboid : public VoxelObject {
 public:
  //! World coordinates min (in m)
  Vector3<real_t> m_min;
  //! World coordinates max (in m)
  Vector3<real_t> m_max;
  //! Min coordinates in lattice units
  Vector3<int> m_voxMin;
  //! Max coordinates in lattice units
  Vector3<int> m_voxMax;
  /**
   * @brief Construct an empty cuboid
   */
  VoxelCuboid()
      : VoxelObject(),
        m_min(-1, -1, -1),
        m_max(-1, -1, -1),
        m_voxMin(-1, -1, -1),
        m_voxMax(-1, -1, -1) {}
  /**
   * @brief Cuboid copy constructor
   * @param other
   */
  VoxelCuboid(const VoxelCuboid& other)
      : VoxelObject(other.m_name),
        m_min(other.m_min),
        m_max(other.m_max),
        m_voxMin(other.m_voxMin),
        m_voxMax(other.m_voxMax) {}
  /**
   * @brief Construct a new voxel cuboid
   *
   * @param name Geometry name
   * @param voxMin Minimum position on lattice
   * @param voxMax Maximum position on lattice
   * @param min Minimum position in real units (i.e. meters)
   * @param max Maximum position in real units (i.e. meters)
   */
  VoxelCuboid(std::string name,
              Vector3<int> voxMin,
              Vector3<int> voxMax,
              Vector3<real_t> min = Vector3<real_t>(-1, -1, -1),
              Vector3<real_t> max = Vector3<real_t>(-1, -1, -1))
      : VoxelObject(name),
        m_min(min),
        m_max(max),
        m_voxMin(voxMin),
        m_voxMax(voxMax) {
    assert((voxMin.x() < voxMax.x() && voxMin.y() < voxMax.y() &&
            voxMin.z() < voxMax.z()));
  }
  /**
   * @return Vector3<int> Minimum lattice position
   */
  inline Vector3<int> getMin() const {
    return Vector3<int>(m_voxMin.x(), m_voxMin.y(), m_voxMin.z());
  }
  /**
   * @return Vector3<int> Maximum lattice position
   */
  inline Vector3<int> getMax() const {
    return Vector3<int>(m_voxMax.x(), m_voxMax.y(), m_voxMax.z());
  }
  /**
   * @return Vector3<int> 3D extents
   */
  inline Vector3<int> getExtents() const {
    return Vector3<int>(max(m_voxMax.x() - m_voxMin.x(), 1),
                        max(m_voxMax.y() - m_voxMin.y(), 1),
                        max(m_voxMax.z() - m_voxMin.z(), 1));
  }
  /**
   * @return size_t Total number of voxels
   */
  inline size_t getNumVoxels() const {
    Vector3<int> n = getExtents();
    return n.x() * n.y() * n.z();
  }

  /**
   * @return int Returns 1 if shape is point or line, 2 if plane, 3 if cuboid
   */
  inline int getRank() const {
    Vector3<int> n =
        Vector3<int>(m_voxMax.x() - m_voxMin.x(), m_voxMax.y() - m_voxMin.y(),
                     m_voxMax.z() - m_voxMin.z());
    int rank = 0;
    rank += n.x() > 1 ? 1 : 0;
    rank += n.y() > 1 ? 1 : 0;
    rank += n.z() > 1 ? 1 : 0;
    rank = rank == 0 ? 1 : rank;
    return rank;
  }

  VoxelCuboid& operator=(const VoxelCuboid& other) {
    VoxelObject::operator=(other);
    m_min = other.m_min;
    m_max = other.m_max;
    m_voxMin = other.m_voxMin;
    m_voxMax = other.m_voxMax;
    return *this;
  }
};

/**
 * @brief  A box of voxels holding temperatures and side quads
 */
class VoxelBox : public VoxelCuboid {
 public:
  //! Temperature (NaN for no temperature)
  real_t m_temperature;
  //! The six quads representing the sides of the box
  std::vector<VoxelQuad> m_quads;
  /**
   * @brief Construct a new voxel box
   *
   * @param name Geometry name
   * @param voxMin Minimum position on lattice
   * @param voxMax Maximum position on lattice
   * @param min Minimum position in real units (i.e. meters)
   * @param max Maximum position in real units (i.e. meters)
   * @param temperature Optional temperature to emit
   */
  VoxelBox(std::string name,
           Vector3<int> voxMin,
           Vector3<int> voxMax,
           Vector3<real_t> min,
           Vector3<real_t> max,
           real_t temperature = NaN);
};

namespace SphereVoxel {
enum Enum { INSIDE, SURFACE, CORNER, OUTSIDE };
}

/**
 * @brief Sphere of voxels
 */
class VoxelSphere : public VoxelObject {
 private:
  const unsigned int m_n;
  std::vector<SphereVoxel::Enum> m_grid;
  std::vector<Vector3<int>> m_normals;

  unsigned int idx(int x, int y, int z);
  unsigned int idxn(int x, int y, int z);
  void fill(const int x, const int y, const int z);
  void fillInside(const int x, const int y, const int z);
  void fillSigns(int x, int y, int z);
  void fillAll(int x, int y, int z);
  void createSphere(float R);

 public:
  /**
   * @brief Get type of SphereVoxel::Enum at position
   *
   * @param x
   * @param y
   * @param z
   * @return SphereVoxel::Enum
   */
  SphereVoxel::Enum getVoxel(unsigned int x, unsigned int y, unsigned int z);
  Vector3<int> getNormal(unsigned int x, unsigned int y, unsigned int z);

  //! Origin world coordinates (in m)
  Vector3<real_t> m_origin;
  //! Origin coordinates in lattice units
  Vector3<int> m_voxOrigin;
  //! Radius in m
  real_t m_radius;
  //! Radius in lattice units
  unsigned int m_voxRadius;
  //! NaN for no temperature
  real_t m_temperature;
  /**
   * @return unsigned int Size along X-axis
   */
  unsigned int getSizeX() { return m_n; }
  /**
   * @return unsigned int Size along Y-axis
   */
  unsigned int getSizeY() { return m_n; }
  /**
   * @return unsigned int Size along Z-axis
   */
  unsigned int getSizeZ() { return m_n; }
  /**
   * @return unsigned int Radius in voxels
   */
  unsigned int getRadius() { return m_voxRadius; }

  /**
   * @brief Construct a new voxel sphere
   *
   * @param name Name of geometry
   * @param voxOrigin Origin in lattice units
   * @param origin Origin in real units (e.g. m)
   * @param radius Radius in real units
   * @param temperature Optional temperature to emit
   */
  VoxelSphere(std::string name,
              Vector3<int> voxOrigin,
              Vector3<real_t> origin,
              real_t radius,
              real_t temperature);
};

typedef std::vector<VoxelCuboid> VoxelCuboidArray;

namespace std {
template <>
struct hash<VoxelCuboid> {
  std::size_t operator()(const VoxelCuboid& area) const {
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
bool operator==(VoxelCuboid const& a, VoxelCuboid const& b);
