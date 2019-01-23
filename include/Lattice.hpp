#pragma once

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "Primitives.hpp"

namespace std {
template <>
struct hash<glm::ivec3> {
  std::size_t operator()(const glm::ivec3 &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, p.x);
    ::hash_combine(seed, p.y);
    ::hash_combine(seed, p.z);
    return seed;
  }
};
}  // namespace std

class SubLatticeSegment {
 public:
  glm::ivec3 m_src;
  glm::ivec3 m_dst;
  size_t m_srcStride;
  size_t m_dstStride;
  size_t m_segmentLength;
  size_t m_numSegments;

  inline SubLatticeSegment()
      : m_src(glm::ivec3(0, 0, 0)),
        m_dst(glm::ivec3(0, 0, 0)),
        m_srcStride(0),
        m_dstStride(0),
        m_segmentLength(0),
        m_numSegments(0) {}
};

class SubLattice {
 private:
  glm::ivec3 m_min, m_max;

 public:
  /**
   * @brief 3D axis enum
   *
   */
  enum Enum { X_AXIS, Y_AXIS, Z_AXIS };

  /**
   * @brief Construct a new SubLattice object
   *
   * @param min Minimum point of subLattice on the lattice
   * @param max Maximum point of subLattice on the lattice
   */
  inline SubLattice(glm::ivec3 min, glm::ivec3 max) : m_min(min), m_max(max) {}
  /**
   * @brief Construct a new empty SubLattice
   *
   */
  inline SubLattice() {}
  /**
   * @brief Copy constructor
   * @param other Another subLattice
   */
  inline SubLattice(const SubLattice &other)
      : m_min(other.m_min), m_max(other.m_max) {}
  inline ~SubLattice() {}
  /**
   * @brief Get the minimum point of subLattice on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getLatticeMin() const { return m_min; }
  /**
   * @brief Get the maximum point of subLattice on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getLatticeMax() const { return m_max; }
  /**
   * @brief Get the 3D sizes of the subLattice on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getLatticeDims() const { return m_max - m_min; }
  /**
   * @brief Get the total size of the subLattice on the lattice
   *
   * @return size_t
   */
  inline size_t getLatticeSize() const {
    glm::ivec3 dims = getLatticeDims();
    return dims.x * dims.y * dims.z;
  }
  /**
   * @brief Get the 3D array dimensions of the first order q of the distribution
   * function (including halos)
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getArrayDims() const {
    glm::ivec3 dims = getLatticeDims();
    return dims + glm::ivec3(2, 2, 2);
  }
  /**
   * @brief Get the array size of the first order q of the distribution
   * function (including halos), or in other words, the array stride between
   * different q > 1
   *
   * @return glm::ivec3
   */
  inline size_t getArrayStride() const {
    glm::ivec3 dims = getArrayDims();
    return dims.x * dims.y * dims.z;
  }

  /**
   * @brief Calculate index in subLattice array from global coordinates, such
   * that the position p >= min-1 && p < max+1.
   *
   * @param df_idx The distribution function index
   * @param x
   * @param y
   * @param z
   * @return int
   */
  int toLocalIndex(unsigned int df_idx, int x, int y, int z = 0);

  /**
   * @brief Finds the axis with the least slice area when cut
   *
   * @return SubLattice::Enum The axis
   */
  SubLattice::Enum getDivisionAxis();

  void getHaloPlane(glm::ivec3 direction, glm::ivec3 *src, size_t *srcStride,
                    glm::ivec3 srcDim, glm::ivec3 *dst, size_t *dstStride,
                    glm::ivec3 dstDim, size_t *width, size_t *height);

  SubLatticeSegment getSubLatticeSegment(glm::ivec3 direction,
                                         SubLattice neighbour);
};
bool operator==(SubLattice const &a, SubLattice const &b);
std::ostream &operator<<(std::ostream &os, SubLattice p);

namespace std {
template <>
struct hash<SubLattice> {
  std::size_t operator()(const SubLattice &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, p.getLatticeMin().x);
    ::hash_combine(seed, p.getLatticeMin().y);
    ::hash_combine(seed, p.getLatticeMin().z);
    ::hash_combine(seed, p.getLatticeMax().x);
    ::hash_combine(seed, p.getLatticeMax().y);
    ::hash_combine(seed, p.getLatticeMax().z);
    return seed;
  }
};
}  // namespace std

class Lattice {
 protected:
  std::vector<SubLattice> m_subLattices;

  glm::ivec3 m_latticeSize;
  glm::ivec3 m_subLatticeCount;
  // Number of arrays (or directions for distribution functions)
  const unsigned int m_Q;

  std::unordered_map<SubLattice, glm::ivec3> m_subLatticePositions;

 public:
  std::unordered_map<
      SubLattice,
      std::unordered_map<SubLattice, std::vector<SubLatticeSegment>>>
      m_segments;

  SubLattice getNeighbour(SubLattice subLattice, glm::ivec3 direction);
  inline SubLattice getNeighbour(SubLattice subLattice,
                                 UnitVector::Enum direction) {
    return getNeighbour(subLattice, D3Q27[direction]);
  }
  inline std::vector<SubLattice> getSubLattices() { return m_subLattices; }
  inline glm::ivec3 getLatticeDims() const { return glm::ivec3(m_latticeSize); }
  inline size_t getLatticeSize() const {
    return m_latticeSize.x * m_latticeSize.y * m_latticeSize.z;
  }
  inline glm::ivec3 getNumSubLattices() {
    return glm::ivec3(m_subLatticeCount);
  }
  inline int getNumSubLatticesTotal() { return m_subLattices.size(); }

  /**
   * @brief Return the number of arrays in the group i.e. the number of
   * distribution functions
   *
   * @return unsigned int
   */
  unsigned int getQ() const { return m_Q; }

  Lattice(unsigned int Q, unsigned int latticeSizeX, unsigned int latticeSizeY,
          unsigned int latticeSizeZ, unsigned int subdivisions = 0);

  inline ~Lattice() {
    // for (SubLattice p : m_subLattices) delete p;
  }

  SubLattice getSubLatticeContaining(unsigned int x, unsigned int y,
                                     unsigned int z);

  inline SubLattice getSubLattice(int x, int y, int z) const {
    // Periodic
    x = x % m_subLatticeCount.x;
    y = y % m_subLatticeCount.y;
    z = z % m_subLatticeCount.z;
    x = (x < 0) ? m_subLatticeCount.x + x : x;
    y = (y < 0) ? m_subLatticeCount.y + y : y;
    z = (z < 0) ? m_subLatticeCount.z + z : z;
    return (
        m_subLattices.data())[I3D(x, y, z, m_subLatticeCount.x,
                                  m_subLatticeCount.y, m_subLatticeCount.z)];
  }

  inline SubLattice getSubLattice(glm::ivec3 pos) {
    return getSubLattice(pos.x, pos.y, pos.z);
  }
};
