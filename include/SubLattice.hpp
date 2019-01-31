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

  SubLatticeSegment()
      : m_src(0, 0, 0),
        m_dst(0, 0, 0),
        m_srcStride(0),
        m_dstStride(0),
        m_segmentLength(0),
        m_numSegments(0) {}
};

class SubLattice {
 private:
  glm::ivec3 m_min, m_max, m_halo;

 public:
  /**
   * @brief Construct a new SubLattice object
   *
   * @param min Minimum point of subLattice on the lattice
   * @param max Maximum point of subLattice on the lattice
   */
  inline SubLattice(glm::ivec3 min, glm::ivec3 max, glm::ivec3 halo)
      : m_min(min), m_max(max), m_halo(halo) {}
  /**
   * @brief Construct a new empty SubLattice
   *
   */
  inline SubLattice() : m_min(0, 0, 0), m_max(0, 0, 0), m_halo(0, 0, 0) {}
  /**
   * @brief Copy constructor
   * @param other Another subLattice
   */
  inline SubLattice(const SubLattice &other)
      : m_min(other.m_min), m_max(other.m_max), m_halo(other.m_halo) {}
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
   * @brief Get the size of the halo in three dimensions
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getHalo() const { return m_halo; }
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
    return dims + m_halo * 2;
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
  int toLocalIndex(unsigned int df_idx, int x, int y, int z);

  /**
   * @brief Finds the axis with the least slice area when cut
   *
   * @return SubLattice::Enum The axis
   */
  D3Q7::Enum getDivisionAxis();

  void getHaloPlane(glm::ivec3 direction, glm::ivec3 *src, size_t *srcStride,
                    glm::ivec3 srcDim, glm::ivec3 *dst, size_t *dstStride,
                    glm::ivec3 dstDim, size_t *width, size_t *height);

  SubLatticeSegment getSubLatticeSegment(glm::ivec3 direction,
                                         SubLattice neighbour);

  void split(unsigned int divisions, glm::ivec3 *subLatticeCount,
             std::vector<SubLattice> *subLattices, unsigned int haloSize);
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
    ::hash_combine(seed, p.getHalo().x);
    ::hash_combine(seed, p.getHalo().y);
    ::hash_combine(seed, p.getHalo().z);
    return seed;
  }
};
}  // namespace std
