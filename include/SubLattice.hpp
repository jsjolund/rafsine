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

class HaloSegment {
 public:
  glm::ivec3 m_src;
  glm::ivec3 m_dst;
  size_t m_spitch;
  size_t m_dpitch;
  size_t m_width;
  size_t m_height;

  HaloSegment()
      : m_src(0, 0, 0),
        m_dst(0, 0, 0),
        m_spitch(0),
        m_dpitch(0),
        m_width(0),
        m_height(0) {}
};

std::ostream &operator<<(std::ostream &os, const HaloSegment p);

class SubLattice {
 private:
  //! Minimum position in lattice
  glm::ivec3 m_min;
  //! Maximum position in lattice
  glm::ivec3 m_max;
  //! Size of halo buffers
  glm::ivec3 m_halo;
  //! Size of sub lattice
  glm::ivec3 m_size;

 public:
  int intersect(glm::ivec3 minIn, glm::ivec3 maxIn, glm::ivec3 *minOut,
                glm::ivec3 *maxOut) const {
    minOut->x = max(minIn.x, m_min.x);
    minOut->y = max(minIn.y, m_min.y);
    minOut->z = max(minIn.z, m_min.z);
    maxOut->x = min(maxIn.x, m_max.x);
    maxOut->y = min(maxIn.y, m_max.y);
    maxOut->z = min(maxIn.z, m_max.z);
    glm::ivec3 d = *maxOut - *minOut;
    d.x = (d.x < 0) ? 0 : d.x;
    d.y = (d.y < 0) ? 0 : d.y;
    d.z = (d.z < 0) ? 0 : d.z;
    return d.x * d.y * d.z;
  }

  /**
   * @brief Construct a new SubLattice object
   *
   * @param min Minimum point in lattice
   * @param max Maximum point in lattice
   */
  inline SubLattice(glm::ivec3 minimum, glm::ivec3 maximum, glm::ivec3 halo)
      : m_min(minimum),
        m_max(maximum),
        m_halo(halo),
        m_size(max(maximum.x - minimum.x, 1), max(maximum.y - minimum.y, 1),
               max(maximum.z - minimum.z, 1)) {
    assert((minimum.x <= maximum.x && minimum.y <= maximum.y &&
            minimum.z <= maximum.z));
  }

  /**
   * @brief Construct a new empty SubLattice
   *
   */
  inline SubLattice()
      : m_min(0, 0, 0), m_max(0, 0, 0), m_halo(0, 0, 0), m_size(0, 0, 0) {}

  /**
   * @brief Copy constructor
   * @param other Another subLattice
   */
  inline SubLattice(const SubLattice &other)
      : m_min(other.m_min),
        m_max(other.m_max),
        m_halo(other.m_halo),
        m_size(other.m_size) {}

  /**
   * @brief Check if volume of sublattice is zero
   *
   * @return true
   * @return false
   */
  inline bool isEmpty() const {
    return m_min == glm::ivec3(0, 0, 0) && m_max == glm::ivec3(0, 0, 0) &&
           m_halo == glm::ivec3(0, 0, 0);
  }
  /**
   * @brief Get the minimum point of subLattice on the lattice
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getMin() const { return m_min; }
  /**
   * @brief Get the maximum point of subLattice on the lattice
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getMax() const { return m_max; }
  /**
   * @brief Get the size of the halo in three dimensions
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getHalo() const { return m_halo; }
  /**
   * @brief Get the 3D sizes of the subLattice on the lattice
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getDims() const { return m_size; }
  /**
   * @brief Get the total size of the subLattice on the lattice
   *
   * @return size_t
   */
  CUDA_CALLABLE_MEMBER inline size_t getSize() const {
    glm::ivec3 dims = getDims();
    return dims.x * dims.y * dims.z;
  }
  CUDA_CALLABLE_MEMBER inline size_t getNumBoundaryElements() const {
    return 2 *
           (m_size.x * m_size.y * m_halo.z + m_size.x * m_size.z * m_halo.y +
            m_size.y * m_size.z * m_halo.x);
  }
  /**
   * @brief Get the 3D array dimensions of the first order q of the distribution
   * function (including halos)
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getArrayDims() const {
    glm::ivec3 dims = getDims();
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
   * @brief Finds the axis with the least slice area when cut
   *
   * @return SubLattice::Enum The axis
   */
  D3Q4::Enum getDivisionAxis() const;

  HaloSegment getHalo(glm::ivec3 direction, SubLattice neighbour) const;

  void split(unsigned int divisions, glm::ivec3 *subLatticeCount,
             std::vector<SubLattice> *subLattices, unsigned int haloSize) const;
};
bool operator==(SubLattice const &a, SubLattice const &b);
std::ostream &operator<<(std::ostream &os, SubLattice p);

namespace std {
template <>
struct hash<SubLattice> {
  std::size_t operator()(const SubLattice &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, p.getMin().x);
    ::hash_combine(seed, p.getMin().y);
    ::hash_combine(seed, p.getMin().z);
    ::hash_combine(seed, p.getMax().x);
    ::hash_combine(seed, p.getMax().y);
    ::hash_combine(seed, p.getMax().z);
    ::hash_combine(seed, p.getHalo().x);
    ::hash_combine(seed, p.getHalo().y);
    ::hash_combine(seed, p.getHalo().z);
    return seed;
  }
};
}  // namespace std
