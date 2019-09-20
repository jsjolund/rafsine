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
#include "Vec3.hpp"

namespace std {
template <>
struct hash<glm::ivec3> {
  std::size_t operator()(const glm::ivec3 &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(&seed, p.x, p.y, p.z);
    return seed;
  }
};
}  // namespace std

class GhostLayerParameters {
 public:
  glm::ivec3 m_src;
  glm::ivec3 m_dst;
  size_t m_spitch;
  size_t m_dpitch;
  size_t m_width;
  size_t m_height;

  GhostLayerParameters()
      : m_src(0, 0, 0),
        m_dst(0, 0, 0),
        m_spitch(0),
        m_dpitch(0),
        m_width(0),
        m_height(0) {}
};

std::ostream &operator<<(std::ostream &os, const GhostLayerParameters p);

class Partition {
 private:
  //! Minimum position in lattice
  glm::ivec3 m_min;
  //! Maximum position in lattice
  glm::ivec3 m_max;
  //! Size of halo buffers
  glm::ivec3 m_halo;

 public:
  int intersect(glm::ivec3 minIn, glm::ivec3 maxIn, glm::ivec3 *minOut,
                glm::ivec3 *maxOut) const;

  /**
   * @brief Construct a new Partition object
   *
   * @param min Minimum point in lattice
   * @param max Maximum point in lattice
   */
  inline Partition(glm::ivec3 minimum, glm::ivec3 maximum, glm::ivec3 halo)
      : m_min(minimum), m_max(maximum), m_halo(halo) {
    glm::ivec3 size(m_max.x - m_min.x, m_max.y - m_min.y, m_max.z - m_min.z);
    assert(size.x >= 0 && size.y >= 0 && size.z >= 0);
  }

  /**
   * @brief Construct a new empty Partition
   *
   */
  inline Partition() : m_min(0, 0, 0), m_max(0, 0, 0), m_halo(0, 0, 0) {}

  /**
   * @brief Copy constructor
   * @param other Another partition
   */
  inline Partition(const Partition &other)
      : m_min(other.m_min), m_max(other.m_max), m_halo(other.m_halo) {}

  /**
   * @brief Check if volume of partition is zero
   *
   * @return true
   * @return false
   */
  inline bool isEmpty() const {
    return m_min == glm::ivec3(0, 0, 0) && m_max == glm::ivec3(0, 0, 0) &&
           m_halo == glm::ivec3(0, 0, 0);
  }
  /**
   * @brief Get the minimum point of partition on the lattice
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getMin() const { return m_min; }
  /**
   * @brief Get the maximum point of partition on the lattice
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getMax() const { return m_max; }
  /**
   * @brief Get the size of the halo in three dimensions
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getGhostLayer() const {
    return m_halo;
  }
  /**
   * @brief Get the 3D sizes of the partition on the lattice
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getDims() const {
    return glm::ivec3(m_max.x - m_min.x, m_max.y - m_min.y, m_max.z - m_min.z);
  }
  /**
   * @brief Get the total size of the partition on the lattice
   *
   * @return size_t
   */
  CUDA_CALLABLE_MEMBER inline size_t getSize() const {
    glm::ivec3 dims = getDims();
    if (dims == glm::ivec3(0, 0, 0)) return 0;
    dims.x = max(dims.x, 1);
    dims.y = max(dims.y, 1);
    dims.z = max(dims.z, 1);
    return dims.x * dims.y * dims.z;
  }
  /**
   * @brief Get the 3D array dimensions of the first order q of the distribution
   * function (including halos)
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getArrayDims() const {
    glm::ivec3 dims = getDims();
    if (dims == glm::ivec3(0, 0, 0)) return glm::ivec3(0, 0, 0);
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
    glm::ivec3 dims = getDims();
    if (dims == glm::ivec3(0, 0, 0)) return 0;
    dims += m_halo * 2;
    dims.x = max(dims.x, 1);
    dims.y = max(dims.y, 1);
    dims.z = max(dims.z, 1);
    return dims.x * dims.y * dims.z;
  }

  /**
   * @brief Finds the axis with the least slice area when cut
   *
   * @return Partition::Enum The axis
   */
  D3Q4::Enum getDivisionAxis() const;

  GhostLayerParameters getGhostLayer(glm::ivec3 direction,
                                     Partition neighbour) const;

  void split(unsigned int divisions, glm::ivec3 *partitionCount,
             std::vector<Partition> *partitions, unsigned int haloSize) const;
};
bool operator==(Partition const &a, Partition const &b);
std::ostream &operator<<(std::ostream &os, Partition p);

namespace std {
template <>
struct hash<Partition> {
  std::size_t operator()(const Partition &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(&seed, p.getMin().x, p.getMin().y, p.getMin().z,
                   p.getMax().x, p.getMax().y, p.getMax().z,
                   p.getGhostLayer().x, p.getGhostLayer().y,
                   p.getGhostLayer().z);
    return seed;
  }
};
}  // namespace std
