#pragma once

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DdQq.hpp"
#include "StdUtils.hpp"
#include "Vector3.hpp"

class GhostLayerParameters {
 public:
  vector3<size_t> m_src;
  vector3<size_t> m_dst;
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

std::ostream& operator<<(std::ostream& os, const GhostLayerParameters p);

class Partition {
 private:
  //! Minimum position in lattice
  vector3<unsigned int> m_min;
  //! Maximum position in lattice
  vector3<unsigned int> m_max;
  //! Size of ghostLayer buffers
  vector3<size_t> m_ghostLayer;

 public:
  unsigned int intersect(vector3<unsigned int> minIn, vector3<unsigned int> maxIn, vector3<unsigned int>* minOut,
                vector3<unsigned int>* maxOut) const;

  /**
   * @brief Construct a new Partition object
   *
   * @param min Minimum point in lattice
   * @param max Maximum point in lattice
   */
  inline Partition(vector3<unsigned int> minimum,
                   vector3<unsigned int> maximum,
                   vector3<size_t> ghostLayer)
      : m_min(minimum), m_max(maximum), m_ghostLayer(ghostLayer) {
    // vector3<int> size(m_max.x() - m_min.x(), m_max.y() - m_min.y(),
    //                   m_max.z() - m_min.z());
    // assert(size.x() >= 0 && size.y() >= 0 && size.z() >= 0);
  }

  /**
   * @brief Construct a new empty Partition
   *
   */
  inline Partition() : m_min(0, 0, 0), m_max(0, 0, 0), m_ghostLayer(0, 0, 0) {}

  /**
   * @brief Copy constructor
   * @param other Another partition
   */
  inline Partition(const Partition& other)
      : m_min(other.m_min),
        m_max(other.m_max),
        m_ghostLayer(other.m_ghostLayer) {}

  Partition& operator=(const Partition& other) {
    m_min = other.m_min;
    m_max = other.m_max;
    m_ghostLayer = other.m_ghostLayer;
    return *this;
  }

  /**
   * @brief Check if volume of partition is zero
   *
   * @return true
   * @return false
   */
  inline bool isEmpty() const {
    return m_min.x() == 0 && m_min.y() == 0 && m_min.z() == 0 &&
           m_max.x() == 0 && m_max.y() == 0 && m_max.z() == 0 &&
           m_ghostLayer.x() == 0 && m_ghostLayer.y() == 0 &&
           m_ghostLayer.z() == 0;
  }
  /**
   * @brief Get the minimum point of partition on the lattice
   *
   * @return vector3<unsigned int>
   */
  CUDA_CALLABLE_MEMBER inline vector3<unsigned int> getMin() const {
    return m_min;
  }
  /**
   * @brief Get the maximum point of partition on the lattice
   *
   * @return vector3<unsigned int>
   */
  CUDA_CALLABLE_MEMBER inline vector3<unsigned int> getMax() const {
    return m_max;
  }
  /**
   * @brief Get the size of the ghostLayer in three dimensions
   *
   * @return vector3<int>
   */
  CUDA_CALLABLE_MEMBER inline vector3<size_t> getGhostLayer() const {
    return m_ghostLayer;
  }
  /**
   * @brief Get the 3D sizes of the partition on the lattice
   *
   * @return vector3<int>
   */
  CUDA_CALLABLE_MEMBER inline vector3<size_t> getExtents() const {
    return vector3<size_t>(m_max.x() - m_min.x(), m_max.y() - m_min.y(),
                           m_max.z() - m_min.z());
  }
  /**
   * @brief Get the total size of the partition on the lattice
   *
   * @return size_t
   */
  CUDA_CALLABLE_MEMBER inline size_t getSize() const {
    vector3<size_t> exts = getExtents();
    if (exts.x() == 0 && exts.y() == 0 && exts.z() == 0) return 0;
    exts.x() = max(exts.x(), (size_t)1);
    exts.y() = max(exts.y(), (size_t)1);
    exts.z() = max(exts.z(), (size_t)1);
    return exts.x() * exts.y() * exts.z();
  }
  /**
   * @brief Get the 3D array dimensions of the first order q of the distribution
   * function (including ghostLayers)
   *
   * @return vector3<int>
   */
  CUDA_CALLABLE_MEMBER inline vector3<size_t> getArrayExtents() const {
    vector3<size_t> exts = getExtents();
    if (exts.x() == 0 && exts.y() == 0 && exts.z() == 0)
      return vector3<size_t>(0, 0, 0);
    exts += vector3<size_t>(m_ghostLayer.x() * (size_t)2, m_ghostLayer.y() * (size_t)2,
                            m_ghostLayer.z() * (size_t)2);
    return exts;
  }
  /**
   * @brief Get the array size of the first order q of the distribution
   * function (including ghostLayers), or in other words, the array stride
   * between different q > 1
   *
   * @return vector3<int>
   */
  inline size_t getArrayStride() const {
    vector3<size_t> exts = getExtents();
    if (exts.x() == 0 && exts.y() == 0 && exts.z() == 0) return 0;
    exts += m_ghostLayer * (size_t)2;
    exts.x() = max(exts.x(), (size_t)1);
    exts.y() = max(exts.y(), (size_t)1);
    exts.z() = max(exts.z(), (size_t)1);
    return exts.x() * exts.y() * exts.z();
  }

  GhostLayerParameters getGhostLayer(vector3<int> direction,
                                     Partition neighbour) const;

  void split(std::vector<Partition>* partitions,
             vector3<int>* partitionCount,
             unsigned int nd,
             unsigned int ghostLayerSize,
             D3Q4::Enum partitioning) const;
};
bool operator==(Partition const& a, Partition const& b);
std::ostream& operator<<(std::ostream& os, Partition p);

namespace std {
template <>
struct hash<Partition> {
  std::size_t operator()(const Partition& p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(&seed, p.getMin().x(), p.getMin().y(), p.getMin().z(),
                   p.getMax().x(), p.getMax().y(), p.getMax().z(),
                   p.getGhostLayer().x(), p.getGhostLayer().y(),
                   p.getGhostLayer().z());
    return seed;
  }
};
}  // namespace std
