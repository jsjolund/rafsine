#pragma once

#include <Eigen/Geometry>

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DdQq.hpp"
#include "StdUtils.hpp"

class GhostLayerParameters {
 public:
  Eigen::Vector3i m_src;
  Eigen::Vector3i m_dst;
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
  Eigen::Vector3i m_min;
  //! Maximum position in lattice
  Eigen::Vector3i m_max;
  //! Size of ghostLayer buffers
  Eigen::Vector3i m_ghostLayer;

 public:
  int intersect(Eigen::Vector3i minIn,
                Eigen::Vector3i maxIn,
                Eigen::Vector3i* minOut,
                Eigen::Vector3i* maxOut) const;

  /**
   * @brief Construct a new Partition object
   *
   * @param min Minimum point in lattice
   * @param max Maximum point in lattice
   */
  inline Partition(Eigen::Vector3i minimum,
                   Eigen::Vector3i maximum,
                   Eigen::Vector3i ghostLayer)
      : m_min(minimum), m_max(maximum), m_ghostLayer(ghostLayer) {
    Eigen::Vector3i size(m_max.x() - m_min.x(), m_max.y() - m_min.y(),
                         m_max.z() - m_min.z());
    assert(size.x() >= 0 && size.y() >= 0 && size.z() >= 0);
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
   * @return Eigen::Vector3i
   */
  CUDA_CALLABLE_MEMBER inline Eigen::Vector3i getMin() const { return m_min; }
  /**
   * @brief Get the maximum point of partition on the lattice
   *
   * @return Eigen::Vector3i
   */
  CUDA_CALLABLE_MEMBER inline Eigen::Vector3i getMax() const { return m_max; }
  /**
   * @brief Get the size of the ghostLayer in three dimensions
   *
   * @return Eigen::Vector3i
   */
  CUDA_CALLABLE_MEMBER inline Eigen::Vector3i getGhostLayer() const {
    return m_ghostLayer;
  }
  /**
   * @brief Get the 3D sizes of the partition on the lattice
   *
   * @return Eigen::Vector3i
   */
  CUDA_CALLABLE_MEMBER inline Eigen::Vector3i getExtents() const {
    return Eigen::Vector3i(m_max.x() - m_min.x(), m_max.y() - m_min.y(),
                           m_max.z() - m_min.z());
  }
  /**
   * @brief Get the total size of the partition on the lattice
   *
   * @return size_t
   */
  CUDA_CALLABLE_MEMBER inline size_t getSize() const {
    Eigen::Vector3i exts = getExtents();
    if (exts.x() == 0 && exts.y() == 0 && exts.z() == 0) return 0;
    exts.x() = max(exts.x(), 1);
    exts.y() = max(exts.y(), 1);
    exts.z() = max(exts.z(), 1);
    return exts.x() * exts.y() * exts.z();
  }
  /**
   * @brief Get the 3D array dimensions of the first order q of the distribution
   * function (including ghostLayers)
   *
   * @return Eigen::Vector3i
   */
  CUDA_CALLABLE_MEMBER inline Eigen::Vector3i getArrayExtents() const {
    Eigen::Vector3i exts = getExtents();
    if (exts.x() == 0 && exts.y() == 0 && exts.z() == 0)
      return Eigen::Vector3i(0, 0, 0);
    return exts + m_ghostLayer * 2;
  }
  /**
   * @brief Get the array size of the first order q of the distribution
   * function (including ghostLayers), or in other words, the array stride
   * between different q > 1
   *
   * @return Eigen::Vector3i
   */
  inline size_t getArrayStride() const {
    Eigen::Vector3i exts = getExtents();
    if (exts.x() == 0 && exts.y() == 0 && exts.z() == 0) return 0;
    exts += m_ghostLayer * 2;
    exts.x() = max(exts.x(), 1);
    exts.y() = max(exts.y(), 1);
    exts.z() = max(exts.z(), 1);
    return exts.x() * exts.y() * exts.z();
  }

  GhostLayerParameters getGhostLayer(Eigen::Vector3i direction,
                                     Partition neighbour) const;

  void split(std::vector<Partition>* partitions,
             Eigen::Vector3i* partitionCount,
             unsigned int divisions,
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
