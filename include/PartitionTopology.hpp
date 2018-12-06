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

class Partition {
 private:
  glm::ivec3 m_min, m_max;

 public:
  /**
   * @brief 3D axis enum
   *
   */
  enum Enum { X_AXIS, Y_AXIS, Z_AXIS };

  /**
   * @brief Construct a new Partition object
   *
   * @param min Minimum point of partition on the lattice
   * @param max Maximum point of partition on the lattice
   */
  inline Partition(glm::ivec3 min, glm::ivec3 max) : m_min(min), m_max(max) {}
  /**
   * @brief Construct a new empty Partition
   *
   */
  inline Partition() {}
  /**
   * @brief Copy constructor
   * @param other Another partition
   */
  inline Partition(const Partition &other)
      : m_min(other.m_min), m_max(other.m_max) {}
  inline ~Partition() {}
  /**
   * @brief Get the minimum point of partition on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getLatticeMin() const { return glm::ivec3(m_min); }
  /**
   * @brief Get the maximum point of partition on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getLatticeMax() const { return glm::ivec3(m_max); }
  /**
   * @brief Get the 3D sizes of the partition on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getLatticeDims() const { return m_max - m_min; }
  /**
   * @brief Get the total size of the partition on the lattice
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
  inline glm::ivec3 getQDims() const {
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
  inline size_t getQStride() const {
    glm::ivec3 dims = getQDims();
    return dims.x * dims.y * dims.z;
  }

  /**
   * @brief Calculate index in partition array from global coordinates, such
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
   * @return Partition::Enum The axis
   */
  Partition::Enum getDivisionAxis();
  /**
   * @brief Get the halo for a specific direction as a list of points
   *
   * @param direction
   * @param srcPoints
   * @param haloPoints
   */
  void getHalo(glm::ivec3 direction, std::vector<glm::ivec3> *srcPoints,
               std::vector<glm::ivec3> *haloPoints);
};
bool operator==(Partition const &a, Partition const &b);
std::ostream &operator<<(std::ostream &os, Partition p);

class HaloParamsLocal {
 public:
  thrust::host_vector<int> srcIndexH;
  thrust::host_vector<int> dstIndexH;
  thrust::device_vector<int> srcIndexD;
  thrust::device_vector<int> dstIndexD;

  inline HaloParamsLocal()
      : srcIndexH(0), dstIndexH(0), srcIndexD(0), dstIndexD(0) {}
};

namespace std {
template <>
struct hash<Partition> {
  std::size_t operator()(const Partition &p) const {
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

class Topology {
 protected:
  std::vector<Partition *> m_partitions;

  glm::ivec3 m_latticeSize;
  glm::ivec3 m_partitionCount;
  // Number of arrays (or directions for distribution functions)
  const unsigned int m_Q;
  const glm::ivec3 *m_Qvecs;
  std::unordered_map<Partition, glm::ivec3> m_partitionPositions;

 public:
  std::unordered_map<Partition,
                     std::unordered_map<Partition, HaloParamsLocal *>>
      m_haloData;

  Partition getNeighbour(Partition partition, int dfIdx);
  inline std::vector<Partition *> getPartitions() { return m_partitions; }
  inline glm::ivec3 getLatticeDims() const { return glm::ivec3(m_latticeSize); }
  inline size_t getLatticeSize() const {
    return m_latticeSize.x * m_latticeSize.y * m_latticeSize.z;
  }
  inline glm::ivec3 getNumPartitions() { return glm::ivec3(m_partitionCount); }
  inline int getNumPartitionsTotal() { return m_partitions.size(); }

  Topology(unsigned int Q, unsigned int latticeSizeX, unsigned int latticeSizeY,
           unsigned int latticeSizeZ, unsigned int subdivisions = 0);

  inline ~Topology() {
    for (Partition *p : m_partitions) delete p;
  }

  Partition *getPartitionContaining(unsigned int x, unsigned int y,
                                    unsigned int z);

  inline Partition *getPartition(int x, int y, int z) const {
    // Periodic
    x = x % m_partitionCount.x;
    y = y % m_partitionCount.y;
    z = z % m_partitionCount.z;
    x = (x < 0) ? m_partitionCount.x + x : x;
    y = (y < 0) ? m_partitionCount.y + y : y;
    z = (z < 0) ? m_partitionCount.z + z : z;
    return (m_partitions.data())[I3D(x, y, z, m_partitionCount.x,
                                     m_partitionCount.y, m_partitionCount.z)];
  }

  inline Partition *getPartition(glm::ivec3 pos) {
    return getPartition(pos.x, pos.y, pos.z);
  }
};
