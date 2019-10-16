#pragma once

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "Eigen/Geometry"
#include "Partition.hpp"

/**
 * @brief An LBM lattice partitioned to be distributed between multiple GPUs
 */
class Lattice {
 protected:
  //! The size of the entire lattice
  Eigen::Vector3i m_latticeSize;
  //! A list of partitions representing domain decomposition
  std::vector<Partition> m_partitions;
  //! The number of partitions in three dimensions
  Eigen::Vector3i m_partitionCount;
  //! Maps partitions to their positions in domain decomposition
  std::unordered_map<Partition, Eigen::Vector3i> m_partitionPositions;
  //! Maps the ghostLayer exchange parameters between two adjacent partitions
  std::unordered_map<
      Partition,
      std::unordered_map<Partition, std::vector<GhostLayerParameters>>>
      m_segments;

 public:
  /**
   * @brief Get the neighbouring partition in a certain direction.
   * @param partition
   * @param direction
   * @return Partition
   */
  inline Partition getNeighbour(Partition partition,
                                Eigen::Vector3i direction) {
    Eigen::Vector3i partPos = m_partitionPositions[partition];
    return getPartition(partPos + direction);
  }
  /**
   * @brief Get the neighbouring partition in a certain direction.
   * @param partition
   * @param direction
   * @return Partition
   */
  inline Partition getNeighbour(Partition partition, D3Q7::Enum direction) {
    return getNeighbour(partition, D3Q27[direction]);
  }
  /**
   * @brief Get the ghost layer parameters for copying between GPUs using
   * cudaMemcpy2D()
   *
   * @param partition
   * @param neighbour
   * @param direction
   * @return GhostLayerParameters
   */
  GhostLayerParameters getGhostLayer(Partition partition,
                                     Partition neighbour,
                                     D3Q7::Enum direction) {
    return m_segments[partition][neighbour].at(direction);
  }
  /**
   * @brief Get a list of all partitions in this lattice
   *
   * @return std::vector<Partition>
   */
  inline std::vector<Partition> getPartitions() const { return m_partitions; }
  /**
   * @brief Get the size/extents of the lattice in 3D
   *
   * @return Eigen::Vector3i
   */
  inline Eigen::Vector3i getExtents() const { return m_latticeSize; }
  /**
   * @brief Get the total number of lattice sites
   *
   * @return size_t
   */
  inline size_t getSize() const {
    return m_latticeSize.x() * m_latticeSize.y() * m_latticeSize.z();
  }
  /**
   * @brief Get an integer vector representing how many times the lattice was
   * partitioned/divided along each 3D axis
   *
   * @return Eigen::Vector3i
   */
  inline Eigen::Vector3i getNumPartitions() const { return m_partitionCount; }
  /**
   * @brief Get the total number of lattice partitions
   *
   * @return int
   */
  inline int getNumPartitionsTotal() const { return m_partitions.size(); }

  Lattice(unsigned int latticeSizeX,
          unsigned int latticeSizeY,
          unsigned int latticeSizeZ,
          unsigned int subdivisions = 1,
          unsigned int ghostLayerSize = 0);
  /**
   * @brief Get the partition containing the 3D lattice coorindate
   *
   * @param x
   * @param y
   * @param z
   * @return Partition
   */
  Partition getPartitionContaining(unsigned int x,
                                   unsigned int y,
                                   unsigned int z) const;
  /**
   * @brief Get a specific partition by its position in the domain decomposition
   *
   * @param x Integer between 0 and less than number of divisions along x-axis
   * @param y Integer between 0 and less than number of divisions along y-axis
   * @param z Integer between 0 and less than number of divisions along z-axis
   * @return Partition
   */
  inline Partition getPartition(int x = 0, int y = 0, int z = 0) const {
    // Periodic
    x = x % m_partitionCount.x();
    y = y % m_partitionCount.y();
    z = z % m_partitionCount.z();
    x = (x < 0) ? m_partitionCount.x() + x : x;
    y = (y < 0) ? m_partitionCount.y() + y : y;
    z = (z < 0) ? m_partitionCount.z() + z : z;
    return (
        m_partitions.data())[I3D(Eigen::Vector3i(x, y, z), m_partitionCount)];
  }

  inline Partition getPartition(Eigen::Vector3i pos) const {
    return getPartition(pos.x(), pos.y(), pos.z());
  }
};
