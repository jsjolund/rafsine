#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "Lattice.hpp"

class DistributedLattice : public Lattice {
 protected:
  //! Number of CUDA devices
  const size_t m_nd;
  //! Maps a sub lattice to a CUDA device
  std::unordered_map<Partition, unsigned int> m_partitionDeviceMap;
  //! Maps a CUDA device number to a sub lattice
  std::vector<Partition> m_devicePartitionMap;

 public:
  /**
   * @brief Get number of CUDA devices for computing stream and collide
   *
   * @return size_t
   */
  inline size_t getnd() { return m_nd; }

  /**
   * @brief Get the device corresponding to lattice partition
   *
   * @param partition
   * @return unsigned int
   */
  inline unsigned int getPartitionDevice(Partition partition) {
    return m_partitionDeviceMap[partition];
  }

  /**
   * @brief Get the lattice partition corresponding to device
   *
   * @param devId
   * @return Partition
   */
  inline Partition getDevicePartition(unsigned int devId) {
    return m_devicePartitionMap.at(devId);
  }

  /**
   * @brief Construct a new Distributed Lattice
   *
   * @param nx Size on X-axis
   * @param ny Size on Y-axis
   * @param nz Size on Z-axis
   * @param nd Number of CUDA devices
   * @param ghostLayerSize Size of ghost layer (0-1)
   * @param partitioning Partitioning axis
   */
  DistributedLattice(const size_t nx,
                     const size_t ny,
                     const size_t nz,
                     const size_t nd,
                     const size_t ghostLayerSize,
                     const D3Q4::Enum partitioning)
      : Lattice(nx, ny, nz, nd, ghostLayerSize, partitioning),
        m_nd(nd),
        m_devicePartitionMap(nd) {
    std::vector<Partition> partitions = getPartitions();

    for (size_t i = 0; i < partitions.size(); i++) {
      Partition partition = partitions.at(i);
      // Distribute the workload. Calculate partitions and assign them to GPUs
      int devIndex = i % m_nd;
      m_partitionDeviceMap[partition] = devIndex;
      m_devicePartitionMap.at(devIndex) = Partition(partition);
    }
  }
};
