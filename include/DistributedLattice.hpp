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
  inline size_t getnd() { return m_nd; }
  inline unsigned int getPartitionDevice(Partition partition) {
    return m_partitionDeviceMap[partition];
  }

  inline Partition getDevicePartition(unsigned int devId) {
    return m_devicePartitionMap.at(devId);
  }

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
