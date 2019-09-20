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
  const int m_numDevices;
  //! Maps a sub lattice to a CUDA device
  std::unordered_map<Partition, int> m_partitionDeviceMap;
  //! Maps a CUDA device number to a sub lattice
  std::vector<Partition> m_devicePartitionMap;

 public:
  inline int getNumDevices() { return m_numDevices; }
  inline int getPartitionDevice(Partition partition) {
    return m_partitionDeviceMap[partition];
  }

  inline Partition getDevicePartition(int devId) {
    return m_devicePartitionMap.at(devId);
  }

  DistributedLattice(int nx, int ny, int nz, int numDevices = 1,
                     int haloSize = 0)
      : Lattice(nx, ny, nz, numDevices, haloSize),
        m_numDevices(numDevices),
        m_devicePartitionMap(numDevices) {
    std::vector<Partition> partitions = getPartitions();

    for (int i = 0; i < partitions.size(); i++) {
      Partition partition = partitions.at(i);
      // Distribute the workload. Calculate partitions and assign them to GPUs
      int devIndex = i % m_numDevices;
      m_partitionDeviceMap[partition] = devIndex;
      m_devicePartitionMap.at(devIndex) = Partition(partition);
    }
  }
};
