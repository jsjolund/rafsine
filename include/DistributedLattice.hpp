#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "Lattice.hpp"

class DistributedLattice : public Lattice {
 protected:
  //! Number of CUDA devices
  int m_numDevices;
  //! Maps a sub lattice to a CUDA device
  std::unordered_map<SubLattice, int> m_subLatticeDeviceMap;
  //! Maps a CUDA device number to a sub lattice
  std::vector<SubLattice> m_deviceSubLatticeMap;

 public:
  inline int getSubLatticeDevice(SubLattice subLattice) {
    return m_subLatticeDeviceMap[subLattice];
  }

  inline SubLattice getDeviceSubLattice(int devId) {
    return m_deviceSubLatticeMap.at(devId);
  }

  DistributedLattice(int nx, int ny, int nz, int numDevices = 1,
                     int haloSize = 0)
      : Lattice(nx, ny, nz, numDevices, haloSize),
        m_numDevices(numDevices),
        m_deviceSubLatticeMap(numDevices) {
    std::vector<SubLattice> subLattices = getSubLattices();

    for (int i = 0; i < subLattices.size(); i++) {
      SubLattice subLattice = subLattices.at(i);
      // Distribute the workload. Calculate subLattices and assign them to GPUs
      int devIndex = i % m_numDevices;
      m_subLatticeDeviceMap[subLattice] = devIndex;
      m_deviceSubLatticeMap.at(devIndex) = SubLattice(subLattice);
    }
  }
};
