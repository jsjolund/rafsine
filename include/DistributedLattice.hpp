#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "DistributionFunction.hpp"
#include "Lattice.hpp"

bool enablePeerAccess(int srcDev, int dstDev,
                      std::vector<bool> *peerAccessList);
void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList);

class DistributedLattice {
 protected:
  class DeviceParams {
   public:
    std::vector<bool> peerAccessList;   //!< List of P2P access enabled
    std::vector<cudaStream_t> streams;  //!< Cuda streams
    explicit DeviceParams(int numDevices)
        : peerAccessList(numDevices), streams(numDevices, 0) {}
  };

  // Number of CUDA devices
  int m_numDevices;
  std::vector<DeviceParams *> m_deviceParams;
  std::unordered_map<SubLattice, int> m_subLatticeDeviceMap;
  std::vector<SubLattice> m_deviceSubLatticeMap;

 public:
  void haloExchange(SubLattice subLattice, DistributionFunction *df,
                    SubLattice neighbour, DistributionFunction *ndf,
                    UnitVector::Enum direction, cudaStream_t stream = 0);

  inline cudaStream_t getP2Pstream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->streams.at(dstDev);
  }
  inline int getDeviceFromSubLattice(SubLattice subLattice) {
    return m_subLatticeDeviceMap[subLattice];
  }
  inline SubLattice getSubLatticeFromDevice(int devId) {
    return m_deviceSubLatticeMap.at(devId);
  }

  DistributedLattice(int numDevices, int nx, int ny, int nz);
  ~DistributedLattice();
};
