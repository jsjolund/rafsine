#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "DistributedLattice.hpp"

bool enablePeerAccess(int srcDev, int dstDev,
                      std::vector<bool> *peerAccessList);
void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList);

class P2PLattice : public DistributedLattice {
 protected:
  class DeviceParams {
   public:
    std::vector<bool> peerAccessList;   //!< List of P2P access enabled
    std::vector<cudaStream_t> streams;  //!< Cuda streams
    explicit DeviceParams(int numDevices)
        : peerAccessList(numDevices), streams(numDevices, 0) {}
  };
  std::vector<DeviceParams *> m_deviceParams;

 public:
  inline cudaStream_t getP2Pstream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->streams.at(dstDev);
  }

  P2PLattice(int nx, int ny, int nz, int numDevices);
  ~P2PLattice();
};
