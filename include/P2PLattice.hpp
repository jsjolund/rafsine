#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "DistributedLattice.hpp"

bool enablePeerAccess(int srcDev, int dstDev, std::vector<bool> *p2pList);
void disablePeerAccess(int srcDev, std::vector<bool> *p2pList);

class P2PLattice : public DistributedLattice {
 protected:
  class DeviceParams {
   public:
    std::vector<bool> p2pList;          //!< List of P2P access enabled
    std::vector<cudaStream_t> streams;  //!< Cuda streams
    explicit DeviceParams(int numDevices)
        : p2pList(numDevices), streams(numDevices, 0) {}
  };

  std::vector<DeviceParams *> m_deviceParams;

 public:
  inline cudaStream_t getP2Pstream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->streams.at(dstDev);
  }

  inline std::vector<bool> getP2PConnections(int dev) {
    return std::vector<bool>(m_deviceParams.at(dev)->p2pList);
  }

  inline size_t getNumP2PConnections(int dev) {
    std::vector<bool> p2pList = m_deviceParams.at(dev)->p2pList;
    size_t count = 0;
    for (int i = 0; i < m_numDevices; i++) {
      if (i != dev && p2pList.at(dev)) count++;
    }
    return count;
  }

  P2PLattice(int nx, int ny, int nz, int numDevices);
  ~P2PLattice();
};
