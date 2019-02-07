#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "DistributedLattice.hpp"

bool enablePeerAccess(int srcDev, int dstDev, std::vector<bool> *p2pList);
void disableAllPeerAccess(int srcDev, std::vector<bool> *p2pList);
void disablePeerAccess(int srcDev, int dstDev, std::vector<bool> *p2pList);

class P2PLattice : public DistributedLattice {
 protected:
  class DeviceParams {
   public:
    //! List of P2P access enabled
    std::vector<bool> m_p2pList;

    //! Cuda streams for halo exchange
    cudaStream_t m_computeStream;
    std::vector<cudaStream_t> m_dfHaloStreams;
    std::vector<cudaStream_t> m_dfTHaloStreams;
    std::vector<cudaStream_t> m_plotStreams;

    explicit DeviceParams(int numDevices)
        : m_p2pList(numDevices, false),
          m_dfHaloStreams(numDevices, 0),
          m_dfTHaloStreams(numDevices, 0),
          m_plotStreams(numDevices, 0),
          m_computeStream(0) {}
  };

  std::vector<DeviceParams *> m_deviceParams;

 public:
  inline cudaStream_t getDfHaloStream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->m_dfHaloStreams.at(dstDev);
  }
  inline cudaStream_t getDfTHaloStream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->m_dfTHaloStreams.at(dstDev);
  }
  inline cudaStream_t getPlotStream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->m_plotStreams.at(dstDev);
  }
  inline cudaStream_t getComputeStream(int srcDev) {
    return m_deviceParams.at(srcDev)->m_computeStream;
  }
  inline std::vector<bool> getP2PConnections(int dev) {
    return std::vector<bool>(m_deviceParams.at(dev)->m_p2pList);
  }

  inline bool hasP2PConnection(int fromDev, int toDev) {
    return m_deviceParams.at(fromDev)->m_p2pList.at(toDev);
  }

  inline size_t getNumP2PConnections(int dev) {
    std::vector<bool> p2pList = m_deviceParams.at(dev)->m_p2pList;
    size_t count = 0;
    for (int i = 0; i < m_numDevices; i++) {
      if (i != dev && p2pList.at(dev)) count++;
    }
    return count;
  }

  P2PLattice(int nx, int ny, int nz, int numDevices);
  ~P2PLattice();
};
