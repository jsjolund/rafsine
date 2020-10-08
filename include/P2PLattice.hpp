#pragma once

#include <cuda_profiler_api.h>
#include <nvToolsExtCudaRt.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "DistributedLattice.hpp"

bool enablePeerAccess(int srcDev, int dstDev, std::vector<bool>* p2pList);
void disableAllPeerAccess(int srcDev, std::vector<bool>* p2pList);
void disablePeerAccess(int srcDev, int dstDev, std::vector<bool>* p2pList);

class P2PLattice : public DistributedLattice {
 protected:
  /**
   * @brief Holds the configuration of P2P access and streams for one GPU
   */
  class DeviceParams {
   public:
    //! List of P2P access enabled
    std::vector<bool> m_p2pList;

    //! LBM kernel compute stream
    cudaStream_t m_computeStream;
    //! LBM kernel lattice boundary compute stream
    cudaStream_t m_computeBoundaryStream;
    //! Plot gathering stream (to the rendering GPU)
    cudaStream_t m_plotStream;
    //! Average gathering stream (to averaging GPU)
    cudaStream_t m_avgStream;
    //! Velocity df ghostLayer exchange stream to each neighbour
    std::vector<cudaStream_t> m_dfGhostLayerStreams;
    //! Temperature df ghostLayer exchange stream to each neighbour
    std::vector<cudaStream_t> m_dfTGhostLayerStreams;

    explicit DeviceParams(size_t nd)
        : m_p2pList(nd, false),
          m_computeStream(0),
          m_computeBoundaryStream(0),
          m_plotStream(0),
          m_avgStream(0),
          m_dfGhostLayerStreams(nd, 0),
          m_dfTGhostLayerStreams(nd, 0) {}
  };

  std::vector<DeviceParams*> m_deviceParams;

 public:
  inline cudaStream_t getDfGhostLayerStream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->m_dfGhostLayerStreams.at(dstDev);
  }
  inline cudaStream_t getDfTGhostLayerStream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->m_dfTGhostLayerStreams.at(dstDev);
  }
  inline cudaStream_t getPlotStream(int srcDev) {
    return m_deviceParams.at(srcDev)->m_plotStream;
  }
  inline cudaStream_t getAvgStream(int srcDev) {
    return m_deviceParams.at(srcDev)->m_avgStream;
  }
  inline cudaStream_t getComputeStream(int srcDev) {
    return m_deviceParams.at(srcDev)->m_computeStream;
  }
  inline cudaStream_t getComputeBoundaryStream(int srcDev) {
    return m_deviceParams.at(srcDev)->m_computeBoundaryStream;
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
    for (int i = 0; i < m_nd; i++) {
      if (i != dev && p2pList.at(dev)) count++;
    }
    return count;
  }

  P2PLattice(size_t nx,
             size_t ny,
             size_t nz,
             size_t nd,
             const D3Q4::Enum partitioning);
  ~P2PLattice();
};
