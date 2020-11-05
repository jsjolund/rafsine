#pragma once

#include <cuda_profiler_api.h>
#include <nvToolsExtCudaRt.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "DistributedLattice.hpp"

/**
 * @brief Enable P2P access from one CUDA GPU to another
 *
 * @param srcDev Source GPU index
 * @param dstDev Destination GPU index
 * @param p2pList Boolean vector will be marked true at index if dstDev is P2P
 * accessible
 * @return true Enabling P2P access succeded
 * @return false Enabling P2P access failed
 */
bool enablePeerAccess(unsigned int srcDev,
                      unsigned int dstDev,
                      std::vector<bool>* p2pList);
/**
 * @brief Disable P2P access from one CUDA GPU to device indices marked true in
 * boolean list
 *
 * @param srcDev Source GPU index
 * @param p2pList Boolean vector marked true where P2P access is currently
 * enabled
 */
void disableAllPeerAccess(unsigned int srcDev, std::vector<bool>* p2pList);
/**
 * @brief Disable P2P access from one CUDA GPU to another
 *
 * @param srcDev Source GPU index
 * @param dstDev Destination GPU index
 * @param p2pList Boolean vector will be marked false at index if dstDev P2P
 * access was disabled
 */
void disablePeerAccess(unsigned int srcDev,
                       unsigned int dstDev,
                       std::vector<bool>* p2pList);

/**
 * @brief Lattice partitioned between one or multiple CUDA GPUs
 *
 */
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
  /**
   * @brief CUDA stream for velocity distribution ghost layer transfers
   *
   * @param srcDev Source GPU index
   * @param dstDev Destination GPU index
   * @return cudaStream_t The CUDA stream
   */
  inline cudaStream_t getDfGhostLayerStream(unsigned int srcDev,
                                            unsigned int dstDev) {
    return m_deviceParams.at(srcDev)->m_dfGhostLayerStreams.at(dstDev);
  }
  /**
   * @brief CUDA stream for temperature distribution ghost layer transfers
   *
   * @param srcDev Source GPU index
   * @param dstDev Destination GPU index
   * @return cudaStream_t The CUDA stream
   */
  inline cudaStream_t getDfTGhostLayerStream(unsigned int srcDev,
                                             unsigned int dstDev) {
    return m_deviceParams.at(srcDev)->m_dfTGhostLayerStreams.at(dstDev);
  }
  /**
   * @brief CUDA stream for graphical plot operations
   *
   * @param srcDev Source GPU index
   * @param dstDev Destination GPU index
   * @return cudaStream_t The CUDA stream
   */
  inline cudaStream_t getPlotStream(unsigned int srcDev) {
    return m_deviceParams.at(srcDev)->m_plotStream;
  }
  /**
   * @brief CUDA stream for averaging operations
   *
   * @param srcDev Source GPU index
   * @param dstDev Destination GPU index
   * @return cudaStream_t The CUDA stream
   */
  inline cudaStream_t getAvgStream(unsigned int srcDev) {
    return m_deviceParams.at(srcDev)->m_avgStream;
  }
  /**
   * @brief CUDA stream for internal (non-boundary) lattice calculations
   *
   * @param srcDev Source GPU index
   * @param dstDev Destination GPU index
   * @return cudaStream_t The CUDA stream
   */
  inline cudaStream_t getComputeStream(unsigned int srcDev) {
    return m_deviceParams.at(srcDev)->m_computeStream;
  }
  /**
   * @brief CUDA stream for boundary (non-internal) lattice calculations
   *
   * @param srcDev Source GPU index
   * @param dstDev Destination GPU index
   * @return cudaStream_t The CUDA stream
   */
  inline cudaStream_t getComputeBoundaryStream(unsigned int srcDev) {
    return m_deviceParams.at(srcDev)->m_computeBoundaryStream;
  }
  /**
   * @brief Get a boolean list of which P2P connections are enabled for device
   *
   * @param dev GPU index
   * @return std::vector<bool> Boolean list marked true where P2P access is
   * enabled
   */
  inline std::vector<bool> getP2PConnections(unsigned int dev) {
    return std::vector<bool>(m_deviceParams.at(dev)->m_p2pList);
  }
  /**
   * @brief Check if two devices have P2P access
   *
   * @param fromDev
   * @param toDev
   * @return true
   * @return false
   */
  inline bool hasP2PConnection(unsigned int fromDev, unsigned int toDev) {
    return m_deviceParams.at(fromDev)->m_p2pList.at(toDev);
  }
  /**
   * @brief Calculate number of P2P connections from device
   *
   * @param dev
   * @return size_t
   */
  inline size_t getNumP2PConnections(unsigned int dev) {
    std::vector<bool> p2pList = m_deviceParams.at(dev)->m_p2pList;
    size_t count = 0;
    for (unsigned int i = 0; i < m_nd; i++) {
      if (i != dev && p2pList.at(dev)) count++;
    }
    return count;
  }
  /**
   * @brief Construct a new P2PLattice
   *
   * @param nx X-axis size
   * @param ny Y-axis size
   * @param nz Z-axis size
   * @param nd Number of CUDA devices to parition on
   * @param partitioning Axes to partition lattice on
   */
  P2PLattice(size_t nx,
             size_t ny,
             size_t nz,
             size_t nd,
             const D3Q4::Enum partitioning);
  /**
   * @brief Destroy the P2PLattice object
   */
  ~P2PLattice();
};
