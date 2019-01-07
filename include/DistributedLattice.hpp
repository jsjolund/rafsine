#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "PartitionTopology.hpp"

class DistributedLattice {
 private:
  class DeviceParams {
   public:
    std::vector<bool> peerAccessList;   //!< List of P2P access enabled
    std::vector<cudaStream_t> streams;  //!< Cuda streams
    DeviceParams(int numDevices)
        : peerAccessList(numDevices), streams(numDevices, 0) {}
  };

  // Number of CUDA devices
  int m_numDevices;
  std::vector<DeviceParams *> m_deviceParams;
  std::unordered_map<Partition, int> m_partitionDeviceMap;
  std::vector<Partition> m_devicePartitionMap;

 public:
  inline void haloExchange(Partition::Enum axis, 
                          DistributionFunction *df,
                           Partition partition,
                          DistributionFunction *ndf,
                           Partition partition,
                           ) {
    int neighbourIdx;

    Partition neighbour = df->getNeighbour(partition, neighbourIdx);
    DistributionFunction *ndf =
        m_computeParams.at(getDeviceFromPartition(neighbour))->df_tmp;
    std::vector<PartitionSegment> segments =
        df->m_segments[partition][neighbour];
    for (int i = 0; i < 9; i++) {
      int qSrc = D3Q27ranks[0][i];
      int qDst = D3Q27ranks[1][i];
      if (qSrc >= df->getQ()) break;
      PartitionSegment segment = segments[qSrc];
      real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                segment.m_src.y, segment.m_src.z, true);
      real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                  segment.m_dst.y, segment.m_dst.z, true);
      CUDA_RT_CALL(cudaMemcpy2DAsync(
          ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
          segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
          dfStream));
    }
  }

  inline cudaStream_t getP2Pstream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->streams.at(dstDev);
  }
  inline int getDeviceFromPartition(Partition partition) {
    return m_partitionDeviceMap[partition];
  }
  inline Partition getPartitionFromDevice(int devId) {
    return m_devicePartitionMap.at(devId);
  }
  bool enablePeerAccess(int srcDev, int dstDev,
                        std::vector<bool> *peerAccessList);
  void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList);

  DistributedLattice(int numDevices, int nx, int ny, int nz);

  // virtual void initPartition(int srcDev, Partition partition) = 0;
  // virtual void computePartition(int srcDev, Partition partition) = 0;
};