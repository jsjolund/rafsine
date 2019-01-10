#pragma once

#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "DistributionFunction.hpp"
#include "PartitionTopology.hpp"

bool enablePeerAccess(int srcDev, int dstDev,
                      std::vector<bool> *peerAccessList);
void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList);

class DistributedLattice {
 protected:
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
  void haloExchange(Partition partition, DistributionFunction *df,
                    Partition neighbour, DistributionFunction *ndf,
                    UnitVector::Enum direction, cudaStream_t stream);

  inline cudaStream_t getP2Pstream(int srcDev, int dstDev) {
    return m_deviceParams.at(srcDev)->streams.at(dstDev);
  }
  inline int getDeviceFromPartition(Partition partition) {
    return m_partitionDeviceMap[partition];
  }
  inline Partition getPartitionFromDevice(int devId) {
    return m_devicePartitionMap.at(devId);
  }

  DistributedLattice(int numDevices, int nx, int ny, int nz);
  ~DistributedLattice();
};