#include "DistributedLattice.hpp"

bool enablePeerAccess(int srcDev, int dstDev,
                      std::vector<bool> *peerAccessList) {
  std::ostringstream ss;
  if (srcDev == dstDev || peerAccessList->at(dstDev)) {
    peerAccessList->at(srcDev) = true;
    return false;
  } else if (!peerAccessList->at(dstDev)) {
    int cudaCanAccessPeer = 0;
    CUDA_RT_CALL(cudaDeviceCanAccessPeer(&cudaCanAccessPeer, srcDev, dstDev));
    if (cudaCanAccessPeer) {
      CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
      peerAccessList->at(dstDev) = true;
      ss << "Enabled P2P from GPU" << srcDev << " to GPU" << dstDev
         << std::endl;
    } else {
      ss << "ERROR: Failed to enable P2P from GPU" << srcDev << " to GPU"
         << dstDev << std::endl;
      throw std::runtime_error(ss.str());
    }
  }
  std::cout << ss.str();
  return peerAccessList->at(dstDev);
}

void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList) {
  std::ostringstream ss;
  for (int dstDev = 0; dstDev < peerAccessList->size(); dstDev++) {
    if (dstDev != srcDev && peerAccessList->at(dstDev)) {
      CUDA_RT_CALL(cudaDeviceDisablePeerAccess(dstDev));
      peerAccessList->at(dstDev) = false;
      ss << "Disabled P2P from GPU" << srcDev << " to GPU" << dstDev
         << std::endl;
    }
  }
  std::cout << ss.str();
}

void DistributedLattice::haloExchange(Partition partition,
                                      DistributionFunction *df,
                                      Partition neighbour,
                                      DistributionFunction *ndf,
                                      UnitVector::Enum direction,
                                      cudaStream_t stream) {
  PartitionSegment segment = df->m_segments[partition][neighbour].at(direction);

  for (int q : D3Q27ranks[direction]) {
    if (q >= df->getQ()) break;
    real *dfPtr = df->gpu_ptr(partition, q, segment.m_src.x, segment.m_src.y,
                              segment.m_src.z, true);
    real *ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x, segment.m_dst.y,
                                segment.m_dst.z, true);
    CUDA_RT_CALL(cudaMemcpy2DAsync(ndfPtr, segment.m_dstStride, dfPtr,
                                   segment.m_srcStride, segment.m_segmentLength,
                                   segment.m_numSegments, cudaMemcpyDefault,
                                   stream));
  }
}

DistributedLattice::DistributedLattice(int numDevices, int nx, int ny, int nz)
    : m_numDevices(numDevices),
      m_deviceParams(numDevices),
      m_devicePartitionMap(numDevices) {
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  std::cout << "Lattice size: (" << nx << ", " << ny << ", " << nz << ")"
            << std::endl
            << "Total number of sites: " << nx * ny * nz << std::endl
            << "Number of devices: " << m_numDevices << std::endl;

  Topology df(1, nx, ny, nz, m_numDevices);
  std::vector<Partition> partitions = df.getPartitions();

  for (int i = 0; i < partitions.size(); i++) {
    Partition partition = partitions.at(i);
    // Distribute the workload. Calculate partitions and assign them to GPUs
    int devIndex = i % m_numDevices;
    m_partitionDeviceMap[partition] = devIndex;
    m_devicePartitionMap.at(devIndex) = Partition(partition);
  }

  std::cout << "Starting GPU threads" << std::endl;

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    const Partition partition = m_devicePartitionMap.at(srcDev);

    DeviceParams *dp = new DeviceParams(numDevices);
    m_deviceParams.at(srcDev) = dp;

    // Enable P2P access between GPUs
    for (int nIdx = 0; nIdx < 27; nIdx++) {
      Partition neighbour = df.getNeighbour(partition, D3Q27[nIdx]);
      const int dstDev = m_partitionDeviceMap[neighbour];
      enablePeerAccess(srcDev, dstDev, &dp->peerAccessList);
      cudaStream_t *dstStream = &dp->streams.at(dstDev);
      if (*dstStream == 0)
        CUDA_RT_CALL(
            cudaStreamCreateWithFlags(dstStream, cudaStreamNonBlocking));
    }
    // All GPUs need access to the rendering GPU0
    enablePeerAccess(srcDev, 0, &dp->peerAccessList);

  }  // end omp parallel num_threads(numDevices)
  std::cout << "GPU configuration complete" << std::endl;
}

DistributedLattice::~DistributedLattice() {
  std::cout << "Deleting distributed lattice" << std::endl;
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    DeviceParams *dp = m_deviceParams.at(srcDev);
    disablePeerAccess(srcDev, &dp->peerAccessList);
    for (int i = 0; i < dp->streams.size(); i++) {
      if (dp->streams.at(i)) CUDA_RT_CALL(cudaStreamDestroy(dp->streams.at(i)));
    }
    delete dp;
  }
}
