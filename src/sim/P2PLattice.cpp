#include "P2PLattice.hpp"

bool enablePeerAccess(int srcDev, int dstDev, std::vector<bool> *p2pList) {
  if (srcDev == dstDev || p2pList->at(dstDev)) {
    p2pList->at(srcDev) = true;
    return false;
  } else if (!p2pList->at(dstDev)) {
    int cudaCanAccessPeer = 0;
    CUDA_RT_CALL(cudaDeviceCanAccessPeer(&cudaCanAccessPeer, srcDev, dstDev));
    if (cudaCanAccessPeer) {
      CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
      p2pList->at(dstDev) = true;
    } else {
      std::ostringstream ss;
      ss << "ERROR: Failed to enable P2P from GPU" << srcDev << " to GPU"
         << dstDev << std::endl;
      throw std::runtime_error(ss.str());
    }
  }
  return p2pList->at(dstDev);
}

void disableAllPeerAccess(int srcDev, std::vector<bool> *p2pList) {
  for (int dstDev = 0; dstDev < p2pList->size(); dstDev++) {
    if (dstDev != srcDev && p2pList->at(dstDev)) {
      CUDA_RT_CALL(cudaDeviceDisablePeerAccess(dstDev));
      p2pList->at(dstDev) = false;
    }
  }
}

void disablePeerAccess(int srcDev, int dstDev, std::vector<bool> *p2pList) {
  if (dstDev != srcDev && p2pList->at(dstDev)) {
    CUDA_RT_CALL(cudaDeviceDisablePeerAccess(dstDev));
    p2pList->at(dstDev) = false;
  }
}

P2PLattice::P2PLattice(int nx, int ny, int nz, int numDevices)
    : DistributedLattice(nx, ny, nz, numDevices, 1),
      m_deviceParams(numDevices) {
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  std::cout << "Lattice size: (" << nx << ", " << ny << ", " << nz << ")"
            << std::endl
            << "Total number of sites: " << nx * ny * nz << std::endl
            << "Number of devices: " << m_numDevices << std::endl;

  std::cout << "Starting GPU threads" << std::endl;

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    const SubLattice subLattice = m_deviceSubLatticeMap.at(srcDev);

    DeviceParams *dp = new DeviceParams(numDevices);
    m_deviceParams.at(srcDev) = dp;
    dp->p2pList.at(srcDev) = true;

    // Enable P2P access between GPUs
    for (int nIdx = 0; nIdx < 27; nIdx++) {
      SubLattice neighbour = getNeighbour(subLattice, D3Q27[nIdx]);
      const int dstDev = m_subLatticeDeviceMap[neighbour];
      enablePeerAccess(srcDev, dstDev, &dp->p2pList);
      cudaStream_t *dstStream = &dp->streams.at(dstDev);
      if (*dstStream == 0)
        CUDA_RT_CALL(
            cudaStreamCreateWithFlags(dstStream, cudaStreamNonBlocking));
    }
  }  // end omp parallel num_threads(numDevices)
  std::cout << "GPU configuration complete" << std::endl;
}

P2PLattice::~P2PLattice() {
  std::cout << "Deleting distributed lattice" << std::endl;
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    DeviceParams *dp = m_deviceParams.at(srcDev);
    disableAllPeerAccess(srcDev, &dp->p2pList);
    for (int i = 0; i < dp->streams.size(); i++) {
      if (dp->streams.at(i)) CUDA_RT_CALL(cudaStreamDestroy(dp->streams.at(i)));
    }
    delete dp;
  }
}
