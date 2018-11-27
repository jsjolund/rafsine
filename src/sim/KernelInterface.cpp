#include "KernelInterface.hpp"

void KernelInterface::runInitKernel(DistributionFunction *df,
                                    DistributionFunction *dfT,
                                    Partition partition, float rho, float vx,
                                    float vy, float vz, float T) {
  /// Initialise distribution functions on the GPU
  float sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
  glm::ivec3 n = partition.getArrayDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);
  real *dfPtr = df->gpu_ptr(partition);
  real *dfTPtr = dfT->gpu_ptr(partition);
  InitKernel<<<gridSize, blockSize>>>(dfPtr, dfTPtr, n.x, n.y, n.z, rho, vx, vy,
                                      vz, T, sq_term);
  CUDA_CHECK_ERRORS("InitKernel");
}

void KernelInterface::runComputeKernel(Partition partition,
                                       KernelParameters *kp,
                                       real *plotGpuPointer,
                                       DisplayQuantity::Enum displayQuantity,
                                       cudaStream_t computeStream) {
  glm::ivec3 n = partition.getLatticeDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);

  glm::ivec3 p = partition.getLatticeMin();
  real *dfPtr = kp->df->gpu_ptr(partition);
  real *df_tmpPtr = kp->df_tmp->gpu_ptr(partition);
  real *dfTPtr = kp->dfT->gpu_ptr(partition);
  real *dfT_tmpPtr = kp->dfT_tmp->gpu_ptr(partition);
  real *avgPtr = kp->avg->gpu_ptr(partition);

  voxel *voxelPtr = kp->voxels->gpu_ptr();
  glm::ivec3 partMin = partition.getLatticeMin();
  glm::ivec3 partMax = partition.getLatticeMax();
  glm::ivec3 latticeSize = kp->df->getLatticeDims();
  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*kp->bcs)[0]);

  ComputeKernel<<<gridSize, blockSize, 0, computeStream>>>(
      dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, plotGpuPointer, voxelPtr, partMin,
      partMax, latticeSize, kp->nu, kp->C, kp->nuT, kp->Pr_t, kp->gBetta,
      kp->Tref, displayQuantity, avgPtr, bcsPtr);
  CUDA_CHECK_ERRORS("ComputeKernel");
}

void KernelInterface::runHaloExchangeKernel(Partition partition,
                                            KernelParameters *kp) {
  // Loop over each lattice direction (19 for velocity)
  for (int q = 0; q < kp->df_tmp->getQ(); q++) {
    Partition neighbour = kp->df_tmp->getNeighbour(partition, q);
    const int dstDev = m_partitionDeviceMap[neighbour];
    DistributionFunction *dstDf = m_params.at(dstDev)->df_tmp;
    kp->df_tmp->pushHaloFull(partition, neighbour, dstDf, kp->streams[q + 1]);
  }
  // Loop over each lattice direction (7 for velocity)
  for (int q = 0; q < kp->dfT_tmp->getQ(); q++) {
    Partition neighbour = kp->dfT_tmp->getNeighbour(partition, q);
    const int dstDev = m_partitionDeviceMap[neighbour];
    DistributionFunction *dstDf = m_params.at(dstDev)->dfT_tmp;
    kp->dfT_tmp->pushHaloFull(partition, neighbour, dstDf,
                              kp->streams[q + 1 + kp->df_tmp->getQ()]);
  }
}

void KernelInterface::compute(real *plotGpuPointer,
                              DisplayQuantity::Enum displayQuantity) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    KernelParameters *kp = m_params.at(srcDev);
    cudaStream_t computeStream = kp->streams[0];

    for (Partition partition : m_devicePartitionMap.at(srcDev))
      runComputeKernel(partition, kp, plotGpuPointer, displayQuantity,
                       computeStream);

    // Halo exchange, loop over each partition on this GPU
    for (Partition partition : m_devicePartitionMap.at(srcDev))
      runHaloExchangeKernel(partition, kp);

    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));

    for (int i = 0; i < 27; i++)
      CUDA_RT_CALL(cudaStreamSynchronize(kp->streams[i]));

    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    DistributionFunction::swap(kp->df, kp->df_tmp);
    DistributionFunction::swap(kp->dfT, kp->dfT_tmp);

#pragma omp barrier
    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));
}

bool KernelInterface::enablePeerAccess(int srcDev, int dstDev,
                                       std::vector<bool> *peerAccessList) {
  if (srcDev == dstDev) {
    peerAccessList->at(srcDev) = true;
  } else if (!peerAccessList->at(dstDev)) {
    int cudaCanAccessPeer = 0;
    CUDA_RT_CALL(cudaDeviceCanAccessPeer(&cudaCanAccessPeer, srcDev, dstDev));
    if (cudaCanAccessPeer) {
      CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
      peerAccessList->at(dstDev) = true;
    } else {
      std::ostringstream ss;
      ss << "ERROR: Failed to enable P2P from GPU " << srcDev << " to GPU "
         << dstDev << std::endl;
      throw std::runtime_error(ss.str());
    }
  }
  return peerAccessList->at(dstDev);
}

void KernelInterface::disablePeerAccess(int srcDev,
                                        std::vector<bool> *peerAccessList) {
  std::ostringstream ss;
  for (int dstDev = 0; dstDev < peerAccessList->size(); dstDev++) {
    if (dstDev != srcDev && peerAccessList->at(dstDev)) {
      CUDA_RT_CALL(cudaDeviceDisablePeerAccess(dstDev));
      peerAccessList->at(dstDev) = false;
      ss << "Disabled P2P from GPU " << srcDev << " to GPU" << dstDev
         << std::endl;
    }
  }
  std::cout << ss.str();
}

KernelInterface::KernelInterface(const KernelParameters *params,
                                 const BoundaryConditionsArray *bcs,
                                 const VoxelArray *voxels,
                                 const int numDevices = 1)
    : m_numDevices(numDevices),
      m_devicePartitionMap(numDevices),
      m_params(numDevices) {
  glm::ivec3 n = glm::ivec3(params->nx, params->ny, params->nz);
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  std::cout << "Domain size : (" << n.x << ", " << n.y << ", " << n.z << ")"
            << std::endl
            << "Total number of nodes : " << n.x * n.y * n.z << std::endl
            << "Number of devices: " << m_numDevices << std::endl;

  {
    DistributionFunction df(19, n.x, n.y, n.z, m_numDevices);
    std::vector<Partition *> partitions = df.getPartitions();
    for (int i = 0; i < partitions.size(); i++) {
      Partition *partition = partitions.at(i);
      // Distribute the workload. Calculate partitions and assign them to GPUs
      int devIndex = i % m_numDevices;
      m_partitionDeviceMap[*partition] = devIndex;
      m_devicePartitionMap.at(devIndex).push_back(Partition(*partition));
    }
  }

  std::cout << "Starting GPU threads" << std::endl;

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    KernelParameters *kp = m_params.at(srcDev) = new KernelParameters();
    *kp = *params;

    // Allocate memory for the velocity distribution functions
    kp->df = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    // Allocate memory for the temperature distribution functions
    kp->dfT = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);
    // Allocate memory for the temporary distribution functions
    kp->df_tmp = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    // Allocate memory for the temporary temperature distribution function
    kp->dfT_tmp = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);
    // Data for averaging are stored in the same structure
    // 0 -> temperature
    // 1 -> x-component of velocity
    // 2 -> y-component of velocity
    // 3 -> z-component of velocity
    kp->avg = new DistributionFunction(4, n.x, n.y, n.z, m_numDevices);

    for (Partition partition : m_devicePartitionMap.at(srcDev)) {
      kp->allocate(partition);
      runInitKernel(kp->df, kp->dfT, partition, 1.0, 0, 0, 0, kp->Tinit);
      runInitKernel(kp->df_tmp, kp->dfT_tmp, partition, 1.0, 0, 0, 0,
                    kp->Tinit);
      ss << "Allocated partition " << partition << " on GPU " << srcDev
         << std::endl;
    }

    kp->init(voxels, bcs);

    // Enable P2P access between GPUs
    kp->peerAccessList = new std::vector<bool>(numDevices);
    for (Partition partition : m_devicePartitionMap.at(srcDev)) {
      std::unordered_map<Partition, HaloExchangeData *> haloDatas =
          kp->df->m_haloData[partition];
      for (std::pair<Partition, HaloExchangeData *> element : haloDatas) {
        const int dstDev = m_partitionDeviceMap[element.first];
        if (enablePeerAccess(srcDev, dstDev, kp->peerAccessList)) {
          ss << "Enabled P2P from GPU " << srcDev << " to GPU" << dstDev
             << std::endl;
        }
      }
    }
    if (enablePeerAccess(srcDev, 0, kp->peerAccessList))
      ss << "Enabled P2P from GPU " << srcDev << " to GPU 0" << std::endl;

    for (int i = 0; i < 27; i++)
      CUDA_RT_CALL(
          cudaStreamCreateWithFlags(&(kp->streams[i]), cudaStreamNonBlocking));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    std::cout << ss.str();
  }
  std::cout << "GPU configuration complete" << std::endl;
}

void KernelInterface::uploadBCs(BoundaryConditionsArray *bcs) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    KernelParameters *kp = m_params.at(srcDev);
    *kp->bcs = *bcs;
  }
}

void KernelInterface::resetAverages() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    KernelParameters *kp = m_params.at(srcDev);
    for (int q = 0; q < 4; q++) kp->avg->fill(q, 0);
    kp->avg->upload();
  }
}

KernelInterface::~KernelInterface() {
  std::cout << "Deleting kernel interface" << std::endl;
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    KernelParameters *kp = m_params.at(srcDev);
    for (int i = 0; i < 27; i++)
      CUDA_RT_CALL(cudaStreamDestroy(kp->streams[i]));
    disablePeerAccess(srcDev, kp->peerAccessList);
    delete kp;
  }
}