#include "KernelInterface.hpp"

void KernelInterface::runInitKernel(DistributionFunction *df,
                                    DistributionFunction *dfT,
                                    Partition partition, float rho, float vx,
                                    float vy, float vz, float T) {
  /// Initialise distribution functions on the GPU
  float sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
  glm::ivec3 n = partition.getQDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);
  real *dfPtr = df->gpu_ptr(partition);
  real *dfTPtr = dfT->gpu_ptr(partition);
  InitKernel<<<gridSize, blockSize>>>(dfPtr, dfTPtr, n.x, n.y, n.z, rho, vx, vy,
                                      vz, T, sq_term);

  CUDA_CHECK_ERRORS("InitKernel");
}

void KernelInterface::runComputeKernel(Partition partition,
                                       ComputeKernelParams *kp,
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

void runHaloExchangeKernel(HaloExchangeParams *hp, cudaStream_t stream) {
  int **srcIdxPtrs = thrust::raw_pointer_cast(&(hp->srcIdxPtrs)[0]);
  real **dstDfPtrs = thrust::raw_pointer_cast(&(hp->dstDfPtrs)[0]);
  int **dstIdxPtrs = thrust::raw_pointer_cast(&(hp->dstIdxPtrs)[0]);
  int *dstQStrides = thrust::raw_pointer_cast(&(hp->dstQStrides)[0]);
  int *idxLengths = thrust::raw_pointer_cast(&(hp->idxLengths)[0]);

  dim3 gridSize(hp->maxHaloSize, hp->nq, 1);
  dim3 blockSize(hp->nq, 1, 1);

  int cudaDev;
  CUDA_RT_CALL(cudaGetDevice(&cudaDev));

  HaloExchangeKernel<<<gridSize, blockSize, 0, stream>>>(
      hp->nq, hp->srcDfPtr, srcIdxPtrs, hp->srcQStride, dstDfPtrs, dstIdxPtrs,
      dstQStrides, idxLengths, cudaDev == 1 && hp->nq == 19);

  CUDA_CHECK_ERRORS("HaloExchangeKernel");
}

void buildHaloExchangeParams(HaloExchangeParams *hp, DistributionFunction *df,
                             std::vector<DistributionFunction *> *neighbourDfs,
                             Partition partition) {
  hp->nq = df->getQ();
  hp->srcDfPtr = df->gpu_ptr(partition);
  hp->srcQStride = partition.getQStride();

  for (int q = 0; q < df->getQ(); q++) {
    Partition neighbour = df->getNeighbour(partition, q);
    DistributionFunction *dstDf = neighbourDfs->at(q);
    hp->dstDfPtrs[q] = dstDf->gpu_ptr(neighbour);
    hp->dstQStrides[q] = neighbour.getQStride();
    HaloExchangeData *haloData = df->m_haloData[partition][neighbour];

    if (haloData->srcIndexH.size() != haloData->srcIndexD.size())
      haloData->srcIndexD = thrust::device_vector<int>(haloData->srcIndexH);
    if (haloData->dstIndexH.size() != haloData->dstIndexD.size())
      haloData->dstIndexD = thrust::device_vector<int>(haloData->dstIndexH);

    hp->srcIdxPtrs[q] = thrust::raw_pointer_cast(&(haloData->srcIndexD)[0]);
    hp->dstIdxPtrs[q] = thrust::raw_pointer_cast(&(haloData->dstIndexD)[0]);

    int haloSize = haloData->srcIndexH.size();
    assert(haloSize == haloData->srcIndexD.size() &&
           haloSize == haloData->dstIndexD.size() &&
           haloSize == haloData->dstIndexH.size());
    hp->idxLengths[q] = haloSize;
    if (haloSize > hp->maxHaloSize) hp->maxHaloSize = haloSize;
  }
}

void KernelInterface::compute(real *plotGpuPointer,
                              DisplayQuantity::Enum displayQuantity) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeKernelParams *kp = m_computeParams.at(srcDev);

    Partition partition = m_devicePartitionMap.at(srcDev);

    cudaStream_t computeStream = m_deviceParams.at(srcDev)->computeStream;

    runComputeKernel(partition, kp, plotGpuPointer, displayQuantity,
                     computeStream);

    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));

    // Halo exchange, loop over each partition on this GPU
    cudaStream_t dfStream = m_deviceParams.at(srcDev)->dfExchangeStream;
    cudaStream_t dfTStream = m_deviceParams.at(srcDev)->dfTExchangeStream;
    runHaloExchangeKernel(m_dfHaloParams.at(srcDev), dfStream);
    runHaloExchangeKernel(m_dfTHaloParams.at(srcDev), dfTStream);

    CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
    CUDA_RT_CALL(cudaStreamSynchronize(dfTStream));

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

KernelInterface::KernelInterface(const ComputeKernelParams *params,
                                 const BoundaryConditionsArray *bcs,
                                 const VoxelArray *voxels,
                                 const int numDevices = 1)
    : m_numDevices(numDevices),
      m_devicePartitionMap(numDevices),
      m_computeParams(numDevices),
      m_dfHaloParams(numDevices),
      m_dfTHaloParams(numDevices),
      m_deviceParams(numDevices) {
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
      m_devicePartitionMap.at(devIndex) = Partition(*partition);
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

    ComputeKernelParams *kp = m_computeParams.at(srcDev) =
        new ComputeKernelParams();
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

    const Partition partition = m_devicePartitionMap.at(srcDev);

    kp->allocate(partition);
    runInitKernel(kp->df, kp->dfT, partition, 1.0, 0, 0, 0, kp->Tinit);
    runInitKernel(kp->df_tmp, kp->dfT_tmp, partition, 1.0, 0, 0, 0, kp->Tinit);
    ss << "Allocated partition " << partition << " on GPU " << srcDev
       << std::endl;

    kp->init(voxels, bcs);

    DeviceParams *dp = m_deviceParams.at(srcDev) = new DeviceParams(numDevices);

    // Enable P2P access between GPUs
    std::unordered_map<Partition, HaloExchangeData *> haloDatas =
        kp->df->m_haloData[partition];
    for (std::pair<Partition, HaloExchangeData *> element : haloDatas) {
      const int dstDev = m_partitionDeviceMap[element.first];
      if (enablePeerAccess(srcDev, dstDev, &dp->peerAccessList)) {
        ss << "Enabled P2P from GPU " << srcDev << " to GPU" << dstDev
           << std::endl;
      }
    }
    // All GPUs need access to the rendering GPU 0
    if (enablePeerAccess(srcDev, 0, &dp->peerAccessList))
      ss << "Enabled P2P from GPU " << srcDev << " to GPU0" << std::endl;

    // Create non-blocking streams
    CUDA_RT_CALL(
        cudaStreamCreateWithFlags(&dp->computeStream, cudaStreamNonBlocking));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&dp->dfExchangeStream,
                                           cudaStreamNonBlocking));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&dp->dfTExchangeStream,
                                           cudaStreamNonBlocking));

// Wait until all threads have finished constructing computeParams
#pragma omp barrier

    {
      int nq = kp->df_tmp->getQ();
      HaloExchangeParams *hp = new HaloExchangeParams(nq);
      m_dfHaloParams.at(srcDev) = hp;
      std::vector<DistributionFunction *> neighbourDf_tmps(nq);
      for (int q = 0; q < nq; q++) {
        Partition neighbour = kp->df_tmp->getNeighbour(partition, q);
        const int dstDev = m_partitionDeviceMap[neighbour];
        neighbourDf_tmps.at(q) = m_computeParams.at(dstDev)->df_tmp;
      }
      buildHaloExchangeParams(hp, kp->df_tmp, &neighbourDf_tmps, partition);
    }
    {
      int nq = kp->dfT_tmp->getQ();
      HaloExchangeParams *hp = new HaloExchangeParams(nq);
      m_dfTHaloParams.at(srcDev) = hp;
      std::vector<DistributionFunction *> neighbourDfT_tmps(nq);
      for (int q = 0; q < nq; q++) {
        Partition neighbour = kp->dfT_tmp->getNeighbour(partition, q);
        const int dstDev = m_partitionDeviceMap[neighbour];
        neighbourDfT_tmps.at(q) = m_computeParams.at(dstDev)->dfT_tmp;
      }
      buildHaloExchangeParams(hp, kp->dfT_tmp, &neighbourDfT_tmps, partition);
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());
    std::cout << ss.str();
  }
  std::cout << "GPU configuration complete" << std::endl;
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

void KernelInterface::uploadBCs(BoundaryConditionsArray *bcs) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    ComputeKernelParams *kp = m_computeParams.at(srcDev);
    *kp->bcs = *bcs;
  }
}

void KernelInterface::resetAverages() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    ComputeKernelParams *kp = m_computeParams.at(srcDev);
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

    DeviceParams *dp = m_deviceParams.at(srcDev);
    disablePeerAccess(srcDev, &dp->peerAccessList);
    CUDA_RT_CALL(cudaStreamDestroy(dp->computeStream));
    CUDA_RT_CALL(cudaStreamDestroy(dp->dfExchangeStream));
    CUDA_RT_CALL(cudaStreamDestroy(dp->dfTExchangeStream));

    delete dp;
  }
}