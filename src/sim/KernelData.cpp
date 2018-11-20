#include "KernelData.hpp"

void KernelData::initDomain(DistributedDFGroup *df, DistributedDFGroup *dfT,
                            Partition partition, float rho, float vx, float vy,
                            float vz, float T) {
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

KernelData::~KernelData() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    KernelParameters kp = m_params.at(srcDev);
    for (int i = 0; i < 27; i++) CUDA_RT_CALL(cudaStreamDestroy(kp.streams[i]));
  }
  for (KernelParameters kp : m_params)
    delete kp.df, kp.df_tmp, kp.dfT, kp.dfT_tmp, kp.average, kp.voxels, kp.bcs;
}

void KernelData::compute(real *plotGpuPointer,
                         DisplayQuantity::Enum displayQuantity) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    KernelParameters kp = m_params.at(srcDev);
    cudaStream_t computeStream = kp.streams[0];

    for (Partition partition : m_devicePartitionMap.at(srcDev)) {
      glm::ivec3 n = partition.getLatticeDims();
      dim3 gridSize(n.y, n.z, 1);
      dim3 blockSize(n.x, 1, 1);

      glm::ivec3 p = partition.getLatticeMin();
      real *dfPtr = kp.df->gpu_ptr(partition);
      real *df_tmpPtr = kp.df_tmp->gpu_ptr(partition);
      real *dfTPtr = kp.dfT->gpu_ptr(partition);
      real *dfT_tmpPtr = kp.dfT_tmp->gpu_ptr(partition);
      real *averagePtr = kp.average->gpu_ptr(partition);

      int *voxelPtr = kp.voxels->gpu_ptr();
      glm::ivec3 partMin = partition.getLatticeMin();
      glm::ivec3 partMax = partition.getLatticeMax();
      glm::ivec3 latticeSize = kp.df->getLatticeDims();
      BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*kp.bcs)[0]);

      ComputeKernel<<<gridSize, blockSize, 0, computeStream>>>(
          dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, plotGpuPointer, voxelPtr,
          partMin, partMax, latticeSize, kp.nu, kp.C, kp.nuT, kp.Pr_t,
          kp.gBetta, kp.Tref, displayQuantity, averagePtr, bcsPtr);
      CUDA_CHECK_ERRORS("ComputeKernel");
    }

    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));

    // Halo exchange, loop over each partition on this GPU
    for (Partition partition : m_devicePartitionMap.at(srcDev)) {
      // Loop over each lattice direction (19 for velocity)
      for (int q = 0; q < kp.df_tmp->getQ(); q++) {
        Partition neighbour = kp.df_tmp->getNeighbour(partition, q);
        const int dstDev = m_partitionDeviceMap[neighbour];
        DistributedDFGroup *dstDf = m_params.at(dstDev).df_tmp;
        kp.df_tmp->pushHaloFull(partition, neighbour, dstDf, kp.streams[q + 1]);
      }
      // Loop over each lattice direction (7 for velocity)
      for (int q = 0; q < kp.dfT_tmp->getQ(); q++) {
        Partition neighbour = kp.dfT_tmp->getNeighbour(partition, q);
        const int dstDev = m_partitionDeviceMap[neighbour];
        DistributedDFGroup *dstDf = m_params.at(dstDev).dfT_tmp;
        kp.dfT_tmp->pushHaloFull(partition, neighbour, dstDf,
                                 kp.streams[q + 1 + kp.df_tmp->getQ()]);
      }
    }
    for (int i = 0; i < 27; i++)
      CUDA_RT_CALL(cudaStreamSynchronize(kp.streams[i]));

    CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier

    DistributedDFGroup::swap(kp.df, kp.df_tmp);
    DistributedDFGroup::swap(kp.dfT, kp.dfT_tmp);

#pragma omp barrier
    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
}

KernelData::KernelData(const KernelParameters *params,
                       const BoundaryConditionsArray *bcs,
                       const VoxelArray *voxels, const int numDevices = 1)
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

  DistributedDFGroup df(19, n.x, n.y, n.z, m_numDevices);
  DistributedDFGroup dfT(7, n.x, n.y, n.z, m_numDevices);

  std::vector<Partition *> partitions = df.getPartitions();
  for (int i = 0; i < partitions.size(); i++) {
    Partition *partition = partitions.at(i);
    // Allocate all partitions
    df.allocate(*partition);
    dfT.allocate(*partition);
    // Distribute the workload. Calculate partitions and assign them to GPUs
    int devIndex = i % m_numDevices;
    m_partitionDeviceMap[*partition] = devIndex;
    m_devicePartitionMap.at(devIndex).push_back(Partition(*partition));
    initDomain(&df, &dfT, *partition, 1.0, 0, 0, 0, params->Tinit);
    // TODO(don't download to CPU...)
    df.download();
    dfT.download();
  }

  std::cout << "Starting GPU threads" << std::endl;

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    KernelParameters *kp = &m_params.at(srcDev);
    *kp = *params;

    // Allocate memory for the velocity distribution functions
    kp->df = new DistributedDFGroup(19, n.x, n.y, n.z, m_numDevices);
    // Allocate memory for the temperature distribution functions
    kp->dfT = new DistributedDFGroup(7, n.x, n.y, n.z, m_numDevices);
    // Allocate memory for the temporary distribution functions
    kp->df_tmp = new DistributedDFGroup(19, n.x, n.y, n.z, m_numDevices);
    // Allocate memory for the temporary temperature distribution function
    kp->dfT_tmp = new DistributedDFGroup(7, n.x, n.y, n.z, m_numDevices);
    // Data for averaging are stored in the same structure
    // 0 -> temperature
    // 1 -> x-component of velocity
    // 2 -> y-component of velocity
    // 3 -> z-component of velocity
    kp->average = new DistributedDFGroup(4, n.x, n.y, n.z, m_numDevices);

    for (Partition partition : m_devicePartitionMap.at(srcDev)) {
      kp->df->allocate(partition);
      kp->df_tmp->allocate(partition);
      kp->dfT->allocate(partition);
      kp->dfT_tmp->allocate(partition);
      kp->average->allocate(partition);
      ss << "Allocated partition " << partition << " on GPU " << srcDev
         << std::endl;
    }

    *kp->df = df;
    *kp->df_tmp = *kp->df;
    *kp->dfT = dfT;
    *kp->dfT_tmp = *kp->dfT;

    kp->df->upload();
    kp->df_tmp->upload();
    kp->dfT->upload();
    kp->dfT_tmp->upload();

    for (int q = 0; q < 4; q++) kp->average->fill(q, 0);
    kp->average->upload();

    kp->voxels = new VoxelArray(*voxels);
    kp->voxels->upload();
    kp->bcs = new thrust::device_vector<BoundaryCondition>(*bcs);

    // Enable P2P access between GPUs
    std::vector<bool> hasPeerAccess(numDevices);
    hasPeerAccess.at(srcDev) = true;
    for (Partition partition : m_devicePartitionMap.at(srcDev)) {
      std::unordered_map<Partition, HaloExchangeData *> haloDatas =
          kp->df->m_haloData[partition];

      for (std::pair<Partition, HaloExchangeData *> element : haloDatas) {
        const int dstDev = m_partitionDeviceMap[element.first];
        if (!hasPeerAccess.at(dstDev)) {
          int cudaCanAccessPeer = 0;
          CUDA_RT_CALL(
              cudaDeviceCanAccessPeer(&cudaCanAccessPeer, srcDev, dstDev));
          if (cudaCanAccessPeer) {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
            hasPeerAccess.at(dstDev) = true;
            ss << "Enabled p2p from GPU " << srcDev << " to GPU" << dstDev
               << std::endl;
          } else {
            ss << "ERROR: Failed to enable p2p from GPU " << srcDev
               << " to GPU " << dstDev << std::endl;
            throw std::runtime_error(ss.str());
          }
        }
      }
    }
    if (srcDev != 0 && !hasPeerAccess.at(0)) {
      // All devices need access to the rendering GPU 0
      int cudaCanAccessPeer = 0;
      CUDA_RT_CALL(cudaDeviceCanAccessPeer(&cudaCanAccessPeer, srcDev, 0));
      if (cudaCanAccessPeer) {
        CUDA_RT_CALL(cudaDeviceEnablePeerAccess(0, 0));
        hasPeerAccess.at(0) = true;
        ss << "Enabled p2p from GPU " << srcDev << " to GPU 0" << std::endl;
      } else {
        ss << "ERROR: Failed to enable p2p from GPU " << srcDev << " to GPU 0"
           << std::endl;
        throw std::runtime_error(ss.str());
      }
    }

    for (int i = 0; i < 27; i++)
      CUDA_RT_CALL(
          cudaStreamCreateWithFlags(&(kp->streams[i]), cudaStreamNonBlocking));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    std::cout << ss.str();
  }
  std::cout << "GPU configuration complete" << std::endl;
}

void KernelData::uploadBCs(BoundaryConditionsArray *bcs) {
  // *m_bcs_d = *bcs;
}

void KernelData::resetAverages() {
  //   m_average->fill(0, 0);
  //   m_average->fill(1, 0);
  //   m_average->fill(2, 0);
  //   m_average->fill(3, 0);
  //   m_average->upload();
}
