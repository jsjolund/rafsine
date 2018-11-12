#include "KernelData.hpp"

void KernelData::initDomain(DistributedDFGroup *df, DistributedDFGroup *dfT,
                            float rho, float vx, float vy, float vz, float T) {
  /// Initialise distribution functions on the CPU
  float sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
#pragma omp parallel for
  for (int i = 0; i < df->getLatticeDims().x; ++i)
    for (int j = 0; j < df->getLatticeDims().y; ++j)
      for (int k = 0; k < df->getLatticeDims().z; ++k) {
        (*df)(0, i, j, k) = rho * (1.f / 3.f) * (1 + sq_term);
        (*df)(1, i, j, k) =
            rho * (1.f / 18.f) * (1 + 3.f * vx + 4.5f * vx * vx + sq_term);
        (*df)(2, i, j, k) =
            rho * (1.f / 18.f) * (1 - 3.f * vx + 4.5f * vx * vx + sq_term);
        (*df)(3, i, j, k) =
            rho * (1.f / 18.f) * (1 + 3.f * vy + 4.5f * vy * vy + sq_term);
        (*df)(4, i, j, k) =
            rho * (1.f / 18.f) * (1 - 3.f * vy + 4.5f * vy * vy + sq_term);
        (*df)(5, i, j, k) =
            rho * (1.f / 18.f) * (1 + 3.f * vz + 4.5f * vz * vz + sq_term);
        (*df)(6, i, j, k) =
            rho * (1.f / 18.f) * (1 - 3.f * vz + 4.5f * vz * vz + sq_term);
        (*df)(7, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
        (*df)(8, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
        (*df)(9, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
        (*df)(10, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
        (*df)(11, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
        (*df)(12, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
        (*df)(13, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
        (*df)(14, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
        (*df)(15, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
        (*df)(16, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
        (*df)(17, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);
        (*df)(18, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);

        (*dfT)(0, i, j, k) = T * (1.f / 7.f) * (1);
        (*dfT)(1, i, j, k) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vx);
        (*dfT)(2, i, j, k) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vx);
        (*dfT)(3, i, j, k) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vy);
        (*dfT)(4, i, j, k) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vy);
        (*dfT)(5, i, j, k) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vz);
        (*dfT)(6, i, j, k) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vz);
      }
}

void KernelData::compute(real *plotGpuPointer,
                         DisplayQuantity::Enum displayQuantity) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));

    KernelParameters kp = m_params.at(srcDev);

    cudaStream_t computeStream = kp.streams.at(srcDev);

    for (Partition partition : m_devicePartitionMap.at(srcDev)) {
      glm::ivec3 n = partition.getLatticeDims();
      dim3 gridSize(n.y + 2, n.z + 2, 1);
      dim3 blockSize(n.x + 2, 1, 1);

      glm::ivec3 p = partition.getLatticeMin();
      real *dfPtr = kp.df->gpu_ptr(partition, 0, p.x, p.y, p.z);
      real *df_tmpPtr = kp.df_tmp->gpu_ptr(partition, 0, p.x, p.y, p.z);
      real *dfTPtr = kp.dfT->gpu_ptr(partition, 0, p.x, p.y, p.z);
      real *dfT_tmpPtr = kp.dfT_tmp->gpu_ptr(partition, 0, p.x, p.y, p.z);
      real *averagePtr = kp.average->gpu_ptr(partition, 0, p.x, p.y, p.z);

      int *voxelPtr = kp.voxels->gpu_ptr();
      glm::ivec3 min = partition.getLatticeMin();
      glm::ivec3 max = partition.getLatticeMax();
      glm::ivec3 size = kp.df->getLatticeDims();
      BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*kp.bcs)[0]);

      ComputeKernel<<<gridSize, blockSize, 0, computeStream>>>(
          dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, plotGpuPointer, voxelPtr, min,
          max, size, kp.nu, kp.C, kp.nuT, kp.Pr_t, kp.gBetta, kp.Tref,
          displayQuantity, averagePtr, bcsPtr);

      CUDA_RT_CALL(cudaStreamSynchronize(computeStream));
      CUDA_CHECK_ERRORS("ComputeKernel");
    }
    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));

    // for (Partition partition : m_devicePartitionMap.at(srcDev)) {
    //   std::vector<HaloExchangeData> haloDatas =
    //       kp.df_tmp->m_haloData[partition];
    //   for (int i = 0; i < haloDatas.size(); i++) {
    //     HaloExchangeData haloData = haloDatas.at(i);
    //     const int dstDev = m_partitionDeviceMap[*haloData.neighbour];
    //     cudaStream_t cpyStream = kp.streams[dstDev];
    //     DistributedDFGroup *dstDf = m_params.at(dstDev).df_tmp;
    //     kp.df->pushHalo(srcDev, partition, dstDev, dstDf, haloData,
    //     cpyStream);
    //   }
    // }
    // for (int i = 0; i < m_numDevices; i++)
    //   CUDA_RT_CALL(cudaStreamSynchronize(kp.streams[i]));
#pragma omp barrier
    CUDA_RT_CALL(cudaDeviceSynchronize());

    DistributedDFGroup::swap(kp.df, kp.df_tmp);
    DistributedDFGroup::swap(kp.dfT, kp.dfT_tmp);
  }
}

KernelData::~KernelData() {
  for (KernelParameters kp : m_params)
    delete kp.df, kp.df_tmp, kp.dfT, kp.dfT_tmp, kp.average, kp.voxels, kp.bcs;
}

KernelData::KernelData(const KernelParameters *params,
                       const BoundaryConditionsArray *bcs,
                       const VoxelArray *voxels, const int numDevices = 1)
    : m_numDevices(numDevices),
      m_devicePartitionMap(numDevices),
      m_params(numDevices) {
  glm::ivec3 n = glm::ivec3(params->nx, params->ny, params->nz);
  CUDA_RT_CALL(cudaSetDevice(0));

  std::cout << "Domain size : (" << n.x << ", " << n.y << ", " << n.z << ")"
            << std::endl
            << "Total number of nodes : " << n.x * n.y * n.z << std::endl;

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
  }
  initDomain(&df, &dfT, 1.0, 0, 0, 0, params->Tinit);

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));

    KernelParameters *kp = &m_params.at(srcDev);
    *kp = *params;

    // Setup streams
    int priorityHigh, priorityLow;
    CUDA_RT_CALL(cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));
    kp->streams = std::vector<cudaStream_t>(numDevices);
    CUDA_RT_CALL(cudaStreamCreateWithPriority(
        &kp->streams.at(srcDev), cudaStreamNonBlocking, priorityHigh));
    for (int i = 0; i < numDevices; i++) {
      if (i != srcDev)
        CUDA_RT_CALL(cudaStreamCreateWithPriority(
            &kp->streams.at(i), cudaStreamNonBlocking, priorityLow));
    }

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
    }

    *kp->df = df;
    *kp->df_tmp = df;
    *kp->dfT = dfT;
    *kp->dfT_tmp = dfT;

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
      std::vector<HaloExchangeData> haloDatas = kp->df->m_haloData[partition];
      for (int i = 0; i < haloDatas.size(); i++) {
        HaloExchangeData haloData = haloDatas.at(i);

        haloData.srcIndexD = new thrust::device_vector<int>();
        haloData.dstIndexD = new thrust::device_vector<int>();

        *haloData.srcIndexD = *haloData.srcIndexH;
        *haloData.dstIndexD = *haloData.dstIndexH;

        const int dstDev = m_partitionDeviceMap[*haloData.neighbour];
        if (!hasPeerAccess.at(dstDev)) {
          int cudaCanAccessPeer = 0;
          CUDA_RT_CALL(
              cudaDeviceCanAccessPeer(&cudaCanAccessPeer, srcDev, dstDev));
          if (cudaCanAccessPeer) {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
            hasPeerAccess.at(dstDev) = true;
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
      }
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
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
