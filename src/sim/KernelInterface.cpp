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

void KernelInterface::runComputeKernel(Partition partition, ComputeParams *kp,
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

void KernelInterface::compute(real *plotGpuPointer,
                              DisplayQuantity::Enum displayQuantity) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));

    // LBM
    ComputeParams *params = m_params.at(srcDev);
    Partition partition = getPartitionFromDevice(srcDev);

    runComputeKernel(partition, params, plotGpuPointer, displayQuantity);
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    {
      Partition neighbour =
          params->df_tmp->getNeighbour(partition, UnitVector::X_AXIS_POS);
      int dstDev = getDeviceFromPartition(neighbour);
      haloExchange(partition, params->df_tmp, neighbour,
                   m_params.at(dstDev)->df_tmp, UnitVector::X_AXIS_POS,
                   getP2Pstream(srcDev, dstDev));
      haloExchange(partition, params->dfT_tmp, neighbour,
                   m_params.at(dstDev)->dfT_tmp, UnitVector::X_AXIS_POS,
                   getP2Pstream(srcDev, dstDev));
    }
    {
      Partition neighbour =
          params->df_tmp->getNeighbour(partition, UnitVector::X_AXIS_NEG);
      int dstDev = getDeviceFromPartition(neighbour);
      haloExchange(partition, params->df_tmp, neighbour,
                   m_params.at(dstDev)->df_tmp, UnitVector::X_AXIS_NEG,
                   getP2Pstream(srcDev, dstDev));
      haloExchange(partition, params->dfT_tmp, neighbour,
                   m_params.at(dstDev)->dfT_tmp, UnitVector::X_AXIS_NEG,
                   getP2Pstream(srcDev, dstDev));
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    {
      Partition neighbour =
          params->df_tmp->getNeighbour(partition, UnitVector::Y_AXIS_POS);
      int dstDev = getDeviceFromPartition(neighbour);
      haloExchange(partition, params->df_tmp, neighbour,
                   m_params.at(dstDev)->df_tmp, UnitVector::Y_AXIS_POS,
                   getP2Pstream(srcDev, dstDev));
      haloExchange(partition, params->dfT_tmp, neighbour,
                   m_params.at(dstDev)->dfT_tmp, UnitVector::Y_AXIS_POS,
                   getP2Pstream(srcDev, dstDev));
    }
    {
      Partition neighbour =
          params->df_tmp->getNeighbour(partition, UnitVector::Y_AXIS_NEG);
      int dstDev = getDeviceFromPartition(neighbour);
      haloExchange(partition, params->df_tmp, neighbour,
                   m_params.at(dstDev)->df_tmp, UnitVector::Y_AXIS_NEG,
                   getP2Pstream(srcDev, dstDev));
      haloExchange(partition, params->dfT_tmp, neighbour,
                   m_params.at(dstDev)->dfT_tmp, UnitVector::Y_AXIS_NEG,
                   getP2Pstream(srcDev, dstDev));
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    {
      Partition neighbour =
          params->df_tmp->getNeighbour(partition, UnitVector::Z_AXIS_POS);
      int dstDev = getDeviceFromPartition(neighbour);
      haloExchange(partition, params->df_tmp, neighbour,
                   m_params.at(dstDev)->df_tmp, UnitVector::Z_AXIS_POS,
                   getP2Pstream(srcDev, dstDev));
      haloExchange(partition, params->dfT_tmp, neighbour,
                   m_params.at(dstDev)->dfT_tmp, UnitVector::Z_AXIS_POS,
                   getP2Pstream(srcDev, dstDev));
    }
    {
      Partition neighbour =
          params->df_tmp->getNeighbour(partition, UnitVector::Z_AXIS_NEG);
      int dstDev = getDeviceFromPartition(neighbour);
      haloExchange(partition, params->df_tmp, neighbour,
                   m_params.at(dstDev)->df_tmp, UnitVector::Z_AXIS_NEG,
                   getP2Pstream(srcDev, dstDev));
      haloExchange(partition, params->dfT_tmp, neighbour,
                   m_params.at(dstDev)->dfT_tmp, UnitVector::Z_AXIS_NEG,
                   getP2Pstream(srcDev, dstDev));
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    DistributionFunction::swap(params->df, params->df_tmp);
    DistributionFunction::swap(params->dfT, params->dfT_tmp);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));
}

KernelInterface::KernelInterface(const ComputeParams *params,
                                 const BoundaryConditionsArray *bcs,
                                 const VoxelArray *voxels,
                                 const int numDevices = 1)
    : DistributedLattice(numDevices, params->nx, params->ny, params->nz),
      m_params(numDevices) {
  glm::ivec3 n = glm::ivec3(params->nx, params->ny, params->nz);
  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeParams *kp = new ComputeParams();
    m_params.at(srcDev) = kp;
    *kp = *params;

    // Initialize distribution functions for temperature, velocity and tmps
    kp->df = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    kp->df_tmp = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    kp->dfT = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);
    kp->dfT_tmp = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);
    // Data for averaging
    // 0 -> temperature
    // 1 -> x-component of velocity
    // 2 -> y-component of velocity
    // 3 -> z-component of velocity
    kp->avg = new DistributionFunction(4, n.x, n.y, n.z, m_numDevices);

    const Partition partition = getPartitionFromDevice(srcDev);

    kp->allocate(partition);
    runInitKernel(kp->df, kp->dfT, partition, 1.0, 0, 0, 0, kp->Tinit);
    runInitKernel(kp->df_tmp, kp->dfT_tmp, partition, 1.0, 0, 0, 0, kp->Tinit);
    ss << "Allocated partition " << partition << " on GPU" << srcDev
       << std::endl;

    for (int q = 0; q < 4; q++) kp->avg->fill(q, 0);
    kp->avg->upload();

    kp->voxels = new VoxelArray(*voxels);
    kp->voxels->upload();
    kp->bcs = new device_vector<BoundaryCondition>(*bcs);

    std::cout << ss.str();
  }  // end omp parallel num_threads(numDevices)
  std::cout << "GPU configuration complete" << std::endl;
}

void KernelInterface::uploadBCs(BoundaryConditionsArray *bcs) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    ComputeParams *kp = m_params.at(srcDev);
    *kp->bcs = *bcs;
  }
}

void KernelInterface::resetAverages() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    ComputeParams *kp = m_params.at(srcDev);
    for (int q = 0; q < 4; q++) kp->avg->fill(q, 0);
    kp->avg->upload();
  }
}

void KernelInterface::resetDfs() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    const Partition partition = m_devicePartitionMap.at(srcDev);
    ComputeParams *kp = m_params.at(srcDev);
    runInitKernel(kp->df, kp->dfT, partition, 1.0, 0, 0, 0, kp->Tinit);
    runInitKernel(kp->df_tmp, kp->dfT_tmp, partition, 1.0, 0, 0, 0, kp->Tinit);
  }
}

KernelInterface::~KernelInterface() {}