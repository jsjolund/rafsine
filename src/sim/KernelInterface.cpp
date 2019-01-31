#include "KernelInterface.hpp"

void KernelInterface::runInitKernel(DistributionFunction *df,
                                    DistributionFunction *dfT,
                                    SubLattice subLattice, float rho, float vx,
                                    float vy, float vz, float T) {
  float sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
  glm::ivec3 n = subLattice.getArrayDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);
  real *dfPtr = df->gpu_ptr(subLattice);
  real *dfTPtr = dfT->gpu_ptr(subLattice);

  InitKernel<<<gridSize, blockSize>>>(dfPtr, dfTPtr, n.x, n.y, n.z, rho, vx, vy,
                                      vz, T, sq_term);

  CUDA_CHECK_ERRORS("InitKernel");
}

void KernelInterface::runComputeKernel(SubLattice subLattice, ComputeParams *kp,
                                       real *plotGpuPointer,
                                       DisplayQuantity::Enum displayQuantity,
                                       cudaStream_t stream) {
  glm::ivec3 n = subLattice.getLatticeDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);

  glm::ivec3 p = subLattice.getLatticeMin();
  real *dfPtr = kp->df->gpu_ptr(subLattice);
  real *df_tmpPtr = kp->df_tmp->gpu_ptr(subLattice);
  real *dfTPtr = kp->dfT->gpu_ptr(subLattice);
  real *dfT_tmpPtr = kp->dfT_tmp->gpu_ptr(subLattice);

  SubLattice subLatticeNoHalo(subLattice.getLatticeMin(),
                              subLattice.getLatticeMax(), glm::ivec3(0, 0, 0));
  real *avgPtr = kp->avg->gpu_ptr(subLatticeNoHalo);

  voxel *voxelPtr = kp->voxels->gpu_ptr();
  glm::ivec3 partMin = subLattice.getLatticeMin();
  glm::ivec3 partMax = subLattice.getLatticeMax();
  glm::ivec3 partHalo = subLattice.getHalo();
  glm::ivec3 latticeSize = kp->df->getLatticeDims();
  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*kp->bcs)[0]);

  ComputeKernel<<<gridSize, blockSize, 0, stream>>>(
      dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, plotGpuPointer, voxelPtr, partMin,
      partMax, partHalo, latticeSize, kp->nu, kp->C, kp->nuT, kp->Pr_t,
      kp->gBetta, kp->Tref, displayQuantity, avgPtr, bcsPtr);

  CUDA_CHECK_ERRORS("ComputeKernel");
}

void KernelInterface::exchange(int srcDev, SubLattice subLattice,
                               D3Q7::Enum direction) {
  ComputeParams *params = m_params.at(srcDev);
  SubLattice neighbour = params->df_tmp->getNeighbour(subLattice, direction);
  int dstDev = getDeviceFromSubLattice(neighbour);
  params->df_tmp->exchange(subLattice, m_params.at(dstDev)->df_tmp, neighbour,
                           direction, getP2Pstream(srcDev, dstDev));
  params->dfT_tmp->exchange(subLattice, m_params.at(dstDev)->dfT_tmp, neighbour,
                            direction, getP2Pstream(srcDev, dstDev));
}

void KernelInterface::compute(real *plotGpuPointer,
                              DisplayQuantity::Enum displayQuantity) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));

    // LBM
    ComputeParams *params = m_params.at(srcDev);
    SubLattice subLattice = getSubLatticeFromDevice(srcDev);

    runComputeKernel(subLattice, params, plotGpuPointer, displayQuantity);
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    if (subLattice.getHalo().x > 0) {
      exchange(srcDev, subLattice, D3Q7::X_AXIS_POS);
      exchange(srcDev, subLattice, D3Q7::X_AXIS_NEG);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    if (subLattice.getHalo().y > 0) {
      exchange(srcDev, subLattice, D3Q7::Y_AXIS_POS);
      exchange(srcDev, subLattice, D3Q7::Y_AXIS_NEG);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    if (subLattice.getHalo().z > 0) {
      exchange(srcDev, subLattice, D3Q7::Z_AXIS_POS);
      exchange(srcDev, subLattice, D3Q7::Z_AXIS_NEG);
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
    : P2PLattice(params->nx, params->ny, params->nz, numDevices),
      m_params(numDevices) {
  glm::ivec3 n = glm::ivec3(params->nx, params->ny, params->nz);
  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeParams *kp = new ComputeParams(*params);
    m_params.at(srcDev) = kp;

    // Initialize distribution functions for temperature, velocity and tmps
    const SubLattice subLattice = getSubLatticeFromDevice(srcDev);

    kp->df = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    kp->df_tmp = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    kp->dfT = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);
    kp->dfT_tmp = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);

    kp->df->allocate(subLattice);
    kp->df_tmp->allocate(subLattice);
    kp->dfT->allocate(subLattice);
    kp->dfT_tmp->allocate(subLattice);

    runInitKernel(kp->df, kp->dfT, subLattice, 1.0, 0, 0, 0, kp->Tinit);
    runInitKernel(kp->df_tmp, kp->dfT_tmp, subLattice, 1.0, 0, 0, 0, kp->Tinit);
    ss << "Allocated subLattice " << subLattice << " on GPU" << srcDev
       << std::endl;

    // Data for averaging
    // 0 -> temperature
    // 1 -> x-component of velocity
    // 2 -> y-component of velocity
    // 3 -> z-component of velocity
    const SubLattice subLatticeNoHalo(subLattice.getLatticeMin(),
                                      subLattice.getLatticeMax(),
                                      glm::ivec3(0, 0, 0));
    kp->avg = new DistributionArray(4, n.x, n.y, n.z, m_numDevices);
    kp->avg->allocate(subLatticeNoHalo);
    for (int q = 0; q < 4; q++) kp->avg->fill(q, 0);

    kp->voxels = new VoxelArray(*voxels);
    kp->voxels->upload();
    kp->bcs = new device_vector<BoundaryCondition>(*bcs);

    CUDA_RT_CALL(cudaDeviceSynchronize());
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
    const SubLattice subLattice = m_deviceSubLatticeMap.at(srcDev);
    ComputeParams *kp = m_params.at(srcDev);
    runInitKernel(kp->df, kp->dfT, subLattice, 1.0, 0, 0, 0, kp->Tinit);
    runInitKernel(kp->df_tmp, kp->dfT_tmp, subLattice, 1.0, 0, 0, 0, kp->Tinit);
  }
}

KernelInterface::~KernelInterface() {}
