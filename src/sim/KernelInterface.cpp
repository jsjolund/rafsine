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
                                       DisplayQuantity::Enum displayQuantity,
                                       cudaStream_t stream) {
  glm::ivec3 latticeSize = getLatticeDims();
  glm::ivec3 partMin = subLattice.getLatticeMin();
  glm::ivec3 partMax = subLattice.getLatticeMax();
  glm::ivec3 partHalo = subLattice.getHalo();
  glm::ivec3 n = subLattice.getLatticeDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);

  real *dfPtr = kp->df->gpu_ptr(subLattice);
  real *df_tmpPtr = kp->df_tmp->gpu_ptr(subLattice);
  real *dfTPtr = kp->dfT->gpu_ptr(subLattice);
  real *dfT_tmpPtr = kp->dfT_tmp->gpu_ptr(subLattice);

  SubLattice subLatticeNoHalo(subLattice.getLatticeMin(),
                              subLattice.getLatticeMax(), glm::ivec3(0, 0, 0));
  real *avgPtr = kp->avg->gpu_ptr(subLatticeNoHalo);
  real *plotPtr = kp->plot->gpu_ptr(subLatticeNoHalo);

  voxel *voxelPtr = kp->voxels->gpu_ptr();

  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*kp->bcs)[0]);

  ComputeKernel<<<gridSize, blockSize, 0, stream>>>(
      dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, plotPtr, voxelPtr, partMin, partMax,
      partHalo, latticeSize, kp->nu, kp->C, kp->nuT, kp->Pr_t, kp->gBetta,
      kp->Tref, displayQuantity, avgPtr, bcsPtr);

  CUDA_CHECK_ERRORS("ComputeKernel");
}

void KernelInterface::exchange(int srcDev, SubLattice subLattice,
                               D3Q7::Enum direction) {
  ComputeParams *params = m_params.at(srcDev);
  SubLattice neighbour = params->df_tmp->getNeighbour(subLattice, direction);
  int dstDev = getSubLatticeDevice(neighbour);
  params->df_tmp->exchange(subLattice, m_params.at(dstDev)->df_tmp, neighbour,
                           direction, getP2Pstream(srcDev, dstDev));
  params->dfT_tmp->exchange(subLattice, m_params.at(dstDev)->dfT_tmp, neighbour,
                            direction, getP2Pstream(srcDev, dstDev));
}

void KernelInterface::compute(DisplayQuantity::Enum displayQuantity) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));

    // LBM
    ComputeParams *params = m_params.at(srcDev);
    SubLattice subLattice = getDeviceSubLattice(srcDev);

    runComputeKernel(subLattice, params, displayQuantity);
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
  const glm::ivec3 n = glm::ivec3(params->nx, params->ny, params->nz);

  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // For gathering distributed plot onto GPU0
  m_plot = new DistributionArray(1, n.x, n.y, n.z);
  const SubLattice fullLattice = m_plot->getSubLattice(0, 0, 0);
  m_plot->allocate(fullLattice);
  m_plot->fill(0, 0);

  // For gathering averages onto GPU0
  m_avg = new DistributionArray(4, n.x, n.y, n.z);
  m_avg->allocate(fullLattice);
  for (int q = 0; q < 4; q++) m_avg->fill(q, 0);

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
    const SubLattice subLattice = getDeviceSubLattice(srcDev);

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

    // Create arrays for averaging and plotting
    const SubLattice subLatticeNoHalo(subLattice.getLatticeMin(),
                                      subLattice.getLatticeMax(),
                                      glm::ivec3(0, 0, 0));
    if (m_numDevices == 1) {
      kp->avg = m_avg;
      kp->plot = m_plot;
    } else {
      kp->avg = new DistributionArray(4, n.x, n.y, n.z, m_numDevices);
      kp->avg->allocate(subLatticeNoHalo);
      for (int q = 0; q < 4; q++) kp->avg->fill(q, 0);

      kp->plot = new DistributionArray(1, n.x, n.y, n.z, m_numDevices);
      kp->plot->allocate(subLatticeNoHalo);
      kp->plot->fill(0, 0);
    }
    // Upload voxels and boundary conditions
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
  }
}

void KernelInterface::resetDfs() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    const SubLattice subLattice = getDeviceSubLattice(srcDev);
    ComputeParams *kp = m_params.at(srcDev);
    runInitKernel(kp->df, kp->dfT, subLattice, 1.0, 0, 0, 0, kp->Tinit);
    runInitKernel(kp->df_tmp, kp->dfT_tmp, subLattice, 1.0, 0, 0, 0, kp->Tinit);
  }
}

real *KernelInterface::plot(DistributionArray *plot) {
  // Gather the partitions into the plot array
#pragma omp parallel num_threads(CUDA_MAX_P2P_DEVS)
#pragma omp for
  for (int srcDev = 0; srcDev < m_numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    ComputeParams *kp = m_params.at(srcDev);
    SubLattice subLattice = kp->plot->getDeviceSubLattice(srcDev);

    std::vector<bool> p2pList = getP2PConnections(srcDev);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->gather(subLattice, newFullArray);
    disablePeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
}

KernelInterface::~KernelInterface() {}
