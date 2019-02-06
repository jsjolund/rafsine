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

void KernelInterface::runComputeKernel(SubLattice subLattice,
                                       ComputeParams *param,
                                       DisplayQuantity::Enum displayQuantity,
                                       cudaStream_t stream) {
  glm::ivec3 partMin = subLattice.getLatticeMin();
  glm::ivec3 partMax = subLattice.getLatticeMax();
  glm::ivec3 partHalo = subLattice.getHalo();
  glm::ivec3 n = subLattice.getLatticeDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);

  real *dfPtr = param->df->gpu_ptr(subLattice);
  real *df_tmpPtr = param->df_tmp->gpu_ptr(subLattice);
  real *dfTPtr = param->dfT->gpu_ptr(subLattice);
  real *dfT_tmpPtr = param->dfT_tmp->gpu_ptr(subLattice);

  SubLattice subLatticeNoHalo(subLattice.getLatticeMin(),
                              subLattice.getLatticeMax(), glm::ivec3(0, 0, 0));
  real *avgPtr = param->avg->gpu_ptr(subLatticeNoHalo);
  real *plotPtr = param->plot->gpu_ptr(subLatticeNoHalo);

  voxel *voxelPtr = param->voxels->gpu_ptr();

  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*param->bcs)[0]);

  ComputeKernel<<<gridSize, blockSize, 0, stream>>>(
      dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, plotPtr, voxelPtr, partMin, partMax,
      partHalo, param->nu, param->C, param->nuT, param->Pr_t, param->gBetta,
      param->Tref, displayQuantity, avgPtr, bcsPtr);

  CUDA_CHECK_ERRORS("ComputeKernel");
}

void KernelInterface::exchange(int srcDev, SubLattice subLattice,
                               D3Q7::Enum direction) {
  ComputeParams *params = m_params.at(srcDev);
  SubLattice neighbour = params->df_tmp->getNeighbour(subLattice, direction);
  int dstDev = getSubLatticeDevice(neighbour);
  params->df_tmp->exchange(subLattice, m_params.at(dstDev)->df_tmp, neighbour,
                           direction, getP2PStream(srcDev, dstDev));
  params->dfT_tmp->exchange(subLattice, m_params.at(dstDev)->dfT_tmp, neighbour,
                            direction, getP2PStream(srcDev, dstDev));
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
  m_plot = new DistributionArray<real>(1, n.x, n.y, n.z);
  const SubLattice fullLattice = m_plot->getSubLattice(0, 0, 0);
  m_plot->allocate(fullLattice);
  m_plot->fill(0, 0);

  // For gathering averages onto GPU0
  m_avg = new DistributionArray<real>(4, n.x, n.y, n.z);
  m_avg->allocate(fullLattice);
  for (int q = 0; q < 4; q++) m_avg->fill(q, 0);

    // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeParams *param = new ComputeParams(*params);
    m_params.at(srcDev) = param;

    // Initialize distribution functions for temperature, velocity and tmps
    const SubLattice subLattice = getDeviceSubLattice(srcDev);

    param->df = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    param->df_tmp = new DistributionFunction(19, n.x, n.y, n.z, m_numDevices);
    param->dfT = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);
    param->dfT_tmp = new DistributionFunction(7, n.x, n.y, n.z, m_numDevices);

    param->df->allocate(subLattice);
    param->df_tmp->allocate(subLattice);
    param->dfT->allocate(subLattice);
    param->dfT_tmp->allocate(subLattice);

    runInitKernel(param->df, param->dfT, subLattice, 1.0, 0, 0, 0,
                  param->Tinit);
    runInitKernel(param->df_tmp, param->dfT_tmp, subLattice, 1.0, 0, 0, 0,
                  param->Tinit);
    ss << "Allocated subLattice " << subLattice << " on GPU" << srcDev
       << std::endl;

    // Create arrays for averaging and plotting
    const SubLattice subLatticeNoHalo(subLattice.getLatticeMin(),
                                      subLattice.getLatticeMax(),
                                      glm::ivec3(0, 0, 0));

    param->avg = new DistributionArray<real>(4, n.x, n.y, n.z, m_numDevices);
    param->avg->allocate(subLatticeNoHalo);
    for (int q = 0; q < param->avg->getQ(); q++) param->avg->fill(q, 0);

    param->plot = new DistributionArray<real>(1, n.x, n.y, n.z, m_numDevices);
    param->plot->allocate(subLatticeNoHalo);
    param->plot->fill(0, 0);

    // Scatter voxel array into sublattices
    param->voxels = new VoxelArray(n.x, n.y, n.z, m_numDevices);
    param->voxels->allocate(subLatticeNoHalo);
    param->voxels->scatter(*voxels, subLatticeNoHalo);

    // Upload boundary conditions array
    param->bcs = new device_vector<BoundaryCondition>(*bcs);

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
    ComputeParams *param = m_params.at(srcDev);
    *param->bcs = *bcs;
  }
}

void KernelInterface::resetAverages() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    ComputeParams *param = m_params.at(srcDev);
    for (int q = 0; q < param->avg->getQ(); q++) param->avg->fill(q, 0);
  }
}

void KernelInterface::resetDfs() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    const SubLattice subLattice = getDeviceSubLattice(srcDev);
    ComputeParams *param = m_params.at(srcDev);
    runInitKernel(param->df, param->dfT, subLattice, 1.0, 0, 0, 0,
                  param->Tinit);
    runInitKernel(param->df_tmp, param->dfT_tmp, subLattice, 1.0, 0, 0, 0,
                  param->Tinit);
  }
}

void KernelInterface::plot(int plotDev, thrust::device_vector<real> *plot) {
#pragma omp parallel num_threads(CUDA_MAX_P2P_DEVS)
#pragma omp for
  for (int srcDev = 0; srcDev < m_numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeParams *param = m_params.at(srcDev);
    SubLattice subLattice = param->plot->getDeviceSubLattice(srcDev);

    std::vector<bool> p2pList = getP2PConnections(srcDev);
    bool prevPlotDevP2P = hasP2PConnection(srcDev, plotDev);
    enablePeerAccess(srcDev, plotDev, &p2pList);
    param->plot->gather(subLattice, m_plot);
    if (!prevPlotDevP2P) disablePeerAccess(srcDev, plotDev, &p2pList);
  }

  thrust::device_ptr<real> dp1(m_plot->gpu_ptr(m_plot->getSubLattice(0, 0, 0)));
  thrust::device_ptr<real> dp2(thrust::raw_pointer_cast(&(*plot)[0]));
  thrust::copy(dp1, dp1 + plot->size(), dp2);
}

KernelInterface::~KernelInterface() {}
