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

void KernelInterface::runComputeKernel(const SubLattice subLattice,
                                       ComputeParams *params,
                                       DisplayQuantity::Enum displayQuantity,
                                       cudaStream_t stream) {
  glm::ivec3 partMin = subLattice.getMin();
  glm::ivec3 partMax = subLattice.getMax();
  glm::ivec3 partHalo = subLattice.getHalo();
  glm::ivec3 n = subLattice.getDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);

  real *dfPtr = params->df->gpu_ptr(subLattice);
  real *df_tmpPtr = params->df_tmp->gpu_ptr(subLattice);
  real *dfTPtr = params->dfT->gpu_ptr(subLattice);
  real *dfT_tmpPtr = params->dfT_tmp->gpu_ptr(subLattice);

  SubLattice subLatticeNoHalo(subLattice.getMin(), subLattice.getMax(),
                              glm::ivec3(0, 0, 0));
  real *avgDstPtr = params->avg->gpu_ptr(subLatticeNoHalo, m_bufferIndex * 4);
  real *avgSrcPtr =
      params->avg->gpu_ptr(subLatticeNoHalo, ((m_bufferIndex + 1) % 2) * 4);
  real *plotPtr = params->plot->gpu_ptr(subLatticeNoHalo, m_bufferIndex);
  voxel *voxelPtr = params->voxels->gpu_ptr(subLatticeNoHalo);

  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*params->bcs)[0]);

  ComputeKernel<<<gridSize, blockSize, 0, stream>>>(
      subLattice, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, plotPtr, avgSrcPtr,
      avgDstPtr, voxelPtr, bcsPtr, params->nu, params->C, params->nuT,
      params->Pr_t, params->gBetta, params->Tref, displayQuantity);

  CUDA_CHECK_ERRORS("ComputeKernel");
}

void KernelInterface::exchange(int srcDev, SubLattice subLattice,
                               D3Q7::Enum direction) {
  ComputeParams *params = m_params.at(srcDev);
  SubLattice neighbour = params->df_tmp->getNeighbour(subLattice, direction);
  int dstDev = getSubLatticeDevice(neighbour);
  cudaStream_t dfStream = getDfHaloStream(srcDev, dstDev);
  cudaStream_t dfTStream = getDfTHaloStream(srcDev, dstDev);
  params->df_tmp->exchange(subLattice, m_params.at(dstDev)->df_tmp, neighbour,
                           direction, dfStream);
  params->dfT_tmp->exchange(subLattice, m_params.at(dstDev)->dfT_tmp, neighbour,
                            direction, dfTStream);
  CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
  CUDA_RT_CALL(cudaStreamSynchronize(dfTStream));
}

Average KernelInterface::getAverage(VoxelArea area, uint64_t deltaTicks) {
  DistributionArray<real> *array = m_avgs[area];
  SubLattice lat = array->getSubLattice(0, 0, 0);
  Average avg;
  avg.m_temperature = array->getAverage(lat, 0, deltaTicks);
  avg.m_velocityX = array->getAverage(lat, 1, deltaTicks);
  avg.m_velocityY = array->getAverage(lat, 2, deltaTicks);
  avg.m_velocityZ = array->getAverage(lat, 3, deltaTicks);
  return avg;
}

void KernelInterface::compute(DisplayQuantity::Enum displayQuantity,
                              glm::ivec3 slicePos) {
  const int bufferIndexPrev = (m_bufferIndex + 1) % 2;

#pragma omp parallel num_threads(m_numDevices * 2)
  {
    const int srcDev = omp_get_thread_num() % m_numDevices;

    const bool computeThread = omp_get_thread_num() % 2 == 0;
    const bool plotThread = omp_get_thread_num() % 2 == 1;

    CUDA_RT_CALL(cudaSetDevice(srcDev));

    ComputeParams *params = m_params.at(srcDev);
    const SubLattice subLattice = getDeviceSubLattice(srcDev);
    const SubLattice subLatticeNoHalo(subLattice.getMin(), subLattice.getMax(),
                                      glm::ivec3(0, 0, 0));

    const cudaStream_t plotStream = getPlotStream(srcDev);
    const cudaStream_t computeStream = getComputeStream(srcDev);
    const cudaStream_t avgStream = getAvgStream(srcDev);

    // If averages were reset on last call, sync it now
    CUDA_RT_CALL(cudaStreamSynchronize(avgStream));

    if (computeThread) {
      // Compute LBM lattice
      runComputeKernel(subLattice, params, displayQuantity, computeStream);
    }
    if (plotThread) {
      // Gather the plot to draw the display slices
      if (slicePos != glm::ivec3(-1, -1, -1)) {
        params->plot->gatherSlice(slicePos, bufferIndexPrev, bufferIndexPrev,
                                  subLatticeNoHalo, m_plot, plotStream);
      }
      // Gather averages into arrays
      for (std::pair<VoxelArea, DistributionArray<real> *> element : m_avgs) {
        VoxelArea area = element.first;
        DistributionArray<real> *areaArray = element.second;
        for (int dstQ = 0; dstQ < 4; dstQ++) {
          const int srcQ = dstQ + bufferIndexPrev * 4;
          params->avg->gather(area.getMin(), area.getMax(), srcQ, dstQ,
                              subLatticeNoHalo, areaArray,
                              areaArray->getSubLattice(0, 0, 0));
        }
      }
    }

    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));

#pragma omp barrier
    if (computeThread && subLattice.getHalo().x > 0) {
      exchange(srcDev, subLattice, D3Q7::X_AXIS_POS);
      exchange(srcDev, subLattice, D3Q7::X_AXIS_NEG);
    }

#pragma omp barrier
    if (computeThread && subLattice.getHalo().y > 0) {
      exchange(srcDev, subLattice, D3Q7::Y_AXIS_POS);
      exchange(srcDev, subLattice, D3Q7::Y_AXIS_NEG);
    }

#pragma omp barrier
    if (computeThread && subLattice.getHalo().z > 0) {
      exchange(srcDev, subLattice, D3Q7::Z_AXIS_POS);
      exchange(srcDev, subLattice, D3Q7::Z_AXIS_NEG);
    }

#pragma omp barrier
    CUDA_RT_CALL(cudaStreamSynchronize(plotStream));
    CUDA_RT_CALL(cudaStreamSynchronize(avgStream));
    if (m_resetAvg) params->avg->fill(0, avgStream);

    if (computeThread) {
      DistributionFunction::swap(params->df, params->df_tmp);
      DistributionFunction::swap(params->dfT, params->dfT_tmp);
    }
  }
  m_bufferIndex = bufferIndexPrev;
  m_resetAvg = false;
}

KernelInterface::KernelInterface(const int nx, const int ny, const int nz,
                                 const ComputeParams *cmptParams,
                                 const std::vector<BoundaryCondition> *bcs,
                                 const VoxelArray *voxels,
                                 const std::vector<VoxelArea> *avgAreas,
                                 const int numDevices = 1)
    : P2PLattice(nx, ny, nz, numDevices),
      m_params(numDevices),
      m_bufferIndex(0),
      m_resetAvg(false) {
  std::cout << "Initializing LBM data structures..." << std::endl;
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // For gathering distributed plot onto GPU0
  m_plot = new DistributionArray<real>(2, nx, ny, nz);
  m_plot->allocate(m_plot->getSubLattice(0, 0, 0));
  m_plot->fill(0);

  for (int i = 0; i < avgAreas->size(); i++) {
    VoxelArea area = avgAreas->at(i);
    glm::ivec3 dims = area.getDims();
    DistributionArray<real> *array =
        new DistributionArray<real>(4, dims.x, dims.y, dims.z);
    array->allocate(array->getSubLattice(0, 0, 0));
    array->fill(0);
    m_avgs[area] = array;
  }

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeParams *params = new ComputeParams(*cmptParams);
    m_params.at(srcDev) = params;

    // Initialize distribution functions for temperature, velocity and tmps
    const SubLattice subLattice = getDeviceSubLattice(srcDev);

    params->df = new DistributionFunction(19, nx, ny, nz, m_numDevices);
    params->df_tmp = new DistributionFunction(19, nx, ny, nz, m_numDevices);
    params->dfT = new DistributionFunction(7, nx, ny, nz, m_numDevices);
    params->dfT_tmp = new DistributionFunction(7, nx, ny, nz, m_numDevices);

    params->df->allocate(subLattice);
    params->df_tmp->allocate(subLattice);
    params->dfT->allocate(subLattice);
    params->dfT_tmp->allocate(subLattice);

    runInitKernel(params->df, params->dfT, subLattice, 1.0, 0, 0, 0,
                  params->Tinit);
    runInitKernel(params->df_tmp, params->dfT_tmp, subLattice, 1.0, 0, 0, 0,
                  params->Tinit);
    ss << "Allocated subLattice " << subLattice << " on GPU" << srcDev
       << std::endl;

    // Create arrays for averaging and plotting
    const SubLattice subLatticeNoHalo(subLattice.getMin(), subLattice.getMax(),
                                      glm::ivec3(0, 0, 0));

    params->avg = new DistributionArray<real>(8, nx, ny, nz, m_numDevices);
    params->avg->allocate(subLatticeNoHalo);
    params->avg->fill(0);

    params->plot = new DistributionArray<real>(2, nx, ny, nz, m_numDevices);
    params->plot->allocate(subLatticeNoHalo);
    params->plot->fill(0);

    // Scatter voxel array into sublattices
    params->voxels = new VoxelArray(nx, ny, nz, m_numDevices);
    params->voxels->allocate(subLatticeNoHalo);
    params->voxels->scatter(*voxels, subLatticeNoHalo);

    // Upload boundary conditions array
    params->bcs = new thrust::device_vector<BoundaryCondition>(*bcs);

    CUDA_RT_CALL(cudaDeviceSynchronize());
    std::cout << ss.str();
  }  // end omp parallel

  std::cout << "LBM initialized" << std::endl;
}

void KernelInterface::uploadBCs(std::vector<BoundaryCondition> *bcs) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    ComputeParams *params = m_params.at(srcDev);
    *params->bcs = *bcs;
  }
}

void KernelInterface::getMinMax(real *min, real *max) {
  *min = REAL_MAX;
  *max = REAL_MIN;
  thrust::host_vector<real> mins(m_numDevices);
  thrust::host_vector<real> maxes(m_numDevices);
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const SubLattice subLattice = getDeviceSubLattice(srcDev);
    const SubLattice subLatticeNoHalo(subLattice.getMin(), subLattice.getMax(),
                                      glm::ivec3(0, 0, 0));
    ComputeParams *params = m_params.at(srcDev);
    params->plot->getMin(subLatticeNoHalo, &mins[srcDev]);
    params->plot->getMax(subLatticeNoHalo, &maxes[srcDev]);
  }
  *max = *thrust::max_element(maxes.begin(), maxes.end());
  *min = *thrust::min_element(mins.begin(), mins.end());
}

void KernelInterface::resetDfs() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const SubLattice subLattice = getDeviceSubLattice(srcDev);
    ComputeParams *params = m_params.at(srcDev);
    runInitKernel(params->df, params->dfT, subLattice, 1.0, 0, 0, 0,
                  params->Tinit);
    runInitKernel(params->df_tmp, params->dfT_tmp, subLattice, 1.0, 0, 0, 0,
                  params->Tinit);
  }
}

void KernelInterface::plot(thrust::device_vector<real> *plot) {
  thrust::device_ptr<real> dp1(
      m_plot->gpu_ptr(m_plot->getSubLattice(0, 0, 0), m_bufferIndex));
  thrust::device_ptr<real> dp2(thrust::raw_pointer_cast(&(*plot)[0]));
  thrust::copy(dp1, dp1 + plot->size(), dp2);
}

KernelInterface::~KernelInterface() {}
