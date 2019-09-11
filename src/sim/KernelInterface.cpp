#include "KernelInterface.hpp"

void KernelInterface::runInitKernel(DistributionFunction *df,
                                    DistributionFunction *dfT,
                                    Partition partition, float rho, float vx,
                                    float vy, float vz, float T) {
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

void KernelInterface::runComputeKernelInterior(
    const Partition partition, ComputeParams *params,
    DisplayQuantity::Enum displayQuantity, cudaStream_t stream) {
  glm::ivec3 n = partition.getDims() - 2 * partition.getHalo();

  real *dfPtr = params->df->gpu_ptr(partition);
  real *df_tmpPtr = params->df_tmp->gpu_ptr(partition);
  real *dfTPtr = params->dfT->gpu_ptr(partition);
  real *dfT_tmpPtr = params->dfT_tmp->gpu_ptr(partition);

  Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                              glm::ivec3(0, 0, 0));
  real *avgDstPtr = params->avg->gpu_ptr(partitionNoHalo, m_bufferIndex * 4);
  real *avgSrcPtr =
      params->avg->gpu_ptr(partitionNoHalo, ((m_bufferIndex + 1) % 2) * 4);
  real *plotPtr = params->plot->gpu_ptr(partitionNoHalo, m_bufferIndex);
  voxel *voxelPtr = params->voxels->gpu_ptr(partitionNoHalo);

  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*params->bcs)[0]);

  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);
  ComputeAndPlotKernelInterior<<<gridSize, blockSize, 0, stream>>>(
      partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
      params->nu, params->C, params->nuT, params->Pr_t, params->gBetta,
      params->Tref, displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);

  CUDA_CHECK_ERRORS("ComputeKernelInterior");
}

void KernelInterface::runComputeKernelBoundary(
    D3Q4::Enum direction, const Partition partition, ComputeParams *params,
    DisplayQuantity::Enum displayQuantity, cudaStream_t stream) {
  glm::ivec3 n = partition.getDims();

  real *dfPtr = params->df->gpu_ptr(partition);
  real *df_tmpPtr = params->df_tmp->gpu_ptr(partition);
  real *dfTPtr = params->dfT->gpu_ptr(partition);
  real *dfT_tmpPtr = params->dfT_tmp->gpu_ptr(partition);

  Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                              glm::ivec3(0, 0, 0));
  real *avgDstPtr = params->avg->gpu_ptr(partitionNoHalo, m_bufferIndex * 4);
  real *avgSrcPtr =
      params->avg->gpu_ptr(partitionNoHalo, ((m_bufferIndex + 1) % 2) * 4);
  real *plotPtr = params->plot->gpu_ptr(partitionNoHalo, m_bufferIndex);
  voxel *voxelPtr = params->voxels->gpu_ptr(partitionNoHalo);

  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*params->bcs)[0]);

  if (direction == D3Q4::X_AXIS) {
    dim3 gridSize(n.z, 2, 1);
    dim3 blockSize(n.y, 1, 1);
    ComputeAndPlotKernelBoundaryX<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        params->nu, params->C, params->nuT, params->Pr_t, params->gBetta,
        params->Tref, displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryX");
  }
  if (direction == D3Q4::Y_AXIS) {
    dim3 gridSize(n.z, 2, 1);
    dim3 blockSize(n.x, 1, 1);
    ComputeAndPlotKernelBoundaryY<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        params->nu, params->C, params->nuT, params->Pr_t, params->gBetta,
        params->Tref, displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryY");
  }
  if (direction == D3Q4::Z_AXIS) {
    dim3 gridSize(n.y, 2, 1);
    dim3 blockSize(n.x, 1, 1);
    ComputeAndPlotKernelBoundaryZ<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        params->nu, params->C, params->nuT, params->Pr_t, params->gBetta,
        params->Tref, displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryZ");
  }
}

std::vector<cudaStream_t> KernelInterface::exchange(int srcDev,
                                                    Partition partition,
                                                    D3Q7::Enum direction) {
  ComputeParams *params = m_params.at(srcDev);
  Partition neighbour = params->df_tmp->getNeighbour(partition, direction);
  int dstDev = getPartitionDevice(neighbour);
  cudaStream_t dfStream = getDfHaloStream(srcDev, dstDev);
  cudaStream_t dfTStream = getDfTHaloStream(srcDev, dstDev);
  params->df_tmp->exchange(partition, m_params.at(dstDev)->df_tmp, neighbour,
                           direction, dfStream);
  params->dfT_tmp->exchange(partition, m_params.at(dstDev)->dfT_tmp, neighbour,
                            direction, dfTStream);
  CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
  CUDA_RT_CALL(cudaStreamSynchronize(dfTStream));
  return std::vector<cudaStream_t>{dfStream, dfTStream};
}

Average KernelInterface::getAverage(VoxelVolume area, uint64_t deltaTicks) {
  DistributionArray<real> *array = m_avgs[area];
  Partition lat = array->getPartition();
  Average avg;
  avg.m_temperature = array->getAverage(lat, 0, deltaTicks);
  avg.m_velocityX = array->getAverage(lat, 1, deltaTicks);
  avg.m_velocityY = array->getAverage(lat, 2, deltaTicks);
  avg.m_velocityZ = array->getAverage(lat, 3, deltaTicks);
  return avg;
}

void KernelInterface::compute(DisplayQuantity::Enum displayQuantity,
                              glm::ivec3 slicePos, real *sliceX, real *sliceY,
                              real *sliceZ) {
  const int bufferIndexPrev = (m_bufferIndex + 1) % 2;

#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num() % m_numDevices;

    CUDA_RT_CALL(cudaSetDevice(srcDev));

    ComputeParams *params = m_params.at(srcDev);
    const Partition partition = getDevicePartition(srcDev);
    const Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                                      glm::ivec3(0, 0, 0));

    const cudaStream_t plotStream = getPlotStream(srcDev);
    const cudaStream_t computeStream = getComputeStream(srcDev);
    const cudaStream_t computeBoundaryStream = getComputeBoundaryStream(srcDev);
    const cudaStream_t avgStream = getAvgStream(srcDev);

    // If averages were reset on last call, sync it now
    CUDA_RT_CALL(cudaStreamSynchronize(avgStream));

    // Compute LBM lattice boundary sites
    if (partition.getHalo().x > 0) {
      runComputeKernelBoundary(D3Q4::X_AXIS, partition, params,
                               displayQuantity, computeBoundaryStream);
    }
    if (partition.getHalo().y > 0) {
      runComputeKernelBoundary(D3Q4::Y_AXIS, partition, params,
                               displayQuantity, computeBoundaryStream);
    }
    if (partition.getHalo().z > 0) {
      runComputeKernelBoundary(D3Q4::Z_AXIS, partition, params,
                               displayQuantity, computeBoundaryStream);
    }

    // Compute inner lattice sites (excluding boundaries)
    runComputeKernelInterior(partition, params, displayQuantity,
                             computeStream);

    // Gather the plot to draw the display slices
    if (slicePos != glm::ivec3(-1, -1, -1)) {
      params->plot->gatherSlice(slicePos, bufferIndexPrev, bufferIndexPrev,
                                partitionNoHalo, m_plot, plotStream);
    }

    // Gather averages into arrays
    for (std::pair<VoxelVolume, DistributionArray<real> *> element : m_avgs) {
      VoxelVolume area = element.first;
      DistributionArray<real> *areaArray = element.second;
      for (int dstQ = 0; dstQ < 4; dstQ++) {
        const int srcQ = dstQ + bufferIndexPrev * 4;
        params->avg->gather(area.getMin(), area.getMax(), srcQ, dstQ,
                            partitionNoHalo, areaArray,
                            areaArray->getPartition(), avgStream);
      }
    }

    // Wait for boundary lattice sites to finish computing
    CUDA_RT_CALL(cudaStreamSynchronize(computeBoundaryStream));

    // Perform halo exchanges
    if (partition.getHalo().x > 0) {
      std::vector<cudaStream_t> streamsPos =
          exchange(srcDev, partition, D3Q7::X_AXIS_POS);
      std::vector<cudaStream_t> streamsNeg =
          exchange(srcDev, partition, D3Q7::X_AXIS_NEG);
      for (cudaStream_t stream : streamsPos)
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
      for (cudaStream_t stream : streamsNeg)
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }

#pragma omp barrier
    if (partition.getHalo().y > 0) {
      std::vector<cudaStream_t> streamsPos =
          exchange(srcDev, partition, D3Q7::Y_AXIS_POS);
      std::vector<cudaStream_t> streamsNeg =
          exchange(srcDev, partition, D3Q7::Y_AXIS_NEG);
      for (cudaStream_t stream : streamsPos)
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
      for (cudaStream_t stream : streamsNeg)
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }

#pragma omp barrier
    if (partition.getHalo().z > 0) {
      std::vector<cudaStream_t> streamsPos =
          exchange(srcDev, partition, D3Q7::Z_AXIS_POS);
      std::vector<cudaStream_t> streamsNeg =
          exchange(srcDev, partition, D3Q7::Z_AXIS_NEG);
      for (cudaStream_t stream : streamsPos)
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
      for (cudaStream_t stream : streamsNeg)
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }
    CUDA_RT_CALL(cudaStreamSynchronize(plotStream));

#pragma omp barrier
    if (srcDev == 0 && slicePos != glm::ivec3(-1, -1, -1)) {
      real *plot3dPtr =
          m_plot->gpu_ptr(m_plot->getPartition(), bufferIndexPrev);
      dim3 blockSize, gridSize;

      setDims(getDims().y * getDims().z, BLOCK_SIZE_DEFAULT, blockSize,
              gridSize);
      SliceXRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, getDims().x, getDims().y, getDims().z, sliceX, slicePos.x);
      CUDA_CHECK_ERRORS("SliceXRenderKernel");

      setDims(getDims().x * getDims().z, BLOCK_SIZE_DEFAULT, blockSize,
              gridSize);
      SliceYRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, getDims().x, getDims().y, getDims().z, sliceY, slicePos.y);
      CUDA_CHECK_ERRORS("SliceYRenderKernel");

      setDims(getDims().x * getDims().y, BLOCK_SIZE_DEFAULT, blockSize,
              gridSize);
      SliceZRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, getDims().x, getDims().y, getDims().z, sliceZ, slicePos.z);
      CUDA_CHECK_ERRORS("SliceZRenderKernel");
    }

    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));
    CUDA_RT_CALL(cudaStreamSynchronize(avgStream));
    CUDA_RT_CALL(cudaStreamSynchronize(plotStream));

#pragma omp barrier
    if (m_resetAvg) params->avg->fill(0, avgStream);
    DistributionFunction::swap(params->df, params->df_tmp);
    DistributionFunction::swap(params->dfT, params->dfT_tmp);
  }
  m_bufferIndex = bufferIndexPrev;
  m_resetAvg = false;
}

KernelInterface::KernelInterface(
    const int nx, const int ny, const int nz,
    const std::shared_ptr<ComputeParams> cmptParams,
    const std::shared_ptr<BoundaryConditions> bcs,
    const std::shared_ptr<VoxelArray> voxels,
    const std::shared_ptr<VoxelVolumeArray> avgAreas, const int numDevices)
    : P2PLattice(nx, ny, nz, numDevices),
      m_params(numDevices),
      m_bufferIndex(0),
      m_resetAvg(false) {
  std::cout << "Initializing LBM data structures..." << std::endl;
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // For gathering distributed plot onto GPU0
  m_plot = new DistributionArray<real>(2, nx, ny, nz);
  m_plot->allocate();
  m_plot->fill(0);

  // int avgsTotalSize = 0;
  // for (VoxelVolume avg : avgAreas) {
  //   glm::ivec3 dims = avg.getDims();
  //   avgsTotalSize += dims.x * dims.y * dims.z;
  // }
  // m_avgs = new DistributionArray<real>(4, avgsTotalSize, 0, 0);
  // m_avgs->allocate();
  // m_avgs->fill(0);

  for (int i = 0; i < avgAreas->size(); i++) {
    VoxelVolume area = avgAreas->at(i);
    glm::ivec3 dims = area.getDims();
    DistributionArray<real> *array =
        new DistributionArray<real>(4, dims.x, dims.y, dims.z);
    array->allocate();
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
    const Partition partition = getDevicePartition(srcDev);

    params->df = new DistributionFunction(19, nx, ny, nz, m_numDevices);
    params->df_tmp = new DistributionFunction(19, nx, ny, nz, m_numDevices);
    params->dfT = new DistributionFunction(7, nx, ny, nz, m_numDevices);
    params->dfT_tmp = new DistributionFunction(7, nx, ny, nz, m_numDevices);

    params->df->allocate(partition);
    params->df_tmp->allocate(partition);
    params->dfT->allocate(partition);
    params->dfT_tmp->allocate(partition);

    runInitKernel(params->df, params->dfT, partition, 1.0, 0, 0, 0,
                  params->Tinit);
    runInitKernel(params->df_tmp, params->dfT_tmp, partition, 1.0, 0, 0, 0,
                  params->Tinit);
    ss << "Allocated partition " << partition << " on GPU" << srcDev
       << std::endl;

    // Create arrays for averaging and plotting
    const Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                                      glm::ivec3(0, 0, 0));

    params->avg = new DistributionArray<real>(8, nx, ny, nz, m_numDevices);
    params->avg->allocate(partitionNoHalo);
    params->avg->fill(0);

    params->plot = new DistributionArray<real>(2, nx, ny, nz, m_numDevices);
    params->plot->allocate(partitionNoHalo);
    params->plot->fill(0);

    // Scatter voxel array into partitions
    params->voxels = new VoxelArray(nx, ny, nz, m_numDevices);
    params->voxels->allocate(partitionNoHalo);
    params->voxels->scatter(*voxels, partitionNoHalo);

    // Upload boundary conditions array
    params->bcs = new thrust::device_vector<BoundaryCondition>(*bcs);

    CUDA_RT_CALL(cudaDeviceSynchronize());
    std::cout << ss.str();
  }  // end omp parallel

  std::cout << "LBM initialized" << std::endl;
}

void KernelInterface::uploadBCs(std::shared_ptr<BoundaryConditions> bcs) {
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
    const Partition partition = getDevicePartition(srcDev);
    const Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                                      glm::ivec3(0, 0, 0));
    ComputeParams *params = m_params.at(srcDev);
    mins[srcDev] = params->plot->getMin(partitionNoHalo);
    maxes[srcDev] = params->plot->getMax(partitionNoHalo);
  }
  *max = *thrust::max_element(maxes.begin(), maxes.end());
  *min = *thrust::min_element(mins.begin(), mins.end());
}

void KernelInterface::resetDfs() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const Partition partition = getDevicePartition(srcDev);
    ComputeParams *params = m_params.at(srcDev);
    runInitKernel(params->df, params->dfT, partition, 1.0, 0, 0, 0,
                  params->Tinit);
    runInitKernel(params->df_tmp, params->dfT_tmp, partition, 1.0, 0, 0, 0,
                  params->Tinit);
  }
}
