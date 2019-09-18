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
    const Partition partition, ComputeParams *par,
    DisplayQuantity::Enum displayQuantity, cudaStream_t stream) {
  glm::ivec3 n = partition.getDims() - 2 * partition.getHalo();

  real *dfPtr = par->df->gpu_ptr(partition);
  real *df_tmpPtr = par->df_tmp->gpu_ptr(partition);
  real *dfTPtr = par->dfT->gpu_ptr(partition);
  real *dfT_tmpPtr = par->dfT_tmp->gpu_ptr(partition);

  Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                            glm::ivec3(0, 0, 0));
  real *avgDstPtr = par->avg->gpu_ptr(partitionNoHalo, m_bufferIndex * 4);
  real *avgSrcPtr =
      par->avg->gpu_ptr(partitionNoHalo, ((m_bufferIndex + 1) % 2) * 4);
  real *plotPtr = par->plot->gpu_ptr(partitionNoHalo, m_bufferIndex);
  voxel *voxelPtr = par->voxels->gpu_ptr(partitionNoHalo);

  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*par->bcs)[0]);

  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);
  ComputeAndPlotKernelInterior<<<gridSize, blockSize, 0, stream>>>(
      partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
      par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref,
      displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);

  CUDA_CHECK_ERRORS("ComputeKernelInterior");
}

void KernelInterface::runComputeKernelBoundary(
    D3Q4::Enum direction, const Partition partition, ComputeParams *par,
    DisplayQuantity::Enum displayQuantity, cudaStream_t stream) {
  glm::ivec3 n = partition.getDims();

  real *dfPtr = par->df->gpu_ptr(partition);
  real *df_tmpPtr = par->df_tmp->gpu_ptr(partition);
  real *dfTPtr = par->dfT->gpu_ptr(partition);
  real *dfT_tmpPtr = par->dfT_tmp->gpu_ptr(partition);

  Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                            glm::ivec3(0, 0, 0));
  real *avgDstPtr = par->avg->gpu_ptr(partitionNoHalo, m_bufferIndex * 4);
  real *avgSrcPtr =
      par->avg->gpu_ptr(partitionNoHalo, ((m_bufferIndex + 1) % 2) * 4);
  real *plotPtr = par->plot->gpu_ptr(partitionNoHalo, m_bufferIndex);
  voxel *voxelPtr = par->voxels->gpu_ptr(partitionNoHalo);

  BoundaryCondition *bcsPtr = thrust::raw_pointer_cast(&(*par->bcs)[0]);

  if (direction == D3Q4::X_AXIS) {
    dim3 gridSize(n.z, 2, 1);
    dim3 blockSize(n.y, 1, 1);
    ComputeAndPlotKernelBoundaryX<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref,
        displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryX");
  }
  if (direction == D3Q4::Y_AXIS) {
    dim3 gridSize(n.z, 2, 1);
    dim3 blockSize(n.x, 1, 1);
    ComputeAndPlotKernelBoundaryY<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref,
        displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryY");
  }
  if (direction == D3Q4::Z_AXIS) {
    dim3 gridSize(n.y, 2, 1);
    dim3 blockSize(n.x, 1, 1);
    ComputeAndPlotKernelBoundaryZ<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref,
        displayQuantity, plotPtr, avgSrcPtr, avgDstPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryZ");
  }
}

std::vector<cudaStream_t> KernelInterface::exchange(int srcDev,
                                                    Partition partition,
                                                    D3Q7::Enum direction) {
  ComputeParams *par = m_params.at(srcDev);
  Partition neighbour = par->df_tmp->getNeighbour(partition, direction);
  int dstDev = getPartitionDevice(neighbour);
  cudaStream_t dfStream = getDfHaloStream(srcDev, dstDev);
  cudaStream_t dfTStream = getDfTHaloStream(srcDev, dstDev);
  par->df_tmp->exchange(partition, m_params.at(dstDev)->df_tmp, neighbour,
                        direction, dfStream);
  par->dfT_tmp->exchange(partition, m_params.at(dstDev)->dfT_tmp, neighbour,
                         direction, dfTStream);
  CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
  CUDA_RT_CALL(cudaStreamSynchronize(dfTStream));
  return std::vector<cudaStream_t>{dfStream, dfTStream};
}

LatticeAverage KernelInterface::getAverage(VoxelVolume vol,
                                           uint64_t deltaTicks) {
  unsigned int offset = m_avgOffsets[vol];
  unsigned int size = vol.getNumVoxels();
  real temperature = m_avgs->getAverage(m_avgs->getPartition(), 0, offset, size,
                                        static_cast<real>(deltaTicks));
  real velocityX = m_avgs->getAverage(m_avgs->getPartition(), 1, offset, size,
                                      static_cast<real>(deltaTicks));
  real velocityY = m_avgs->getAverage(m_avgs->getPartition(), 2, offset, size,
                                      static_cast<real>(deltaTicks));
  real velocityZ = m_avgs->getAverage(m_avgs->getPartition(), 3, offset, size,
                                      static_cast<real>(deltaTicks));
  return LatticeAverage(temperature, velocityX, velocityY, velocityZ);
}

void KernelInterface::compute(DisplayQuantity::Enum displayQuantity,
                              glm::ivec3 slicePos, real *sliceX, real *sliceY,
                              real *sliceZ) {
  const int bufferIndexPrev = (m_bufferIndex + 1) % 2;

#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num() % m_numDevices;

    CUDA_RT_CALL(cudaSetDevice(srcDev));

    ComputeParams *par = m_params.at(srcDev);
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
      runComputeKernelBoundary(D3Q4::X_AXIS, partition, par, displayQuantity,
                               computeBoundaryStream);
    }
    if (partition.getHalo().y > 0) {
      runComputeKernelBoundary(D3Q4::Y_AXIS, partition, par, displayQuantity,
                               computeBoundaryStream);
    }
    if (partition.getHalo().z > 0) {
      runComputeKernelBoundary(D3Q4::Z_AXIS, partition, par, displayQuantity,
                               computeBoundaryStream);
    }

    // Compute inner lattice sites (excluding boundaries)
    runComputeKernelInterior(partition, par, displayQuantity, computeStream);

    // Gather the plot to draw the display slices
    if (slicePos != glm::ivec3(-1, -1, -1)) {
      par->plot->gatherSlice(slicePos, bufferIndexPrev, bufferIndexPrev,
                             partitionNoHalo, m_plot, plotStream);
    }

    // Gather averages from GPU array
    thrust::device_vector<real> *values =
        par->avg->getDeviceVector(partitionNoHalo);
    thrust::device_vector<real> *output =
        m_avgs->getDeviceVector(m_avgs->getPartition());
    size_t bufOffset = bufferIndexPrev * 4 * partitionNoHalo.getArrayStride();

    thrust::gather_if(thrust::cuda::par.on(avgStream), par->avgMap->begin(),
                      par->avgMap->end(), par->avgStencil->begin(),
                      values->begin() + bufOffset, output->begin());

    // for (int i = 0; i < par->avgMap->size(); i++) {
    //   int m = (*par->avgMap)[i];
    //   int s = (*par->avgStencil)[i];
    //   if (s) {
    //     real v = (*values)[m + bufOffset];
    //     (*output)[i] = v;
    //   }
    // }

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
    if (m_resetAvg) par->avg->fill(0, avgStream);
    DistributionFunction::swap(par->df, par->df_tmp);
    DistributionFunction::swap(par->dfT, par->dfT_tmp);
  }
  m_bufferIndex = bufferIndexPrev;
  m_resetAvg = false;
}

KernelInterface::KernelInterface(
    const int nx, const int ny, const int nz,
    const std::shared_ptr<ComputeParams> cmptParams,
    const std::shared_ptr<BoundaryConditions> bcs,
    const std::shared_ptr<VoxelArray> voxels,
    const std::shared_ptr<VoxelVolumeArray> avgVols, const int numDevices)
    : P2PLattice(nx, ny, nz, numDevices),
      m_params(numDevices),
      m_bufferIndex(0),
      m_resetAvg(false) {
  std::cout << "Initializing LBM data structures..." << std::endl;
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // Array for gathering distributed plot onto GPU0
  m_plot = new DistributionArray<real>(2, nx, ny, nz);
  m_plot->allocate();
  m_plot->fill(0);

  // Array for gathering simulation averages onto GPU0
  const int avgNq = 4;
  int avgSizeTotal = 0;
  for (int avgIdx = 0; avgIdx < avgVols->size(); avgIdx++) {
    VoxelVolume vol = avgVols->at(avgIdx);
    m_avgOffsets[vol] = avgSizeTotal;
    glm::ivec3 aDims = vol.getDims();
    avgSizeTotal += aDims.x * aDims.y * aDims.z;
  }
  m_avgs = new DistributionArray<real>(avgNq, avgSizeTotal, 0, 0);
  m_avgs->allocate();
  m_avgs->fill(0);

  size_t avgGpuSize = m_avgs->size(m_avgs->getPartition());
  assert(avgGpuSize == avgNq * avgSizeTotal);

  // Create avgMaps and avgStencils for gather_if
  std::vector<int> *avgMaps[m_numDevices];
  std::vector<int> *avgStencils[m_numDevices];
  for (int srcDev = 0; srcDev < m_numDevices; srcDev++) {
    avgMaps[srcDev] = new std::vector<int>(avgNq * avgSizeTotal, 0);
    avgStencils[srcDev] = new std::vector<int>(avgNq * avgSizeTotal, 0);
  }
  int avgArrayIdx = 0;
  for (int avgIdx = 0; avgIdx < avgVols->size(); avgIdx++) {
    VoxelVolume avg = avgVols->at(avgIdx);
    glm::ivec3 aMin = avg.getMin();
    glm::ivec3 aMax = avg.getMax();

    for (int z = aMin.z; z < aMax.z; z++)
      for (int y = aMin.y; y < aMax.y; y++)
        for (int x = aMin.x; x < aMax.x; x++) {
          glm::ivec3 avgVox = glm::ivec3(x, y, z);

          for (int srcDev = 0; srcDev < m_numDevices; srcDev++) {
            const Partition partition = getDevicePartition(srcDev);
            const Partition partitionNoHalo(
                partition.getMin(), partition.getMax(), glm::ivec3(0, 0, 0));

            const glm::ivec3 pMin = partitionNoHalo.getMin();
            const glm::ivec3 pMax = partitionNoHalo.getMax();
            const glm::ivec3 pDims = partitionNoHalo.getDims();
            const glm::ivec3 pArrDims = partitionNoHalo.getArrayDims();
            const glm::ivec3 pHalo = partitionNoHalo.getHalo();

            if ((pMin.x <= avgVox.x && avgVox.x < pMax.x) &&
                (pMin.y <= avgVox.y && avgVox.y < pMax.y) &&
                (pMin.z <= avgVox.z && avgVox.z < pMax.z)) {
              glm::ivec3 srcPos = avgVox - pMin + pHalo;
              for (int q = 0; q < avgNq; q++) {
                int srcIndex = I4D(q, srcPos.x, srcPos.y, srcPos.z, pArrDims.x,
                                   pArrDims.y, pArrDims.z);
                int mapIdx = q * avgSizeTotal + avgArrayIdx;
                avgMaps[srcDev]->at(mapIdx) = srcIndex;
                avgStencils[srcDev]->at(mapIdx) = 1;
              }
              // Voxel can only be on one GPU...
              break;
            }
          }
          avgArrayIdx++;
        }
  }
  assert(avgArrayIdx == avgSizeTotal);

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(m_numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeParams *par = new ComputeParams(*cmptParams);
    m_params.at(srcDev) = par;

    // Initialize distribution functions for temperature, velocity and tmps
    const Partition partition = getDevicePartition(srcDev);

    par->df = new DistributionFunction(19, nx, ny, nz, m_numDevices);
    par->df_tmp = new DistributionFunction(19, nx, ny, nz, m_numDevices);
    par->dfT = new DistributionFunction(7, nx, ny, nz, m_numDevices);
    par->dfT_tmp = new DistributionFunction(7, nx, ny, nz, m_numDevices);

    par->df->allocate(partition);
    par->df_tmp->allocate(partition);
    par->dfT->allocate(partition);
    par->dfT_tmp->allocate(partition);

    runInitKernel(par->df, par->dfT, partition, 1.0, 0, 0, 0, par->Tinit);
    runInitKernel(par->df_tmp, par->dfT_tmp, partition, 1.0, 0, 0, 0,
                  par->Tinit);
    ss << "Allocated partition " << partition << " on GPU" << srcDev
       << std::endl;

    // Create arrays for averaging and plotting
    const Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                                    glm::ivec3(0, 0, 0));

    par->avg = new DistributionArray<real>(8, nx, ny, nz, m_numDevices);
    par->avg->allocate(partitionNoHalo);
    par->avg->fill(0);

    par->avgMap = new thrust::device_vector<int>(*avgMaps[srcDev]);
    par->avgStencil = new thrust::device_vector<int>(*avgStencils[srcDev]);

    par->plot = new DistributionArray<real>(2, nx, ny, nz, m_numDevices);
    par->plot->allocate(partitionNoHalo);
    par->plot->fill(0);

    // Scatter voxel array into partitions
    par->voxels = new VoxelArray(nx, ny, nz, m_numDevices);
    par->voxels->allocate(partitionNoHalo);
    par->voxels->scatter(*voxels, partitionNoHalo);

    // Upload boundary conditions array
    par->bcs = new thrust::device_vector<BoundaryCondition>(*bcs);

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
    ComputeParams *par = m_params.at(srcDev);
    *par->bcs = *bcs;
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
    ComputeParams *par = m_params.at(srcDev);
    mins[srcDev] = par->plot->getMin(partitionNoHalo);
    maxes[srcDev] = par->plot->getMax(partitionNoHalo);
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
    ComputeParams *par = m_params.at(srcDev);
    runInitKernel(par->df, par->dfT, partition, 1.0, 0, 0, 0, par->Tinit);
    runInitKernel(par->df_tmp, par->dfT_tmp, partition, 1.0, 0, 0, 0,
                  par->Tinit);
  }
}
