#include "KernelInterface.hpp"

void KernelInterface::runInitKernel(DistributionFunction* df,
                                    DistributionFunction* dfT,
                                    Partition partition,
                                    float rho,
                                    float vx,
                                    float vy,
                                    float vz,
                                    float T) {
  float sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
  Eigen::Vector3i n = partition.getArrayExtents();
  dim3 gridSize(n.y(), n.z(), 1);
  dim3 blockSize(n.x(), 1, 1);
  real* dfPtr = df->gpu_ptr(partition);
  real* dfTPtr = dfT->gpu_ptr(partition);

  InitKernel<<<gridSize, blockSize>>>(dfPtr, dfTPtr, n.x(), n.y(), n.z(), rho,
                                      vx, vy, vz, T, sq_term);

  CUDA_CHECK_ERRORS("InitKernel");
}

void KernelInterface::runComputeKernelInterior(
    const Partition partition,
    ComputeParams* par,
    DisplayQuantity::Enum displayQuantity,
    cudaStream_t stream) {
  Eigen::Vector3i n = partition.getExtents() - 2 * partition.getGhostLayer();

  real* dfPtr = par->df->gpu_ptr(partition);
  real* df_tmpPtr = par->df_tmp->gpu_ptr(partition);
  real* dfTPtr = par->dfT->gpu_ptr(partition);
  real* dfT_tmpPtr = par->dfT_tmp->gpu_ptr(partition);

  Partition partitionNoGhostLayer(partition.getMin(), partition.getMax(),
                                  Eigen::Vector3i(0, 0, 0));
  real* avgSrcPtr = par->avg->gpu_ptr(partitionNoGhostLayer);
  real* avgDstPtr = par->avg_tmp->gpu_ptr(partitionNoGhostLayer);
  real* plotPtr = par->plot_tmp->gpu_ptr(partitionNoGhostLayer);
  voxel_t* voxelPtr = par->voxels->gpu_ptr(partitionNoGhostLayer);

  BoundaryCondition* bcsPtr = thrust::raw_pointer_cast(&(*par->bcs)[0]);

  dim3 gridSize(n.y(), n.z(), 1);
  dim3 blockSize(n.x(), 1, 1);
  ComputeAndPlotKernelInterior<<<gridSize, blockSize, 0, stream>>>(
      partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
      par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref, avgSrcPtr,
      avgDstPtr, displayQuantity, plotPtr);

  CUDA_CHECK_ERRORS("ComputeKernelInterior");
}

void KernelInterface::runComputeKernelBoundary(
    D3Q4::Enum direction,
    const Partition partition,
    ComputeParams* par,
    DisplayQuantity::Enum displayQuantity,
    cudaStream_t stream) {
  Eigen::Vector3i n = partition.getExtents();

  real* dfPtr = par->df->gpu_ptr(partition);
  real* df_tmpPtr = par->df_tmp->gpu_ptr(partition);
  real* dfTPtr = par->dfT->gpu_ptr(partition);
  real* dfT_tmpPtr = par->dfT_tmp->gpu_ptr(partition);

  Partition partitionNoGhostLayer(partition.getMin(), partition.getMax(),
                                  Eigen::Vector3i(0, 0, 0));
  real* avgSrcPtr = par->avg->gpu_ptr(partitionNoGhostLayer);
  real* avgDstPtr = par->avg_tmp->gpu_ptr(partitionNoGhostLayer);
  real* plotPtr = par->plot_tmp->gpu_ptr(partitionNoGhostLayer);
  voxel_t* voxelPtr = par->voxels->gpu_ptr(partitionNoGhostLayer);

  BoundaryCondition* bcsPtr = thrust::raw_pointer_cast(&(*par->bcs)[0]);

  if (direction == D3Q4::X_AXIS) {
    dim3 gridSize(n.z(), 2, 1);
    dim3 blockSize(n.y(), 1, 1);
    ComputeAndPlotKernelBoundaryX<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref, avgSrcPtr,
        avgDstPtr, displayQuantity, plotPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryX");
  }
  if (direction == D3Q4::Y_AXIS) {
    dim3 gridSize(n.z(), 2, 1);
    dim3 blockSize(n.x(), 1, 1);
    ComputeAndPlotKernelBoundaryY<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref, avgSrcPtr,
        avgDstPtr, displayQuantity, plotPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryY");
  }
  if (direction == D3Q4::Z_AXIS) {
    dim3 gridSize(n.y(), 2, 1);
    dim3 blockSize(n.x(), 1, 1);
    ComputeAndPlotKernelBoundaryZ<<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, voxelPtr, bcsPtr,
        par->nu, par->C, par->nuT, par->Pr_t, par->gBetta, par->Tref, avgSrcPtr,
        avgDstPtr, displayQuantity, plotPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryZ");
  }
}

std::vector<cudaStream_t> KernelInterface::exchange(int srcDev,
                                                    Partition partition,
                                                    D3Q7::Enum direction) {
  ComputeParams* par = m_params.at(srcDev);
  Partition neighbour = par->df_tmp->getNeighbour(partition, direction);
  int dstDev = getPartitionDevice(neighbour);
  cudaStream_t dfStream = getDfGhostLayerStream(srcDev, dstDev);
  cudaStream_t dfTStream = getDfTGhostLayerStream(srcDev, dstDev);
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
                              Eigen::Vector3i slicePos,
                              real* sliceX,
                              real* sliceY,
                              real* sliceZ,
                              bool runSimulation) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num() % m_numDevices;

    CUDA_RT_CALL(cudaSetDevice(srcDev));

    ComputeParams* par = m_params.at(srcDev);
    const Partition partition = getDevicePartition(srcDev);
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), Eigen::Vector3i(0, 0, 0));

    const cudaStream_t plotStream = getPlotStream(srcDev);
    const cudaStream_t computeStream = getComputeStream(srcDev);
    const cudaStream_t computeBoundaryStream = getComputeBoundaryStream(srcDev);
    const cudaStream_t avgStream = getAvgStream(srcDev);

    // Compute LBM lattice boundary sites
    if (partition.getGhostLayer().x() > 0 && runSimulation) {
      runComputeKernelBoundary(D3Q4::X_AXIS, partition, par, displayQuantity,
                               computeBoundaryStream);
    }
    if (partition.getGhostLayer().y() > 0 && runSimulation) {
      runComputeKernelBoundary(D3Q4::Y_AXIS, partition, par, displayQuantity,
                               computeBoundaryStream);
    }
    if (partition.getGhostLayer().z() > 0 && runSimulation) {
      runComputeKernelBoundary(D3Q4::Z_AXIS, partition, par, displayQuantity,
                               computeBoundaryStream);
    }

    // Compute inner lattice sites (excluding boundaries)
    if (runSimulation)
      runComputeKernelInterior(partition, par, displayQuantity, computeStream);

    // Gather the plot to draw the display slices
    if (slicePos != Eigen::Vector3i(-1, -1, -1)) {
      par->plot->gatherSlice(slicePos, 0, 0, partitionNoGhostLayer, m_plot,
                             plotStream);
    }

    // Gather averages from GPU array
    if (runSimulation) {
      thrust::device_vector<real>* input =
          par->avg->getDeviceVector(partitionNoGhostLayer);
      thrust::device_vector<real>* output =
          m_avgs->getDeviceVector(m_avgs->getPartition());

      // thrust::gather_if(thrust::cuda::par.on(avgStream),
      // par->avgMap->begin(),
      //                   par->avgMap->end(), par->avgStencil->begin(),
      //                   input->begin(), output->begin());

      dim3 blockSize, gridSize;
      setExtents(par->avgMap->size(), BLOCK_SIZE_DEFAULT, &blockSize,
                 &gridSize);

      GatherKernel<<<gridSize, blockSize, 0, avgStream>>>(
          thrust::raw_pointer_cast(&(*par->avgMap)[0]), par->avgMap->size(),
          thrust::raw_pointer_cast(&(*par->avgStencil)[0]),
          thrust::raw_pointer_cast(&(*input)[0]),
          thrust::raw_pointer_cast(&(*output)[0]));
      CUDA_CHECK_ERRORS("GatherKernel");
      CUDA_RT_CALL(cudaStreamSynchronize(avgStream));

      // TODO(gather_if fails when number of GPUs are 4-9 for some reason...)
      // for (int i = 0; i < par->avgMap->size(); i++) {
      //   int m = (*par->avgMap)[i];
      //   int s = (*par->avgStencil)[i];
      //   if (s) {
      //     real v = (*input)[m];
      //     (*output)[i] = v;
      //   }
      // }

      if (m_resetAvg) {
        CUDA_RT_CALL(cudaStreamSynchronize(avgStream));
        par->avg->fill(0, avgStream);
        par->avg_tmp->fill(0, avgStream);
      }
    }

    // Wait for boundary lattice sites to finish computing
    CUDA_RT_CALL(cudaStreamSynchronize(computeBoundaryStream));

    // Perform ghost layer exchanges
    if (partition.getGhostLayer().x() > 0 && runSimulation) {
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
    if (partition.getGhostLayer().y() > 0 && runSimulation) {
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
    if (partition.getGhostLayer().z() > 0 && runSimulation) {
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
    if (srcDev == 0 && slicePos != Eigen::Vector3i(-1, -1, -1)) {
      real* plot3dPtr = m_plot->gpu_ptr(m_plot->getPartition());
      dim3 blockSize, gridSize;

      setExtents(getExtents().y() * getExtents().z(), BLOCK_SIZE_DEFAULT,
                 &blockSize, &gridSize);
      SliceXRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, getExtents().x(), getExtents().y(), getExtents().z(),
          sliceX, slicePos.x());
      CUDA_CHECK_ERRORS("SliceXRenderKernel");

      setExtents(getExtents().x() * getExtents().z(), BLOCK_SIZE_DEFAULT,
                 &blockSize, &gridSize);
      SliceYRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, getExtents().x(), getExtents().y(), getExtents().z(),
          sliceY, slicePos.y());
      CUDA_CHECK_ERRORS("SliceYRenderKernel");

      setExtents(getExtents().x() * getExtents().y(), BLOCK_SIZE_DEFAULT,
                 &blockSize, &gridSize);
      SliceZRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, getExtents().x(), getExtents().y(), getExtents().z(),
          sliceZ, slicePos.z());
      CUDA_CHECK_ERRORS("SliceZRenderKernel");
    }

    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));
    CUDA_RT_CALL(cudaStreamSynchronize(avgStream));
    CUDA_RT_CALL(cudaStreamSynchronize(plotStream));

#pragma omp barrier
    if (runSimulation) {
      DistributionFunction::swap(par->df, par->df_tmp);
      DistributionFunction::swap(par->dfT, par->dfT_tmp);
      DistributionFunction::swap(par->plot, par->plot_tmp);
      DistributionFunction::swap(par->avg, par->avg_tmp);
    }
  }
  m_resetAvg = false;
}

KernelInterface::KernelInterface(
    const int nx,
    const int ny,
    const int nz,
    const std::shared_ptr<ComputeParams> cmptParams,
    const std::shared_ptr<BoundaryConditions> bcs,
    const std::shared_ptr<VoxelArray> voxels,
    const std::shared_ptr<VoxelVolumeArray> avgVols,
    const int numDevices)
    : P2PLattice(nx, ny, nz, numDevices),
      m_params(numDevices),
      m_resetAvg(false) {
  std::cout << "Initializing LBM data structures..." << std::endl;
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // Arrays for gathering distributed plot with back buffering
  m_plot = new DistributionArray<real>(1, nx, ny, nz);
  m_plot_tmp = new DistributionArray<real>(1, nx, ny, nz);
  m_plot->allocate();
  m_plot_tmp->allocate();
  m_plot->fill(0);
  m_plot_tmp->fill(0);

  // Array for gathering simulation averages
  int volumeVoxels = 0;
  for (int i = 0; i < avgVols->size(); i++) {
    VoxelVolume vol = avgVols->at(i);
    m_avgOffsets[vol] = volumeVoxels;
    Eigen::Vector3i ext = vol.getExtents();
    volumeVoxels += ext.x() * ext.y() * ext.z();
  }
  m_avgs = new DistributionArray<real>(4, volumeVoxels, 0, 0);
  m_avgs->allocate();
  m_avgs->fill(0);

  assert(m_avgs->size(m_avgs->getPartition()) == 4 * volumeVoxels);

  // Create maps and stencils for averaging with gather_if
  std::vector<int>* avgMaps[m_numDevices];
  std::vector<int>* avgStencils[m_numDevices];
  for (int srcDev = 0; srcDev < m_numDevices; srcDev++) {
    avgMaps[srcDev] = new std::vector<int>(4 * volumeVoxels, 0);
    avgStencils[srcDev] = new std::vector<int>(4 * volumeVoxels, 0);
  }
  int voxCount = 0;
  // Loop over all volumes
  for (int i = 0; i < avgVols->size(); i++) {
    VoxelVolume avg = avgVols->at(i);
    // Global minimum and maximum of volumes
    Eigen::Vector3i aMin = avg.getMin();
    Eigen::Vector3i aMax = avg.getMax();

    // Loop over all voxels in volume
    for (int z = aMin.z(); z < aMax.z(); z++)
      for (int y = aMin.y(); y < aMax.y(); y++)
        for (int x = aMin.x(); x < aMax.x(); x++) {
          // Voxel in volume in global coordinates
          Eigen::Vector3i vox = Eigen::Vector3i(x, y, z);
          // Loop over all lattice partitions
          for (int srcDev = 0; srcDev < m_numDevices; srcDev++) {
            const Partition latticePartition = getDevicePartition(srcDev);
            const Partition avgPartition(latticePartition.getMin(),
                                         latticePartition.getMax(),
                                         Eigen::Vector3i(0, 0, 0));

            const Eigen::Vector3i pMin = avgPartition.getMin();
            const Eigen::Vector3i pMax = avgPartition.getMax();
            const Eigen::Vector3i pExtents = avgPartition.getExtents();

            // Check if voxel is inside partition
            if ((pMin.x() <= vox.x() && vox.x() < pMax.x()) &&
                (pMin.y() <= vox.y() && vox.y() < pMax.y()) &&
                (pMin.z() <= vox.z() && vox.z() < pMax.z())) {
              // Convert voxel to local coordinate in partition
              Eigen::Vector3i srcPos = vox - pMin;
              // Loop over temperature (0) and each velocity (1-3)
              for (int q = 0; q < 4; q++) {
                // Convert local coordinate to array index
                int srcIndex = I4D(q, srcPos.x(), srcPos.y(), srcPos.z(),
                                   pExtents.x(), pExtents.y(), pExtents.z());
                int mapIdx = q * volumeVoxels + voxCount;
                avgMaps[srcDev]->at(mapIdx) = srcIndex;
                avgStencils[srcDev]->at(mapIdx) = 1;
                assert(srcIndex > 0 && srcIndex < avgPartition.getSize() * 4);
              }
              // Voxel can only be on one GPU...
              break;
            }
          }
          voxCount++;
        }
  }
  assert(voxCount == volumeVoxels);

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(m_numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeParams* par = new ComputeParams(*cmptParams);
    m_params.at(srcDev) = par;

    // Initialize distribution functions for temperature and velocity
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

    // Arrays for averaging and plotting using back buffering
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), Eigen::Vector3i(0, 0, 0));

    par->avg = new DistributionArray<real>(4, nx, ny, nz, m_numDevices);
    par->avg_tmp = new DistributionArray<real>(4, nx, ny, nz, m_numDevices);
    par->avg->allocate(partitionNoGhostLayer);
    par->avg_tmp->allocate(partitionNoGhostLayer);
    par->avg->fill(0);
    par->avg_tmp->fill(0);

    par->avgMap = new thrust::device_vector<int>(*avgMaps[srcDev]);
    par->avgStencil = new thrust::device_vector<int>(*avgStencils[srcDev]);

    // GPU local plot array with back buffering
    par->plot = new DistributionArray<real>(1, nx, ny, nz, m_numDevices);
    par->plot_tmp = new DistributionArray<real>(1, nx, ny, nz, m_numDevices);
    par->plot->allocate(partitionNoGhostLayer);
    par->plot_tmp->allocate(partitionNoGhostLayer);
    par->plot->fill(0);
    par->plot_tmp->fill(0);

    // Scatter voxel array into partitions
    par->voxels = new VoxelArray(nx, ny, nz, m_numDevices);
    par->voxels->allocate(partitionNoGhostLayer);
    par->voxels->scatter(*voxels, partitionNoGhostLayer);

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
    ComputeParams* par = m_params.at(srcDev);
    *par->bcs = *bcs;
  }
}

void KernelInterface::getMinMax(real* min,
                                real* max,
                                thrust::host_vector<real>* histogram) {
  *min = REAL_MAX;
  *max = REAL_MIN;
  thrust::host_vector<real> mins(m_numDevices);
  thrust::host_vector<real> maxes(m_numDevices);
  thrust::fill(histogram->begin(), histogram->end(), 0.0);

#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const Partition partition = getDevicePartition(srcDev);
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), Eigen::Vector3i(0, 0, 0));
    ComputeParams* par = m_params.at(srcDev);
    mins[srcDev] = par->plot->getMin(partitionNoGhostLayer);
    maxes[srcDev] = par->plot->getMax(partitionNoGhostLayer);
#pragma omp barrier
#pragma omp single
    {
      *max = *thrust::max_element(maxes.begin(), maxes.end());
      *min = *thrust::min_element(mins.begin(), mins.end());
    }
    int nBins = histogram->size();
    thrust::host_vector<int> result(nBins);
    LatticeHistogram lHist;
    thrust::device_vector<real>* input =
        par->plot->getDeviceVector(partitionNoGhostLayer);
    lHist.calculate(input, *min, *max, nBins, &result);
#pragma omp critical
    for (int i = 0; i < nBins; i++) (*histogram)[i] += result[i];
  }
  for (int i = 0; i < histogram->size(); i++) (*histogram)[i] /= getSize();
}

void KernelInterface::resetDfs() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const Partition partition = getDevicePartition(srcDev);
    ComputeParams* par = m_params.at(srcDev);
    runInitKernel(par->df, par->dfT, partition, 1.0, 0, 0, 0, par->Tinit);
    runInitKernel(par->df_tmp, par->dfT_tmp, partition, 1.0, 0, 0, 0,
                  par->Tinit);
  }
}
