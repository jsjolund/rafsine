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
  vector3<int> n = partition.getArrayExtents();
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
    SimulationParams* param,
    SimulationState* state,
    DisplayQuantity::Enum displayQuantity,
    cudaStream_t stream) {
  vector3<int> n = partition.getExtents() - partition.getGhostLayer() * 2;

  real* dfPtr = state->df->gpu_ptr(partition);
  real* df_tmpPtr = state->df_tmp->gpu_ptr(partition);
  real* dfTPtr = state->dfT->gpu_ptr(partition);
  real* dfT_tmpPtr = state->dfT_tmp->gpu_ptr(partition);
  real* dfTeffPtr = state->dfTeff->gpu_ptr(partition);
  real* dfTeff_tmpPtr = state->dfTeff_tmp->gpu_ptr(partition);

  Partition partitionNoGhostLayer(partition.getMin(), partition.getMax(),
                                  vector3<int>(0, 0, 0));
  real* avgSrcPtr = state->avg->gpu_ptr(partitionNoGhostLayer);
  real* avgDstPtr = state->avg_tmp->gpu_ptr(partitionNoGhostLayer);
  real* plotPtr = state->plot_tmp->gpu_ptr(partitionNoGhostLayer);
  voxel_t* voxelPtr = state->voxels->gpu_ptr(partitionNoGhostLayer);

  BoundaryCondition* bcsPtr = thrust::raw_pointer_cast(&(*state->bcs)[0]);

  dim3 gridSize(n.y(), n.z(), 1);
  dim3 blockSize(n.x(), 1, 1);
  if (m_method == LBM::BGK)
    ComputeKernel<LBM::BGK, D3Q4::ORIGIN><<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
        dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C, param->nuT,
        param->Pr_t, param->gBetta, param->Tref, avgSrcPtr, avgDstPtr,
        displayQuantity, plotPtr);
  else if (m_method == LBM::MRT)
    ComputeKernel<LBM::MRT, D3Q4::ORIGIN><<<gridSize, blockSize, 0, stream>>>(
        partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
        dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C, param->nuT,
        param->Pr_t, param->gBetta, param->Tref, avgSrcPtr, avgDstPtr,
        displayQuantity, plotPtr);
  CUDA_CHECK_ERRORS("ComputeKernelInterior");
}

void KernelInterface::runComputeKernelBoundary(
    D3Q4::Enum direction,
    const Partition partition,
    SimulationParams* param,
    SimulationState* state,
    DisplayQuantity::Enum displayQuantity,
    cudaStream_t stream) {
  vector3<int> n = partition.getExtents();

  real* dfPtr = state->df->gpu_ptr(partition);
  real* df_tmpPtr = state->df_tmp->gpu_ptr(partition);
  real* dfTPtr = state->dfT->gpu_ptr(partition);
  real* dfT_tmpPtr = state->dfT_tmp->gpu_ptr(partition);
  real* dfTeffPtr = state->dfTeff->gpu_ptr(partition);
  real* dfTeff_tmpPtr = state->dfTeff_tmp->gpu_ptr(partition);

  Partition partitionNoGhostLayer(partition.getMin(), partition.getMax(),
                                  vector3<int>(0, 0, 0));
  real* avgSrcPtr = state->avg->gpu_ptr(partitionNoGhostLayer);
  real* avgDstPtr = state->avg_tmp->gpu_ptr(partitionNoGhostLayer);
  real* plotPtr = state->plot_tmp->gpu_ptr(partitionNoGhostLayer);
  voxel_t* voxelPtr = state->voxels->gpu_ptr(partitionNoGhostLayer);

  BoundaryCondition* bcsPtr = thrust::raw_pointer_cast(&(*state->bcs)[0]);

  if (direction == D3Q4::X_AXIS) {
    dim3 gridSize(n.z(), 2, 1);
    dim3 blockSize(n.y(), 1, 1);
    if (m_method == LBM::BGK)
      ComputeKernel<LBM::BGK, D3Q4::X_AXIS><<<gridSize, blockSize, 0, stream>>>(
          partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
          dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C,
          param->nuT, param->Pr_t, param->gBetta, param->Tref, avgSrcPtr,
          avgDstPtr, displayQuantity, plotPtr);
    else if (m_method == LBM::MRT)
      ComputeKernel<LBM::MRT, D3Q4::X_AXIS><<<gridSize, blockSize, 0, stream>>>(
          partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
          dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C,
          param->nuT, param->Pr_t, param->gBetta, param->Tref, avgSrcPtr,
          avgDstPtr, displayQuantity, plotPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryX");
  }
  if (direction == D3Q4::Y_AXIS) {
    dim3 gridSize(n.z(), 2, 1);
    dim3 blockSize(n.x(), 1, 1);
    if (m_method == LBM::BGK)
      ComputeKernel<LBM::BGK, D3Q4::Y_AXIS><<<gridSize, blockSize, 0, stream>>>(
          partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
          dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C,
          param->nuT, param->Pr_t, param->gBetta, param->Tref, avgSrcPtr,
          avgDstPtr, displayQuantity, plotPtr);
    else if (m_method == LBM::MRT)
      ComputeKernel<LBM::MRT, D3Q4::Y_AXIS><<<gridSize, blockSize, 0, stream>>>(
          partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
          dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C,
          param->nuT, param->Pr_t, param->gBetta, param->Tref, avgSrcPtr,
          avgDstPtr, displayQuantity, plotPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryY");
  }
  if (direction == D3Q4::Z_AXIS) {
    dim3 gridSize(n.y(), 2, 1);
    dim3 blockSize(n.x(), 1, 1);
    if (m_method == LBM::BGK)
      ComputeKernel<LBM::BGK, D3Q4::Z_AXIS><<<gridSize, blockSize, 0, stream>>>(
          partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
          dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C,
          param->nuT, param->Pr_t, param->gBetta, param->Tref, avgSrcPtr,
          avgDstPtr, displayQuantity, plotPtr);
    else if (m_method == LBM::MRT)
      ComputeKernel<LBM::MRT, D3Q4::Z_AXIS><<<gridSize, blockSize, 0, stream>>>(
          partition, dfPtr, df_tmpPtr, dfTPtr, dfT_tmpPtr, dfTeffPtr,
          dfTeff_tmpPtr, voxelPtr, bcsPtr, m_dt, param->nu, param->C,
          param->nuT, param->Pr_t, param->gBetta, param->Tref, avgSrcPtr,
          avgDstPtr, displayQuantity, plotPtr);
    CUDA_CHECK_ERRORS("ComputeKernelBoundaryZ");
  }
}

std::vector<cudaStream_t> KernelInterface::exchange(int srcDev,
                                                    Partition partition,
                                                    D3Q7::Enum direction) {
  SimulationState* state = m_state.at(srcDev);
  Partition neighbour = state->df_tmp->getNeighbour(partition, direction);
  int dstDev = getPartitionDevice(neighbour);
  cudaStream_t dfStream = getDfGhostLayerStream(srcDev, dstDev);
  cudaStream_t dfTStream = getDfTGhostLayerStream(srcDev, dstDev);
  state->df_tmp->exchange(partition, m_state.at(dstDev)->df_tmp, neighbour,
                          direction, dfStream);
  state->dfT_tmp->exchange(partition, m_state.at(dstDev)->dfT_tmp, neighbour,
                           direction, dfTStream);
  state->dfTeff_tmp->exchange(partition, m_state.at(dstDev)->dfTeff_tmp,
                              neighbour, direction, dfTStream);
  CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
  CUDA_RT_CALL(cudaStreamSynchronize(dfTStream));
  return std::vector<cudaStream_t>{dfStream, dfTStream};
}

void KernelInterface::calculateAverages() {
  thrust::host_vector<real>* avgs =
      m_avgs->getHostVector(m_avgs->getPartition());
  for (int srcDev = 0; srcDev < m_nd; srcDev++) {
    SimulationState* state = m_state.at(srcDev);
    thrust::host_vector<real> avgPartial =
        *state->avgResult->getHostVector(state->avgResult->getPartition());
    thrust::host_vector<int> avgStencil = *state->avgStencil;
    thrust::counting_iterator<int> iter(0);

    thrust::gather_if(iter, iter + avgPartial.size(), avgStencil.begin(),
                      avgPartial.begin(), avgs->begin());
  }
  m_avgs->upload();
}

LatticeAverage KernelInterface::getAverage(VoxelVolume vol,
                                           uint64_t deltaTicks) {
  unsigned int offset = m_avgOffsets[vol];
  unsigned int size = vol.getNumVoxels();
  real ticks = static_cast<real>(deltaTicks);
  Partition partition = m_avgs->getPartition();
  real temperature = m_avgs->getAverage(partition, 0, offset, size, ticks);
  real velocityX = m_avgs->getAverage(partition, 1, offset, size, ticks);
  real velocityY = m_avgs->getAverage(partition, 2, offset, size, ticks);
  real velocityZ = m_avgs->getAverage(partition, 3, offset, size, ticks);
  return LatticeAverage(temperature, velocityX, velocityY, velocityZ);
}

void KernelInterface::compute(DisplayQuantity::Enum displayQuantity,
                              vector3<int> slicePos,
                              real* sliceX,
                              real* sliceY,
                              real* sliceZ,
                              bool runSimulation) {
#pragma omp parallel num_threads(m_nd)
  {
    const int srcDev = omp_get_thread_num() % m_nd;

    CUDA_RT_CALL(cudaSetDevice(srcDev));

    SimulationParams* param = m_params.at(srcDev);
    SimulationState* state = m_state.at(srcDev);
    const Partition partition = getDevicePartition(srcDev);
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), vector3<int>(0, 0, 0));

    const cudaStream_t plotStream = getPlotStream(srcDev);
    const cudaStream_t computeStream = getComputeStream(srcDev);
    const cudaStream_t computeBoundaryStream = getComputeBoundaryStream(srcDev);
    const cudaStream_t avgStream = getAvgStream(srcDev);

    // Compute LBM lattice boundary sites
    if (partition.getGhostLayer().x() > 0 && runSimulation) {
      runComputeKernelBoundary(D3Q4::X_AXIS, partition, param, state,
                               displayQuantity, computeBoundaryStream);
    }
    if (partition.getGhostLayer().y() > 0 && runSimulation) {
      runComputeKernelBoundary(D3Q4::Y_AXIS, partition, param, state,
                               displayQuantity, computeBoundaryStream);
    }
    if (partition.getGhostLayer().z() > 0 && runSimulation) {
      runComputeKernelBoundary(D3Q4::Z_AXIS, partition, param, state,
                               displayQuantity, computeBoundaryStream);
    }

    // Compute inner lattice sites (excluding boundaries)
    if (runSimulation)
      runComputeKernelInterior(partition, param, state, displayQuantity,
                               computeStream);

    // Gather the plot to draw the display slices
    if (slicePos != vector3<int>(-1, -1, -1)) {
      state->plot->gatherSlice(slicePos, 0, 0, partitionNoGhostLayer, m_plot,
                               plotStream);
    }

    // Gather averages from GPU array
    if (runSimulation) {
      thrust::device_vector<real>* input =
          state->avg->getDeviceVector(partitionNoGhostLayer);
      thrust::device_vector<real>* output =
          state->avgResult->getDeviceVector(state->avgResult->getPartition());
      thrust::gather(thrust::cuda::par.on(avgStream), state->avgMap->begin(),
                     state->avgMap->end(), input->begin(), output->begin());
      state->avgResult->download();
      if (m_resetAvg) {
        state->avg->fill(0, avgStream);
        state->avg_tmp->fill(0, avgStream);
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
    if (srcDev == 0 && slicePos != vector3<int>(-1, -1, -1)) {
      real* plot3dPtr = m_plot->gpu_ptr(m_plot->getPartition());
      dim3 blockSize, gridSize;

      vector3<int> n = getExtents();
      setExtents(n.y() * n.z(), BLOCK_SIZE_DEFAULT, &blockSize, &gridSize);
      SliceXRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, n.x(), n.y(), n.z(), sliceX, slicePos.x());
      CUDA_CHECK_ERRORS("SliceXRenderKernel");

      setExtents(n.x() * n.z(), BLOCK_SIZE_DEFAULT, &blockSize, &gridSize);
      SliceYRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, n.x(), n.y(), n.z(), sliceY, slicePos.y());
      CUDA_CHECK_ERRORS("SliceYRenderKernel");

      setExtents(n.x() * n.y(), BLOCK_SIZE_DEFAULT, &blockSize, &gridSize);
      SliceZRenderKernel<<<gridSize, blockSize, 0, plotStream>>>(
          plot3dPtr, n.x(), n.y(), n.z(), sliceZ, slicePos.z());
      CUDA_CHECK_ERRORS("SliceZRenderKernel");
    }

    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));
    CUDA_RT_CALL(cudaStreamSynchronize(avgStream));
    CUDA_RT_CALL(cudaStreamSynchronize(plotStream));

#pragma omp barrier
    if (runSimulation) {
      DistributionFunction::swap(state->df, state->df_tmp);
      DistributionFunction::swap(state->dfT, state->dfT_tmp);
      DistributionFunction::swap(state->dfTeff, state->dfTeff_tmp);
      DistributionFunction::swap(state->plot, state->plot_tmp);
      DistributionFunction::swap(state->avg, state->avg_tmp);
    }
  }
  m_resetAvg = false;
}

KernelInterface::KernelInterface(
    const int nx,
    const int ny,
    const int nz,
    const real dt,
    const std::shared_ptr<SimulationParams> cmptParams,
    const std::shared_ptr<BoundaryConditions> bcs,
    const std::shared_ptr<VoxelArray> voxels,
    const std::shared_ptr<VoxelVolumeArray> avgVols,
    const int nd,
    const LBM::Enum method,
    const D3Q4::Enum partitioning)
    : P2PLattice(nx, ny, nz, nd, partitioning),
      m_params(nd),
      m_state(nd),
      m_method(method),
      m_resetAvg(false),
      m_dt(dt) {
  std::cout << "Initializing LBM data structures..." << std::endl;
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // Arrays for gathering distributed plot with back buffering
  m_plot = new DistributionArray<real>(1, nx, ny, nz, 1, 0, partitioning);
  m_plot_tmp = new DistributionArray<real>(1, nx, ny, nz, 1, 0, partitioning);
  m_plot->allocate();
  m_plot_tmp->allocate();
  m_plot->fill(0);
  m_plot_tmp->fill(0);

  // Array for gathering simulation averages
  int numAvgVoxels = 0;
  for (int i = 0; i < avgVols->size(); i++) {
    VoxelVolume vol = avgVols->at(i);
    m_avgOffsets[vol] = numAvgVoxels;
    vector3<int> ext = vol.getExtents();
    numAvgVoxels += ext.x() * ext.y() * ext.z();
  }
  m_avgs =
      new DistributionArray<real>(4, numAvgVoxels, 0, 0, 1, 0, partitioning);
  m_avgs->allocate();
  m_avgs->fill(0);

  // Create maps and stencils for averaging with gather_if
  std::vector<int>* avgMaps[nd];
  std::vector<int>* avgStencils[nd];
  for (int srcDev = 0; srcDev < nd; srcDev++) {
    avgMaps[srcDev] = new std::vector<int>(4 * numAvgVoxels, 0);
    avgStencils[srcDev] = new std::vector<int>(4 * numAvgVoxels, 0);
  }
  int voxCounter = 0;
  // Loop over all volumes
  for (int i = 0; i < avgVols->size(); i++) {
    VoxelVolume avg = avgVols->at(i);
    // Global minimum and maximum of volumes
    vector3<int> aMin = avg.getMin();
    vector3<int> aMax = avg.getMax();

    // Loop over all voxels in volume
    for (int z = aMin.z(); z < aMax.z(); z++)
      for (int y = aMin.y(); y < aMax.y(); y++)
        for (int x = aMin.x(); x < aMax.x(); x++) {
          // Voxel in volume in global coordinates
          vector3<int> vox = vector3<int>(x, y, z);
          // Loop over all lattice partitions
          for (int srcDev = 0; srcDev < nd; srcDev++) {
            const Partition latticePartition = getDevicePartition(srcDev);
            const Partition avgPartition(latticePartition.getMin(),
                                         latticePartition.getMax(),
                                         vector3<int>(0, 0, 0));

            const vector3<int> pMin = avgPartition.getMin();
            const vector3<int> pMax = avgPartition.getMax();
            const vector3<int> pExt = avgPartition.getExtents();

            // Check if voxel is inside partition
            if ((pMin.x() <= vox.x() && vox.x() < pMax.x()) &&
                (pMin.y() <= vox.y() && vox.y() < pMax.y()) &&
                (pMin.z() <= vox.z() && vox.z() < pMax.z())) {
              // Convert voxel to local coordinate in partition
              vector3<int> srcPos = vox - pMin;
              // Loop over temperature (0) and each velocity (1-3)
              for (int q = 0; q < 4; q++) {
                // Convert local coordinate to array index
                int srcIndex = I4D(q, srcPos.x(), srcPos.y(), srcPos.z(),
                                   pExt.x(), pExt.y(), pExt.z());
                int mapIdx = q * numAvgVoxels + voxCounter;
                avgMaps[srcDev]->at(mapIdx) = srcIndex;
                avgStencils[srcDev]->at(mapIdx) = 1;
                assert(srcIndex > 0 && srcIndex < avgPartition.getSize() * 4);
              }
              // Voxel can only be on one GPU...
              break;
            }
          }
          voxCounter++;
        }
  }
  assert(voxCounter == numAvgVoxels);

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(nd)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    SimulationParams* param = new SimulationParams(*cmptParams);
    m_params.at(srcDev) = param;
    SimulationState* state = new SimulationState();
    m_state.at(srcDev) = state;

    // Initialize distribution functions for temperature and velocity
    const Partition partition = getDevicePartition(srcDev);

    state->df = new DistributionFunction(19, nx, ny, nz, nd, partitioning);
    state->df_tmp = new DistributionFunction(19, nx, ny, nz, nd, partitioning);
    state->dfT = new DistributionFunction(7, nx, ny, nz, nd, partitioning);
    state->dfT_tmp = new DistributionFunction(7, nx, ny, nz, nd, partitioning);
    state->dfTeff = new DistributionFunction(1, nx, ny, nz, nd, partitioning);
    state->dfTeff_tmp =
        new DistributionFunction(1, nx, ny, nz, nd, partitioning);

    state->df->allocate(partition);
    state->df_tmp->allocate(partition);
    state->dfT->allocate(partition);
    state->dfT_tmp->allocate(partition);
    state->dfTeff->allocate(partition);
    state->dfTeff_tmp->allocate(partition);

    runInitKernel(state->df, state->dfT, partition, 1.0, 0, 0, 0, param->Tinit);
    runInitKernel(state->df_tmp, state->dfT_tmp, partition, 1.0, 0, 0, 0,
                  param->Tinit);
    state->dfTeff->fill(param->Tinit);
    state->dfTeff_tmp->fill(param->Tinit);
    ss << "Allocated partition " << partition << " on GPU" << srcDev
       << std::endl;

    // Arrays for averaging and plotting using back buffering
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), vector3<int>(0, 0, 0));

    state->avg =
        new DistributionArray<real>(4, nx, ny, nz, nd, 0, partitioning);
    state->avg_tmp =
        new DistributionArray<real>(4, nx, ny, nz, nd, 0, partitioning);
    state->avg->allocate(partitionNoGhostLayer);
    state->avg_tmp->allocate(partitionNoGhostLayer);
    state->avg->fill(0);
    state->avg_tmp->fill(0);

    state->avgMap = new thrust::device_vector<int>(*avgMaps[srcDev]);
    state->avgStencil = new thrust::host_vector<int>(*avgStencils[srcDev]);

    state->avgResult =
        new DistributionArray<real>(4, numAvgVoxels, 0, 0, 1, 0, partitioning);
    state->avgResult->allocate();
    state->avgResult->fill(0);
    assert(state->avgResult->size(state->avgResult->getPartition()) ==
           4 * numAvgVoxels);

    // GPU local plot array with back buffering
    state->plot =
        new DistributionArray<real>(1, nx, ny, nz, nd, 0, partitioning);
    state->plot_tmp =
        new DistributionArray<real>(1, nx, ny, nz, nd, 0, partitioning);
    state->plot->allocate(partitionNoGhostLayer);
    state->plot_tmp->allocate(partitionNoGhostLayer);
    state->plot->fill(0);
    state->plot_tmp->fill(0);

    // Scatter voxel array into partitions
    state->voxels = new VoxelArray(nx, ny, nz, nd, partitioning);
    state->voxels->allocate(partitionNoGhostLayer);
    state->voxels->scatter(*voxels, partitionNoGhostLayer);

    // Upload boundary conditions array
    state->bcs = new thrust::device_vector<BoundaryCondition>(*bcs);

    CUDA_RT_CALL(cudaDeviceSynchronize());
    std::cout << ss.str();
  }  // end omp parallel

  std::cout << "LBM initialized" << std::endl;
}

void KernelInterface::uploadBCs(std::shared_ptr<BoundaryConditions> bcs) {
#pragma omp parallel num_threads(m_nd)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    SimulationState* state = m_state.at(srcDev);
    *state->bcs = *bcs;
  }
}

void KernelInterface::getMinMax(real* min,
                                real* max,
                                thrust::host_vector<real>* histogram) {
  // *min = 20.0f;
  // *max = 30.0f;
  *min = REAL_MAX;
  *max = REAL_MIN;
  thrust::host_vector<real> mins(m_nd);
  thrust::host_vector<real> maxes(m_nd);
  thrust::fill(histogram->begin(), histogram->end(), 0.0);

#pragma omp parallel num_threads(m_nd)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const Partition partition = getDevicePartition(srcDev);
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), vector3<int>(0, 0, 0));
    SimulationState* state = m_state.at(srcDev);
    mins[srcDev] = state->plot->getMin(partitionNoGhostLayer);
    maxes[srcDev] = state->plot->getMax(partitionNoGhostLayer);
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
        state->plot->getDeviceVector(partitionNoGhostLayer);
    lHist.calculate(input, *min, *max, nBins, &result);
#pragma omp critical
    for (int i = 0; i < nBins; i++) (*histogram)[i] += result[i];
  }
  for (int i = 0; i < histogram->size(); i++) (*histogram)[i] /= getSize();
}

void KernelInterface::resetDfs() {
#pragma omp parallel num_threads(m_nd)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const Partition partition = getDevicePartition(srcDev);
    SimulationParams* param = m_params.at(srcDev);
    SimulationState* state = m_state.at(srcDev);
    runInitKernel(state->df, state->dfT, partition, 1.0, 0, 0, 0, param->Tinit);
    runInitKernel(state->df_tmp, state->dfT_tmp, partition, 1.0, 0, 0, 0,
                  param->Tinit);
    state->dfTeff->fill(param->Tinit);
    state->dfTeff_tmp->fill(param->Tinit);
  }
}
