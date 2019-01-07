#include "KernelInterface.hpp"

void KernelInterface::runInitKernel(DistributionFunction *df,
                                    DistributionFunction *dfT,
                                    Partition partition, float rho, float vx,
                                    float vy, float vz, float T) {
  /// Initialise distribution functions on the GPU
  float sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
  glm::ivec3 n = partition.getQDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);
  real *dfPtr = df->gpu_ptr(partition);
  real *dfTPtr = dfT->gpu_ptr(partition);
  InitKernel<<<gridSize, blockSize>>>(dfPtr, dfTPtr, n.x, n.y, n.z, rho, vx, vy,
                                      vz, T, sq_term);

  CUDA_CHECK_ERRORS("InitKernel");
}

void KernelInterface::runComputeKernel(Partition partition,
                                       ComputeKernelParams *kp,
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
    ComputeKernelParams *kp = m_computeParams.at(srcDev);
    Partition partition = getPartitionFromDevice(srcDev);

    cudaStream_t computeStream = getP2Pstream(srcDev, srcDev);
    runComputeKernel(partition, kp, plotGpuPointer, displayQuantity,
                     computeStream);
    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));
<<<<<<< HEAD
    
=======
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
#pragma omp barrier

    cudaStream_t dfStream = m_deviceParams.at(srcDev)->dfExchangeStream;
    DistributionFunction *df = kp->df_tmp;
<<<<<<< HEAD
    haloExchange(Partition::Enum::X_AXIS,
    
    );
=======

    // {
    //   cudaStream_t dfStream = m_deviceParams.at(srcDev)->dfExchangeStream;
    //   DistributionFunction *df = kp->df_tmp;
    //   int nNeighbours = df->getQ();
    //   for (int q = 0; q < nNeighbours; q++) {
    //     Partition neighbour = df->getNeighbour(partition, q);
    //     const int dstDev = m_partitionDeviceMap[neighbour];
    //     DistributionFunction *ndf = m_computeParams.at(dstDev)->df_tmp;

    //     std::vector<PartitionSegment> segments =
    //         df->m_segments[partition][neighbour];
    //     for (PartitionSegment segment : segments) {
    //       if (segment.m_segmentLength <= 0) continue;
    //       real *dfPtr = df->gpu_ptr(partition, segment.m_src.w,
    //       segment.m_src.x,
    //                                 segment.m_src.y, segment.m_src.z);
    //       real *ndfPtr =
    //           ndf->gpu_ptr(neighbour, segment.m_dst.w, segment.m_dst.x,
    //                        segment.m_dst.y, segment.m_dst.z);

    //       CUDA_RT_CALL(cudaMemcpy2DAsync(
    //           ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
    //           segment.m_segmentLength, segment.m_numSegments,
    //           cudaMemcpyDefault, dfStream));
    //     }
    //   }
    // }
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12

    {
      Partition neighbour = df->getNeighbour(partition, 1);
      DistributionFunction *ndf =
<<<<<<< HEAD
          m_computeParams.at(getDeviceFromPartition(neighbour))->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];
      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[0][i];
        int qDst = D3Q27ranks[1][i];
        if (qSrc >= df->getQ()) break;
=======
          m_computeParams.at(m_partitionDeviceMap[neighbour])->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];

      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[0][i];
        int qDst = D3Q27ranks[1][i];
        if (qSrc > df->getQ()) break;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
        PartitionSegment segment = segments[qSrc];
        real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z, true);
        real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z, true);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfStream));
      }
    }
    {
      Partition neighbour = df->getNeighbour(partition, 2);
      DistributionFunction *ndf =
<<<<<<< HEAD
          m_computeParams.at(getDeviceFromPartition(neighbour))->df_tmp;
=======
          m_computeParams.at(m_partitionDeviceMap[neighbour])->df_tmp;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];
      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[1][i];
        int qDst = D3Q27ranks[0][i];
<<<<<<< HEAD
        if (qSrc >= df->getQ()) break;
=======
        if (qSrc > df->getQ()) break;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
        PartitionSegment segment = segments[qSrc];
        real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z, true);
        real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z, true);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfStream));
      }
    }
    CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
#pragma omp barrier

    {
      Partition neighbour = df->getNeighbour(partition, 3);
      DistributionFunction *ndf =
<<<<<<< HEAD
          m_computeParams.at(getDeviceFromPartition(neighbour))->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];
      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[2][i];
        int qDst = D3Q27ranks[3][i];
        if (qSrc >= df->getQ()) break;
=======
          m_computeParams.at(m_partitionDeviceMap[neighbour])->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];

      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[2][i];
        int qDst = D3Q27ranks[3][i];
        if (qSrc > df->getQ()) break;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
        PartitionSegment segment = segments[qSrc];
        real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z, true);
        real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z, true);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfStream));
      }
    }
    {
      Partition neighbour = df->getNeighbour(partition, 4);
      DistributionFunction *ndf =
<<<<<<< HEAD
          m_computeParams.at(getDeviceFromPartition(neighbour))->df_tmp;
=======
          m_computeParams.at(m_partitionDeviceMap[neighbour])->df_tmp;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];
      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[3][i];
        int qDst = D3Q27ranks[2][i];
<<<<<<< HEAD
        if (qSrc >= df->getQ()) break;
=======
        if (qSrc > df->getQ()) break;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
        PartitionSegment segment = segments[qSrc];
        real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z, true);
        real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z, true);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfStream));
      }
    }
    CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
#pragma omp barrier

    {
      Partition neighbour = df->getNeighbour(partition, 5);
      DistributionFunction *ndf =
<<<<<<< HEAD
          m_computeParams.at(getDeviceFromPartition(neighbour))->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];
      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[4][i];
        int qDst = D3Q27ranks[5][i];
        if (qSrc >= df->getQ()) break;
=======
          m_computeParams.at(m_partitionDeviceMap[neighbour])->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];

      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[4][i];
        int qDst = D3Q27ranks[5][i];
        if (qSrc > df->getQ()) break;
        PartitionSegment segment = segments[qSrc];
        real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z, true);
        real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z, true);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfStream));
      }
    }
    {
      Partition neighbour = df->getNeighbour(partition, 6);
      DistributionFunction *ndf =
          m_computeParams.at(m_partitionDeviceMap[neighbour])->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];
      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[5][i];
        int qDst = D3Q27ranks[4][i];
        if (qSrc > df->getQ()) break;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
        PartitionSegment segment = segments[qSrc];
        real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z, true);
        real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z, true);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfStream));
      }
    }
<<<<<<< HEAD
    {
      Partition neighbour = df->getNeighbour(partition, 6);
      DistributionFunction *ndf =
          m_computeParams.at(getDeviceFromPartition(neighbour))->df_tmp;
      std::vector<PartitionSegment> segments =
          df->m_segments[partition][neighbour];
      for (int i = 0; i < 9; i++) {
        int qSrc = D3Q27ranks[5][i];
        int qDst = D3Q27ranks[4][i];
        if (qSrc >= df->getQ()) break;
        PartitionSegment segment = segments[qSrc];
        real *dfPtr = df->gpu_ptr(partition, qSrc, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z, true);
        real *ndfPtr = ndf->gpu_ptr(neighbour, qDst, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z, true);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfStream));
      }
    }
=======
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
    CUDA_RT_CALL(cudaStreamSynchronize(dfStream));
    CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
    DistributionFunction::swap(kp->df, kp->df_tmp);
    DistributionFunction::swap(kp->dfT, kp->dfT_tmp);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));
}

KernelInterface::KernelInterface(const ComputeKernelParams *params,
                                 const BoundaryConditionsArray *bcs,
                                 const VoxelArray *voxels,
                                 const int numDevices = 1)
<<<<<<< HEAD
    : DistributedLattice(numDevices, params->nx, params->ny, params->nz),
      m_computeParams(numDevices) {
=======
    : m_numDevices(numDevices),
      m_devicePartitionMap(numDevices),
      m_computeParams(numDevices),
      m_deviceParams(numDevices) {
  glm::ivec3 n = glm::ivec3(params->nx, params->ny, params->nz);
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  std::cout << "Domain size : (" << n.x << ", " << n.y << ", " << n.z << ")"
            << std::endl
            << "Total number of nodes : " << n.x * n.y * n.z << std::endl
            << "Number of devices: " << m_numDevices << std::endl;
  {
    DistributionFunction df(19, n.x, n.y, n.z, m_numDevices);
    std::vector<Partition> partitions = df.getPartitions();
    for (int i = 0; i < partitions.size(); i++) {
      Partition partition = partitions.at(i);
      // Distribute the workload. Calculate partitions and assign them to GPUs
      int devIndex = i % m_numDevices;
      m_partitionDeviceMap[partition] = devIndex;
      m_devicePartitionMap.at(devIndex) = Partition(partition);
    }
  }

  std::cout << "Starting GPU threads" << std::endl;
>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    std::stringstream ss;

    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    ComputeKernelParams *kp = new ComputeKernelParams();
    m_computeParams.at(srcDev) = kp;
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

<<<<<<< HEAD
=======
    DeviceParams *dp = new DeviceParams(numDevices);
    m_deviceParams.at(srcDev) = dp;

    // Enable P2P access between GPUs
    int nNeighbours = kp->df->getQ();
    for (int q = 0; q < nNeighbours; q++) {
      Partition neighbour = kp->df->getNeighbour(partition, q);
      const int dstDev = m_partitionDeviceMap[neighbour];
      enablePeerAccess(srcDev, dstDev, &dp->peerAccessList);
    }
    // All GPUs need access to the rendering GPU0
    enablePeerAccess(srcDev, 0, &dp->peerAccessList);

    // Create non-blocking streams
    CUDA_RT_CALL(
        cudaStreamCreateWithFlags(&dp->computeStream, cudaStreamDefault));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&dp->dfExchangeStream,
                                           cudaStreamNonBlocking));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&dp->dfTExchangeStream,
                                           cudaStreamNonBlocking));

// Wait until all threads have finished constructing computeParams
#pragma omp barrier

>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
    std::cout << ss.str();
  }  // end omp parallel num_threads(numDevices)
  std::cout << "GPU configuration complete" << std::endl;
}

<<<<<<< HEAD
=======
bool enablePeerAccess(int srcDev, int dstDev,
                      std::vector<bool> *peerAccessList) {
  std::ostringstream ss;
  if (srcDev == dstDev || peerAccessList->at(dstDev)) {
    peerAccessList->at(srcDev) = true;
    return false;
  } else if (!peerAccessList->at(dstDev)) {
    int cudaCanAccessPeer = 0;
    CUDA_RT_CALL(cudaDeviceCanAccessPeer(&cudaCanAccessPeer, srcDev, dstDev));
    if (cudaCanAccessPeer) {
      CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
      peerAccessList->at(dstDev) = true;
      ss << "Enabled P2P from GPU" << srcDev << " to GPU" << dstDev
         << std::endl;
    } else {
      ss << "ERROR: Failed to enable P2P from GPU" << srcDev << " to GPU"
         << dstDev << std::endl;
      throw std::runtime_error(ss.str());
    }
  }
  std::cout << ss.str();
  return peerAccessList->at(dstDev);
}

void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList) {
  std::ostringstream ss;
  for (int dstDev = 0; dstDev < peerAccessList->size(); dstDev++) {
    if (dstDev != srcDev && peerAccessList->at(dstDev)) {
      CUDA_RT_CALL(cudaDeviceDisablePeerAccess(dstDev));
      peerAccessList->at(dstDev) = false;
      ss << "Disabled P2P from GPU" << srcDev << " to GPU" << dstDev
         << std::endl;
    }
  }
  std::cout << ss.str();
}

>>>>>>> ab3319b44635e5115e000055d3c56e8944800e12
void KernelInterface::uploadBCs(BoundaryConditionsArray *bcs) {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    ComputeKernelParams *kp = m_computeParams.at(srcDev);
    *kp->bcs = *bcs;
  }
}

void KernelInterface::resetAverages() {
#pragma omp parallel num_threads(m_numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
    ComputeKernelParams *kp = m_computeParams.at(srcDev);
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
    ComputeKernelParams *kp = m_computeParams.at(srcDev);
    runInitKernel(kp->df, kp->dfT, partition, 1.0, 0, 0, 0, kp->Tinit);
    runInitKernel(kp->df_tmp, kp->dfT_tmp, partition, 1.0, 0, 0, 0, kp->Tinit);
  }
}
