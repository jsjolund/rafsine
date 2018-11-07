#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "DistributedDFGroup.hpp"

// Reference for empty DF group
static real pEmpty[4][4][4] = {{                 //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0}},   //
                               {                 //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0}},   //
                               {                 //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0}},   //
                               {                 //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0},    //
                                {0, 0, 0, 0}}};  //
// Reference for initial values in lattice, before halo exchange
static real pBefore[4][4][4] = {{                 //
                                 {0, 0, 0, 0},    //
                                 {0, 0, 0, 0},    //
                                 {0, 0, 0, 0},    //
                                 {0, 0, 0, 0}},   //
                                {                 //
                                 {0, 0, 0, 0},    //
                                 {0, 1, 2, 0},    //
                                 {0, 3, 4, 0},    //
                                 {0, 0, 0, 0}},   //
                                {                 //
                                 {0, 0, 0, 0},    //
                                 {0, 5, 6, 0},    //
                                 {0, 7, 8, 0},    //
                                 {0, 0, 0, 0}},   //
                                {                 //
                                 {0, 0, 0, 0},    //
                                 {0, 0, 0, 0},    //
                                 {0, 0, 0, 0},    //
                                 {0, 0, 0, 0}}};  //
// Reference for values after halo exchange
static real pAfter[4][4][4] = {{                 //
                                {8, 7, 8, 7},    //
                                {6, 5, 6, 5},    //
                                {8, 7, 8, 7},    //
                                {6, 5, 6, 5}},   //
                               {                 //
                                {4, 3, 4, 3},    //
                                {2, 1, 2, 1},    //
                                {4, 3, 4, 3},    //
                                {2, 1, 2, 1}},   //
                               {                 //
                                {8, 7, 8, 7},    //
                                {6, 5, 6, 5},    //
                                {8, 7, 8, 7},    //
                                {6, 5, 6, 5}},   //
                               {                 //
                                {4, 3, 4, 3},    //
                                {2, 1, 2, 1},    //
                                {4, 3, 4, 3},    //
                                {2, 1, 2, 1}}};  //

/**
 * @brief Compare a partition with a reference array
 */
template <size_t nx, size_t ny, size_t nz>
static bool comparePartitions(DistributedDFGroup *df, Partition *p0,
                              real (&ref)[nx][ny][nz]) {
  glm::ivec3 min = p0->getLatticeMin() - glm::ivec3(1, 1, 1);
  glm::ivec3 max = p0->getLatticeMax() + glm::ivec3(1, 1, 1);
  for (int hq = 0; hq < df->getQ(); hq++, hq++)
    for (int hz = min.z, rz = 0; hz < max.z; hz++, rz++)
      for (int hy = min.y, ry = 0; hy < max.y; hy++, ry++)
        for (int hx = min.x, rx = 0; hx < max.x; hx++, rx++) {
          real a = ref[rz][ry][rx];
          real b = (*df)(*p0, hq, hx, hy, hz);
          EXPECT_EQ(a, b);
          if (a != b) {
            return false;
          }
        }
  return true;
}

/**
 * @brief Simple kernel which puts sequential numbers on non-halo positions
 */
__global__ void TestKernel(real *__restrict__ df, glm::ivec3 pMin,
                           glm::ivec3 pMax) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y;
  glm::ivec3 p(x, y, z);
  glm::ivec3 dfSize = pMax - pMin;
  if ((p.x >= dfSize.x) || (p.y >= dfSize.y) || (p.z >= dfSize.z)) return;
  real value = 1 + I3D(x, y, z, dfSize.x, dfSize.y, dfSize.z);
  glm::ivec3 arrSize = dfSize + glm::ivec3(2, 2, 2);
  df[I4D(0, p.x, p.y, p.z, arrSize.x, arrSize.y, arrSize.z)] = value;
}

/**
 * @brief Launcher for the test kernel
 */
void runTestKernel(DistributedDFGroup *df, Partition partition,
                   cudaStream_t stream) {
  glm::ivec3 n = partition.getLatticeDims();
  dim3 gridSize(n.y + 2, n.z + 2, 1);
  dim3 blockSize(n.x + 2, 1, 1);
  glm::ivec3 p = partition.getLatticeMin();
  for (int q = 0; q < df->getQ(); q++)
    TestKernel<<<gridSize, blockSize, 0, stream>>>(
        df->gpu_ptr(partition, q, p.x, p.y, p.z), partition.getLatticeMin(),
        partition.getLatticeMax());
}

TEST(DistributedDFTest, HaloExchangeCPU) {
  int nq = 1, nx = 2, ny = 2, nz = 4, divisions = 1;
  DistributedDFGroup *df = new DistributedDFGroup(nq, nx, ny, nz, divisions);

  std::vector<Partition *> partitions = df->getPartitions();
  for (Partition *p : partitions) {
    df->allocate(*p);
  }
  df->fill(0, 0);

  int i = 0;
  for (int q = 0; q < nq; ++q)
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          (*df)(q, x, y, z) = 1 + (i++ % 8);
        }
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pBefore));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pBefore));

  for (std::pair<Partition, std::vector<HaloExchangeData>> element :
       df->m_haloData) {
    Partition *partition = &element.first;
    std::vector<HaloExchangeData> neighbours = element.second;

    for (int i = 0; i < neighbours.size(); i++) {
      Partition *neighbour = neighbours.at(i).neighbour;

      std::vector<glm::ivec3> pSrc, nSrc, pDst, nDst;
      glm::ivec3 direction = D3Q19directionVectors[i];

      partition->getHalo(direction, &pSrc, &nDst);
      neighbour->getHalo(-direction, &nSrc, &pDst);
      ASSERT_EQ(pSrc.size(), nDst.size());
      ASSERT_EQ(pSrc.size(), nSrc.size());
      ASSERT_EQ(pSrc.size(), pDst.size());

      for (int j = 0; j < pSrc.size(); j++) {
        glm::ivec3 src = pSrc.at(j);
        glm::ivec3 dst = pDst.at(j);
        for (int q = 0; q < nq; ++q) {
          (*df)(*neighbour, q, dst.x, dst.y, dst.z) =
              (*df)(*partition, q, src.x, src.y, src.z);
        }
      }
    }
  }
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pAfter));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pAfter));
}

TEST(DistributedDFTest, SingleGPUKernelPartition) {
  const int nq = 1, nx = 2, ny = 2, nz = 4, divisions = 1;
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributedDFGroup *df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
  for (Partition *partition : df->getPartitions()) df->allocate(*partition);
  for (int q = 0; q < nq; q++) df->fill(q, 0);
  df->upload();
  cudaStream_t computeStream;
  CUDA_RT_CALL(cudaStreamCreate(&computeStream));
  std::vector<Partition *> partitions = df->getPartitions();
  for (Partition *partition : partitions) {
    runTestKernel(df, *partition, computeStream);
  }
  df->download();
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pBefore));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pBefore));
  CUDA_RT_CALL(cudaStreamDestroy(computeStream));
  CUDA_RT_CALL(cudaDeviceReset());
}

TEST(DistributedDFTest, SingleGPUKernelSwapAndEquals) {
  const int nq = 1, nx = 2, ny = 2, nz = 4, divisions = 1;
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributedDFGroup *df, *dfTmp;
  df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
  dfTmp = new DistributedDFGroup(nq, nx, ny, nz, divisions);
  std::vector<Partition *> partitions = df->getPartitions();
  for (Partition *partition : partitions) {
    df->allocate(*partition);
    dfTmp->allocate(*partition);
  }
  for (int q = 0; q < nq; q++) {
    df->fill(q, 0);
    dfTmp->fill(q, 0);
  }
  df->upload();
  dfTmp->upload();
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pEmpty));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pEmpty));
  ASSERT_TRUE(comparePartitions(dfTmp, partitions.at(0), pEmpty));
  ASSERT_TRUE(comparePartitions(dfTmp, partitions.at(1), pEmpty));
  cudaStream_t computeStream;
  CUDA_RT_CALL(cudaStreamCreate(&computeStream));
  for (Partition *partition : partitions) {
    runTestKernel(df, *partition, computeStream);
  }
  DistributedDFGroup::swap(df, dfTmp);
  df->download();
  dfTmp->download();
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pEmpty));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pEmpty));
  ASSERT_TRUE(comparePartitions(dfTmp, partitions.at(0), pBefore));
  ASSERT_TRUE(comparePartitions(dfTmp, partitions.at(1), pBefore));
  df = dfTmp;
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pBefore));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pBefore));
  ASSERT_TRUE(comparePartitions(dfTmp, partitions.at(0), pBefore));
  ASSERT_TRUE(comparePartitions(dfTmp, partitions.at(1), pBefore));
  CUDA_RT_CALL(cudaStreamDestroy(computeStream));
  CUDA_RT_CALL(cudaDeviceReset());
}

TEST(DistributedDFTest, HaloExchangeMultiGPU) {
  int numDevices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, 8);  // Limit to 2 for this test
  // ASSERT_EQ(numDevices, 2);
  CUDA_RT_CALL(cudaSetDevice(0));

  // Create more or equal number of partitions as there are GPUs
  int nq = 3, nx = 4, ny = 4, nz = 4, divisions = 0;
  while (1 << divisions < numDevices) divisions++;

  // Create as many DF groups as there are GPUs
  DistributedDFGroup *dfs[numDevices];

  // Distribute the workload
  std::unordered_map<Partition, int> partitionDeviceMap;
  std::vector<std::vector<Partition>> devicePartitionMap(numDevices);

  // Calculate partitions and assign them to GPUs
  DistributedDFGroup *masterDf =
      new DistributedDFGroup(nq, nx, ny, nz, divisions);
  std::vector<Partition *> partitions = masterDf->getPartitions();
  for (int i = 0; i < partitions.size(); i++) {
    Partition partition = *partitions.at(i);
    // masterDf->allocate(partition);

    int devIndex = i % numDevices;
    partitionDeviceMap[partition] = devIndex;
    devicePartitionMap.at(devIndex).push_back(partition);
  }
  bool success = true;

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    for (int dstDev = 0; dstDev < numDevices; dstDev++) {
      if (dstDev != srcDev) {
        int canAccessPeer = 0;
        cudaError_t cudaPeerAccessStatus;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, srcDev, dstDev));
        if (canAccessPeer) {
          cudaPeerAccessStatus = cudaDeviceEnablePeerAccess(dstDev, 0);
        }
        if (!cudaDeviceCanAccessPeer || cudaPeerAccessStatus != cudaSuccess) {
#pragma omp critical
          {
            if (success) success = false;
          }
        }
      }
    }

    // Setup streams
    int priorityHigh, priorityLow;
    cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
    cudaStream_t computeStream;
    cudaStreamCreateWithPriority(&computeStream, cudaStreamNonBlocking,
                                 priorityHigh);
    std::vector<cudaStream_t> cpyStreams(numDevices);
    for (int i = 0; i < numDevices; i++)
      cudaStreamCreateWithPriority(&cpyStreams[i], cudaStreamNonBlocking,
                                   priorityLow);
#pragma omp barrier
    DistributedDFGroup *df;
    if (srcDev == 0) {
      df = masterDf;
      // for (Partition *partition : df->getPartitions())
      // df->allocate(*partition);
    } else {
      df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
    }
    for (Partition partition : devicePartitionMap.at(srcDev))
      df->allocate(partition);
    dfs[srcDev] = df;
    for (int q = 0; q < nq; q++) df->fill(q, 0);
    df->upload();
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    for (Partition partition : devicePartitionMap.at(srcDev)) {
      runTestKernel(df, partition, computeStream);
    }
    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    // Exchange halos
    for (Partition partition : devicePartitionMap.at(srcDev)) {
      std::vector<HaloExchangeData> haloDatas = df->m_haloData[partition];
      for (int i = 0; i < haloDatas.size(); i++) {
        HaloExchangeData haloData = haloDatas.at(i);

        haloData.srcIndexD = new thrust::device_vector<int>();
        haloData.dstIndexD = new thrust::device_vector<int>();

        *haloData.srcIndexD = *haloData.srcIndexH;
        *haloData.dstIndexD = *haloData.dstIndexH;

        const int dstDev = partitionDeviceMap[*haloData.neighbour];
        DistributedDFGroup *dstDf = dfs[dstDev];
        cudaStream_t cpyStream = cpyStreams[dstDev];
        df->pushHalo(srcDev, partition, dstDev, dstDf, haloData, cpyStream);
      }
    }

    for (int i = 0; i < numDevices; i++)
      CUDA_RT_CALL(cudaStreamSynchronize(cpyStreams[i]));

    // for (Partition partition : devicePartitionMap.at(srcDev)) {
    //   // Merge partition array into device 0
    //   if (srcDev != 0) {
    //     DistributedDFGroup *dstDf = dfs[0];
    //     cudaStream_t cpyStream = cpyStreams[0];
    //     df->pushPartition(srcDev, partition, 0, dstDf, cpyStream);
    //   }
    // }
  }

  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    DistributedDFGroup *df = dfs[srcDev];
    df->download();

    // std::cout << "Device " << srcDev << std::endl;
    // std::cout << *df << std::endl;

    // Check after halo exchange
    for (Partition partition : df->getAllocatedPartitions()) {
      std::stringstream ss;
      ss << "Device " << srcDev << " failed" << std::endl;
      // ss << *df << std::endl;
      EXPECT_TRUE(comparePartitions(df, &partition, pAfter)) << ss.str();
    }
    delete df;
    CUDA_RT_CALL(cudaDeviceReset());
  }
  ASSERT_TRUE(success);
}
