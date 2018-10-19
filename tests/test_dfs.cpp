#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "DistributedDFGroup.hpp"

template <size_t nx, size_t ny, size_t nz>
static bool comparePartitions(DistributedDFGroup *df, Partition *p0,
                              real (&ref)[nx][ny][nz]) {
  glm::ivec3 min = p0->getLatticeMin() - glm::ivec3(1, 1, 1);
  glm::ivec3 max = p0->getLatticeMax() + glm::ivec3(1, 1, 1);
  for (int hz = min.z, rz = 0; hz < max.z; hz++, rz++)
    for (int hy = min.y, ry = 0; hy < max.y; hy++, ry++)
      for (int hx = min.x, rx = 0; hx < max.x; hx++, rx++) {
        real a = ref[rz][ry][rx];
        real b = (*df)(*p0, 0, hx, hy, hz);
        if (a != b) return false;
      }
  return true;
}

// Reference for initial values in lattice, before halo exchange
static real pBefore[4][4][4] = {
    {
    {0, 0, 0, 0}, 
    {0, 0, 0, 0}, 
    {0, 0, 0, 0}, 
    {0, 0, 0, 0}},
    {
    {0, 0, 0, 0}, 
    {0, 1, 2, 0}, 
    {0, 3, 4, 0}, 
    {0, 0, 0, 0}},
    {
    {0, 0, 0, 0}, 
    {0, 5, 6, 0}, 
    {0, 7, 8, 0}, 
    {0, 0, 0, 0}},
    {
    {0, 0, 0, 0}, 
    {0, 0, 0, 0}, 
    {0, 0, 0, 0}, 
    {0, 0, 0, 0}}};

// Reference for after halo exchange
static real pAfter[4][4][4] = {
    {
    {8, 7, 8, 7}, 
    {6, 5, 6, 5}, 
    {8, 7, 8, 7}, 
    {6, 5, 6, 5}},
    {
    {4, 3, 4, 3}, 
    {2, 1, 2, 1}, 
    {4, 3, 4, 3}, 
    {2, 1, 2, 1}},
    {
    {8, 7, 8, 7}, 
    {6, 5, 6, 5}, 
    {8, 7, 8, 7},
    {6, 5, 6, 5}},
    {
    {4, 3, 4, 3}, 
    {2, 1, 2, 1}, 
    {4, 3, 4, 3}, 
    {2, 1, 2, 1}}};

TEST(DistributedDF, HaloExchangeCPU) {
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
  std::cout << *df << std::endl;
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pBefore));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pBefore));

  for (std::pair<Partition, std::vector<Partition *>> element :
       df->m_neighbours) {
    Partition *partition = &element.first;
    std::vector<Partition *> neighbours = element.second;

    for (int i = 0; i < neighbours.size(); i++) {
      Partition *neighbour = neighbours.at(i);

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
  std::cout << *df << std::endl;
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pAfter));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pAfter));
}

__global__ void TestKernel(real *__restrict__ df, glm::ivec3 pMin,
                           glm::ivec3 pMax) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y;
  glm::ivec3 p(x, y, z);
  glm::ivec3 dfSize = pMax - pMin;
  glm::ivec3 arrSize = dfSize + glm::ivec3(2, 2, 2);
  if ((p.x >= dfSize.x) || (p.y >= dfSize.y) || (p.z >= dfSize.z)) return;
  glm::ivec3 q = p + glm::ivec3(1, 1, 1);
  real value = 1 + I3D(x, y, z, dfSize.x, dfSize.y, dfSize.z);
  df[I4D(0, q.x, q.y, q.z, arrSize.x, arrSize.y, arrSize.z)] = value;
}

void runTestKernel(DistributedDFGroup *df, Partition partition) {
  glm::ivec3 n = partition.getLatticeSize();
  dim3 grid_size(n.y + 2, n.z + 2, 1);
  dim3 block_size(n.x + 2, 1, 1);
  glm::ivec3 p = partition.getLatticeMin() - glm::ivec3(1, 1, 1);
  TestKernel<<<grid_size, block_size>>>(
      df->gpu_ptr(partition, 0, p.x, p.y, p.z), partition.getLatticeMin(),
      partition.getLatticeMax());
}

TEST(DistributedDF, SingleGPUKernelPartition) {
  const int nq = 1, nx = 2, ny = 2, nz = 4, divisions = 1;
  DistributedDFGroup *df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
  for (Partition *partition : df->getPartitions()) df->allocate(*partition);
  for (int q = 0; q < nq; q++) df->fill(q, 0);
  df->upload();
  std::vector<Partition *> partitions = df->getPartitions();
  for (Partition *partition : partitions) {
    runTestKernel(df, *partition);
  }
  df->download();
  ASSERT_TRUE(comparePartitions(df, partitions.at(0), pBefore));
  ASSERT_TRUE(comparePartitions(df, partitions.at(1), pBefore));
}

TEST(DistributedDF, MultiGPU) {
  int numDevices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, 2);  // Limit to 2 for this test
  ASSERT_EQ(numDevices, 2);

  // Create more or equal number of partitions as there are GPUs
  int nq = 1, nx = 2, ny = 2, nz = 4, divisions = 0;
  while (1 << divisions < numDevices) divisions++;

  // Create as many DF groups as there are GPUs
  DistributedDFGroup *dfs[numDevices];

  // Distribute the workload
  std::unordered_map<Partition, int> partitionDeviceMap;
  std::vector<std::vector<Partition>> devicePartitionMap(numDevices);
  // Calculate partitions and assign them to GPUs
  {
    DistributedDFGroup *df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
    std::vector<Partition *> partitions = df->getPartitions();
    for (int i = 0; i < partitions.size(); i++) {
      Partition partition = *partitions.at(i);
      int devIndex = i % numDevices;
      partitionDeviceMap[partition] = devIndex;
      devicePartitionMap.at(devIndex).push_back(partition);
    }
    delete df;
  }

  bool partitionsOk = true;

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    const int devId = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(devId));
    CUDA_RT_CALL(cudaFree(0));
#pragma omp barrier
    DistributedDFGroup *df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
    dfs[devId] = df;
    for (Partition partition : devicePartitionMap.at(devId))
      df->allocate(partition);
#pragma omp barrier

    for (int q = 0; q < nq; q++) df->fill(q, 0);
    df->upload();

    for (Partition partition : df->getAllocatedPartitions()) {
      runTestKernel(df, partition);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
    df->download();
    for (Partition partition : df->getAllocatedPartitions()) {
#pragma omp critical
      {
        if (!comparePartitions(df, &partition, pBefore)) partitionsOk = false;
      }
    }
    for (Partition partition : df->getAllocatedPartitions()) {
      std::vector<Partition *> neighbours = df->m_neighbours[partition];

      for (int i = 0; i < neighbours.size(); i++) {
        Partition neighbour = *neighbours.at(i);
        const int nDevId = partitionDeviceMap[neighbour];

        DistributedDFGroup *nDf = dfs[nDevId];

        std::vector<glm::ivec3> pSrc, nSrc, pDst, nDst;
        glm::ivec3 direction = D3Q19directionVectors[i];
        partition.getHalo(direction, &pSrc, &nDst);
        neighbour.getHalo(-direction, &nSrc, &pDst);

        for (int j = 0; j < pSrc.size(); j++) {
          glm::ivec3 src = pSrc.at(j);
          glm::ivec3 dst = pDst.at(j);
          // TODO(Only take the relevant direction vector)
          for (int q = 0; q < nq; ++q) {
            int srcDev = devId;
            int dstDev = nDevId;
            real *srcPtr = df->gpu_ptr(partition, q, src.x, src.y, src.z);
            real *dstPtr = nDf->gpu_ptr(neighbour, q, dst.x, dst.y, dst.z);
            size_t size = sizeof(real);

            if (nDevId == devId) {
              CUDA_RT_CALL(
                  cudaMemcpy(dstPtr, srcPtr, size, cudaMemcpyDeviceToDevice));
            } else {
              CUDA_RT_CALL(
                  cudaMemcpyPeer(dstPtr, dstDev, srcPtr, srcDev, size));
            }
            CUDA_RT_CALL(cudaDeviceSynchronize());
          }
        }
      }
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    df->download();
#pragma omp barrier
    CUDA_RT_CALL(cudaDeviceReset());
  }
  ASSERT_TRUE(partitionsOk);
  for (int i = 0; i < numDevices; i++) {
    DistributedDFGroup *df = dfs[i];
    std::cout << *df << std::endl;
    for (Partition partition : df->getAllocatedPartitions()) {
      ASSERT_TRUE(comparePartitions(df, &partition, pAfter));
    }
  }
}
