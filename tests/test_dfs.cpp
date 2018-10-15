#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "DistributedDFGroup.hpp"

#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
  }

template <size_t nx, size_t ny, size_t nz>
void assertPartitionEq(DistributedDFGroup *df, Partition *p0,
                       real (&ref)[nx][ny][nz]) {
  glm::ivec3 min = p0->getLatticeMin() - glm::ivec3(1, 1, 1);
  glm::ivec3 max = p0->getLatticeMax() + glm::ivec3(1, 1, 1);
  for (int hz = min.z, rz = 0; hz < max.z; hz++, rz++)
    for (int hy = min.y, ry = 0; hy < max.y; hy++, ry++)
      for (int hx = min.x, rx = 0; hx < max.x; hx++, rx++) {
        real a = ref[rz][ry][rx];
        real b = (*df)(*p0, 0, hx, hy, hz);
        ASSERT_EQ(a, b);
      }
}

TEST(DistributedDF, HaloExchangeCPU) {
  // Reference for initial values in lattice, before halo exchange
  real p0a[4][4][4] = {
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 1, 2, 0}, {0, 3, 4, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 5, 6, 0}, {0, 7, 8, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
  real p1a[4][4][4] = {
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 9, 10, 0}, {0, 11, 12, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 13, 14, 0}, {0, 15, 16, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
  // Reference for after halo exchange
  real p0b[4][4][4] = {
      {{16, 15, 16, 15}, {14, 13, 14, 13}, {16, 15, 16, 15}, {14, 13, 14, 13}},
      {{4, 3, 4, 3}, {2, 1, 2, 1}, {4, 3, 4, 3}, {2, 1, 2, 1}},
      {{8, 7, 8, 7}, {6, 5, 6, 5}, {8, 7, 8, 7}, {6, 5, 6, 5}},
      {{12, 11, 12, 11}, {10, 9, 10, 9}, {12, 11, 12, 11}, {10, 9, 10, 9}}};
  real p1b[4][4][4] = {
      {{8, 7, 8, 7}, {6, 5, 6, 5}, {8, 7, 8, 7}, {6, 5, 6, 5}},
      {{12, 11, 12, 11}, {10, 9, 10, 9}, {12, 11, 12, 11}, {10, 9, 10, 9}},
      {{16, 15, 16, 15}, {14, 13, 14, 13}, {16, 15, 16, 15}, {14, 13, 14, 13}},
      {{4, 3, 4, 3}, {2, 1, 2, 1}, {4, 3, 4, 3}, {2, 1, 2, 1}}};

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
          (*df)(q, x, y, z) = ++i;
        }

  assertPartitionEq(df, partitions.at(0), p0a);
  assertPartitionEq(df, partitions.at(1), p1a);

  for (std::pair<Partition *, std::vector<Partition *>> element :
       df->m_neighbours) {
    Partition *partition = element.first;
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

      for (int i = 0; i < pSrc.size(); i++) {
        glm::ivec3 src = pSrc.at(i);
        glm::ivec3 dst = pDst.at(i);
        for (int q = 0; q < nq; ++q) {
          (*df)(*neighbour, q, dst.x, dst.y, dst.z) =
              (*df)(*partition, q, src.x, src.y, src.z);
        }
      }
    }
  }
  assertPartitionEq(df, partitions.at(0), p0b);
  assertPartitionEq(df, partitions.at(1), p1b);
}

TEST(DistributedDF, One) {
  int numDevices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, 8);  // TODO

  // Create more or equal number of partitions as there are GPUs
  int nq = 19, nx = 32, ny = 32, nz = 16, divisions = 0;
  while (1 << divisions < numDevices) divisions++;

  // Create as many DF groups as there are GPUs
  DistributedDFGroup *dfs[numDevices - 1];
  DistributedDFGroup *dfMaster =
      new DistributedDFGroup(nq, nx, ny, nz, divisions);

  std::vector<Partition *> partitions = dfMaster->getPartitions();
  int numPartitions = partitions.size();
  std::unordered_map<Partition, int> m_partitionDeviceMap;

  bool p2pWorks = true;

  // Create as many threads as there are GPUs
#pragma omp parallel num_threads(numDevices)
  {
    const int devId = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(devId));
    CUDA_RT_CALL(cudaFree(0));
#pragma omp barrier
    DistributedDFGroup *df =
        (devId == 0) ? dfMaster
                     : new DistributedDFGroup(nq, nx, ny, nz, divisions);
    dfs[devId] = df;

    for (int i = devId; i < numPartitions; i += numDevices) {
      Partition p = *partitions.at(i);
      df->allocate(p);
      m_partitionDeviceMap[p] = devId;
    }
    for (int q = 0; q < nq; q++) df->fill(q, 0);
    df->upload();

    std::vector<bool> hasPeerAccess(numDevices);
    hasPeerAccess.at(devId) = true;

    for (std::pair<Partition *, std::vector<Partition *>> element :
         df->m_neighbours) {
      // Partition *partition = element.first;
      std::vector<Partition *> neighbours = element.second;

      for (int i = 0; i < neighbours.size(); i++) {
        Partition *neighbour = neighbours.at(i);
        int nDevId = m_partitionDeviceMap[*neighbour];

        if (!hasPeerAccess.at(nDevId)) {
          int cudaCanAccessPeer = 0;
          cudaError_t cudaPeerAccessStatus;
          CUDA_RT_CALL(
              cudaDeviceCanAccessPeer(&cudaCanAccessPeer, devId, nDevId));
          if (cudaCanAccessPeer) {
            cudaPeerAccessStatus = cudaDeviceEnablePeerAccess(nDevId, 0);
            hasPeerAccess.at(nDevId) = true;
          }
          if (!cudaDeviceCanAccessPeer || cudaPeerAccessStatus != cudaSuccess) {
#pragma omp critical
            {
              if (p2pWorks) p2pWorks = false;
            }
          }
        }
      }
    }

    CUDA_RT_CALL(cudaDeviceReset());
  }
  ASSERT_EQ(p2pWorks, true);
}
