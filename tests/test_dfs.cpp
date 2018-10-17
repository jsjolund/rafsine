#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <sys/time.h>
#include <time.h>

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

// Reference for initial values in lattice, before halo exchange
real p0a[4][4][4] = {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
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
          (*df)(q, x, y, z) = ++i;
        }

  assertPartitionEq(df, partitions.at(0), p0a);
  assertPartitionEq(df, partitions.at(1), p1a);

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
  assertPartitionEq(df, partitions.at(0), p0b);
  assertPartitionEq(df, partitions.at(1), p1b);
}

__global__ void TestKernel1(real *__restrict__ df, glm::ivec3 pMin,
                            glm::ivec3 pMax, int i) {
  int x = threadIdx.x;
  int y = blockIdx.x;
  int z = blockIdx.y;
  glm::ivec3 p(x, y, z);
  glm::ivec3 dfSize = pMax - pMin;
  glm::ivec3 arrSize = dfSize + glm::ivec3(2, 2, 2);
  if ((p.x >= dfSize.x) || (p.y >= dfSize.y) || (p.z >= dfSize.z)) return;
  glm::ivec3 q = p + glm::ivec3(1, 1, 1);
  real value = i + I3D(x, y, z, dfSize.x, dfSize.y, dfSize.z);
  df[I4D(0, q.x, q.y, q.z, arrSize.x, arrSize.y, arrSize.z)] = value;
}

void runTestKernel(DistributedDFGroup *df, Partition partition, int i) {
  glm::ivec3 n = partition.getLatticeSize();
  dim3 grid_size(n.y + 2, n.z + 2, 1);
  dim3 block_size(n.x + 2, 1, 1);
  TestKernel1<<<grid_size, block_size>>>(df->gpu_ptr(partition),
                                         partition.getLatticeMin(),
                                         partition.getLatticeMax(), i);
}

TEST(DistributedDF, SingleGPU) {
  int nq = 1, nx = 2, ny = 2, nz = 4, divisions = 1;
  DistributedDFGroup *df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
  for (Partition *partition : df->getPartitions()) df->allocate(*partition);
  for (int q = 0; q < nq; q++) df->fill(q, 0);
  df->upload();
  int i = 1;
  std::vector<Partition *> partitions = df->getPartitions();
  for (Partition *partition : partitions) {
    runTestKernel(df, *partition, i);
    i += partition->getVolume();
  }
  df->download();
  assertPartitionEq(df, partitions.at(0), p0a);
  assertPartitionEq(df, partitions.at(1), p1a);
}

TEST(DistributedDF, ExplicitP2PCopy) {
  int numDevices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  ASSERT_GE(numDevices, 2);

  int devIdSrc = 0, devIdDst = 1;
  const size_t SIZE = 1000;
  char *vec0, *vec1, *vecH;

  // Allocate memory on gpu0 and set it to some value
  CUDA_RT_CALL(cudaSetDevice(devIdSrc));
  CUDA_RT_CALL(cudaMalloc(&vec0, SIZE * sizeof(char)));
  CUDA_RT_CALL(cudaMemset(vec0, 'y', SIZE * sizeof(char)));
  // Allocate memory on gpu1 and set it to some other value
  CUDA_RT_CALL(cudaSetDevice(devIdDst));
  CUDA_RT_CALL(cudaMalloc(&vec1, SIZE * sizeof(char)));
  CUDA_RT_CALL(cudaMemset(vec1, 'n', SIZE * sizeof(char)));
  // Copy P2P
  CUDA_RT_CALL(
      cudaMemcpyPeer(vec1, devIdDst, vec0, devIdSrc, SIZE * sizeof(char)));
  // Allocate memory on host, copy from gpu1 to host, and verify P2P copy worked
  CUDA_RT_CALL(cudaMallocHost(&vecH, SIZE * sizeof(char)));
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(
      cudaMemcpy(vecH, vec1, SIZE * sizeof(char), cudaMemcpyDeviceToHost));
  for (int i = 0; i < SIZE; i++) ASSERT_EQ(vecH[i], 'y');
}

// TEST(DistributedDF, MultiGPUCopy) {
//   int numDevices = 0;
//   CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
//   ASSERT_GE(numDevices, 2);

//   const size_t SIZE = 100;

//   CUDA_RT_CALL(cudaSetDevice(0));
//   CUDA_RT_CALL(cudaDeviceEnablePeerAccess(1, 0));

//   thrust::device_vector<int> vec0(SIZE);
//   int *vec0 = thrust::raw_pointer_cast(&vec0[0]);
//   thrust::sequence(vec0.begin(), vec0.end(), 23);
//   std::cout << "Device 0: ";
//   for (int i = 0; i < SIZE; i++) std::cout << vec0[i] << " ";
//   std::cout << std::endl;

//   CUDA_RT_CALL(cudaSetDevice(1));
//   CUDA_RT_CALL(cudaDeviceEnablePeerAccess(0, 0));

//   thrust::device_vector<int> vec1(SIZE);
//   int *vec1 = thrust::raw_pointer_cast(&vec1[0]);

//   // CUDA_RT_CALL(cudaMemcpyPeer(vec1, 1, vec0, 0, sizeof(int) *
//   SIZE));
//   // CUDA_RT_CALL(cudaMemcpy(vec1, vec0, sizeof(int) * SIZE,
//   cudaMemcpyDefault)); CUDA_RT_CALL(cudaMemcpyAsync(vec1, vec0,
//   sizeof(int) * SIZE, cudaMemcpyDefault));

//   // thrust::transform(vec1.begin(), vec1.end(), vec1.begin(),
//   // thrust::negate<int>());

//   CUDA_RT_CALL(cudaDeviceSynchronize());

//   std::cout << "Device 1: ";
//   for (int i = 0; i < SIZE; i++) std::cout << vec1[i] << " ";
//   std::cout << std::endl;
//   CUDA_RT_CALL(cudaSetDevice(0));
// }

// TEST(DistributedDF, MultiGPU) {
//   int numDevices = 0;
//   CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
//   numDevices = min(numDevices, 8);  // TODO

//   // Create more or equal number of partitions as there are GPUs
//   int nq = 1, nx = 32, ny = 32, nz = 16, divisions = 0;
//   while (1 << divisions < numDevices) divisions++;

//   // Create as many DF groups as there are GPUs
//   DistributedDFGroup *dfMaster =
//       new DistributedDFGroup(nq, nx, ny, nz, divisions);
//   DistributedDFGroup *dfs[numDevices];
//   dfs[0] = dfMaster;

//   // Distribute the workload
//   std::vector<Partition *> partitions = dfMaster->getPartitions();
//   int numPartitions = partitions.size();
//   std::unordered_map<Partition, int> partitionDeviceMap;
//   for (int i = 0; i < numPartitions; i++) {
//     Partition p = *partitions.at(i);
//     int devIndex = i % numDevices;
//     partitionDeviceMap[p] = devIndex;
//   }

//   bool p2pWorks = true;

//   // Create one CPU thread per GPU
// #pragma omp parallel num_threads(numDevices)
//   {
//     const int devId = omp_get_thread_num();
//     CUDA_RT_CALL(cudaSetDevice(devId));
//     CUDA_RT_CALL(cudaFree(0));
// #pragma omp barrier
//     DistributedDFGroup *df =
//         (devId == 0) ? dfMaster
//                      : new DistributedDFGroup(nq, nx, ny, nz, divisions);
//     dfs[devId] = df;

//     std::vector<bool> hasPeerAccess(numDevices);
//     hasPeerAccess.at(devId) = true;

//     // Enable P2P access between GPUs
//     for (std::pair<Partition, std::vector<Partition *>> element :
//          df->m_neighbours) {
//       std::vector<Partition *> neighbours = element.second;
//       for (int i = 0; i < neighbours.size(); i++) {
//         Partition *neighbour = neighbours.at(i);
//         int nDevId = partitionDeviceMap[*neighbour];

//         if (!hasPeerAccess.at(nDevId)) {
//           int cudaCanAccessPeer = 0;
//           cudaError_t cudaPeerAccessStatus;
//           CUDA_RT_CALL(
//               cudaDeviceCanAccessPeer(&cudaCanAccessPeer, devId, nDevId));
//           if (cudaCanAccessPeer) {
//             cudaPeerAccessStatus = cudaDeviceEnablePeerAccess(nDevId, 0);
//             hasPeerAccess.at(nDevId) = true;
//           }
//           if (!cudaDeviceCanAccessPeer || cudaPeerAccessStatus !=
//           cudaSuccess) {
// #pragma omp critical
//             {
//               if (p2pWorks) p2pWorks = false;
//             }
//           }
//         }
//       }
//     }
//     for (std::pair<Partition, int> element : partitionDeviceMap)
//       if (element.second == devId) df->allocate(element.first);
// #pragma omp barrier

//     for (int q = 0; q < nq; q++) df->fill(q, 0);
//     df->upload();

//     for (Partition partition : df->getAllocatedPartitions()) {
//       runTestKernel(df, partition, 0);
//     }
//     CUDA_RT_CALL(cudaDeviceSynchronize());
// #pragma omp barrier
//     for (Partition partition : df->getAllocatedPartitions()) {
//       std::vector<Partition *> neighbours = df->m_neighbours[partition];

//       for (int i = 0; i < neighbours.size(); i++) {
//         Partition neighbour = *neighbours.at(i);
//         int nDevId = partitionDeviceMap[neighbour];
//         DistributedDFGroup *nDf = dfs[nDevId];
//         glm::ivec3 direction = D3Q19directionVectors[i];

//         std::vector<glm::ivec3> pSrc, nSrc, pDst, nDst;

//         partition.getHalo(direction, &pSrc, &nDst);
//         neighbour.getHalo(-direction, &nSrc, &pDst);

//         for (int j = 0; j < pSrc.size(); j++) {
//           glm::ivec3 src = pSrc.at(j);
//           glm::ivec3 dst = pDst.at(j);
//           for (int q = 0; q < nq; ++q) {
//             real * srcPtr = df->gpu_ptr(partition, q, src.x, src.y, src.z);
//             real * dstPtr = nDf->gpu_ptr(neighbour, q, dst.x, dst.y, dst.z);
//             CUDA_RT_CALL(cudaMemcpy(dstPtr, srcPtr, sizeof(real),
//             cudaMemcpyDefault)); std::cout << std::endl;
//           }
//         }
//       }
//     }

//     CUDA_RT_CALL(cudaDeviceSynchronize());

// #pragma omp barrier
//     CUDA_RT_CALL(cudaDeviceReset());
//   }

// ASSERT_EQ(p2pWorks, true);
// }
