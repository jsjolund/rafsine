#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <gtest/gtest.h>

#include "DistributionFunctionsGroup.hpp"
#include "LuaContext.hpp"
#include "PartitionTopology.hpp"

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

TEST(BasicTopology, Volume) {
  int nx = 371, ny = 531, nz = 764;
  int divisions = 8;
  Topology topology(nx, ny, nz, divisions);
  int totalVol = 0;
  for (int x = 0; x < topology.getNumPartitions().x; x++)
    for (int y = 0; y < topology.getNumPartitions().y; y++)
      for (int z = 0; z < topology.getNumPartitions().z; z++) {
        Partition *p = topology.getPartition(x, y, z);
        totalVol += p->getVolume();
      }
  EXPECT_EQ(totalVol, topology.getLatticeSize().x *
                          topology.getLatticeSize().y *
                          topology.getLatticeSize().z);
  EXPECT_EQ(totalVol, nx * ny * nz);
  EXPECT_EQ(1 << divisions, topology.getNumPartitions().x *
                                topology.getNumPartitions().y *
                                topology.getNumPartitions().z);
}

TEST(BasicTopology, One) {
  int nx = 512, ny = 511, nz = 510;
  int divisions = 0;
  Topology topology(nx, ny, nz, divisions);
  Partition *p0 = topology.getPartition(0, 0, 0);
  EXPECT_EQ(p0->getLatticeSize().x, 512);
  EXPECT_EQ(p0->getLatticeSize().y, 511);
  EXPECT_EQ(p0->getLatticeSize().z, 510);
}

TEST(BasicTopology, Two) {
  int nx = 128, ny = 128, nz = 257;
  int divisions = 1;
  Topology topology(nx, ny, nz, divisions);
  Partition *p0 = topology.getPartition(0, 0, 0);
  EXPECT_EQ(p0->getLatticeSize().x, 128);
  EXPECT_EQ(p0->getLatticeSize().y, 128);
  EXPECT_EQ(p0->getLatticeSize().z, 129);
  Partition *p1 = topology.getPartition(0, 0, 1);
  EXPECT_EQ(p1->getLatticeSize().x, 128);
  EXPECT_EQ(p1->getLatticeSize().y, 128);
  EXPECT_EQ(p1->getLatticeSize().z, 128);
  EXPECT_EQ(p0, p0);
  EXPECT_NE(p0, p1);
}

TEST(BasicTopologyKernel, One) {
  //     int nx = 32, ny = 32, nz = 16;

  //     int numDevices = 0;
  //     CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));

  //     // Create more or equal number of partitions as there are GPUs, closest
  //     power of two int divisions = 0; while (1 << divisions < numDevices)
  //     {
  //         divisions++;
  //     }

  //     // Create as many DF groups as there are GPUs
  //     DistributionFunctionsGroup *dfs[numDevices];
  //     for (int i = 0; i < numDevices; i++)
  //     {
  //         dfs[i] = new DistributionFunctionsGroup(4, nx, ny, nz, divisions);
  //     }

  //     // Create as many threads as there are GPUs
  // #pragma omp parallel num_threads(numDevices) shared(dfs)
  //     {
  //         int devId = omp_get_thread_num();
  //         std::cout << devId << std::endl;
  //         CUDA_RT_CALL(cudaSetDevice(devId));
  //         CUDA_RT_CALL(cudaFree(0));
  // #pragma omp barrier

  //         DistributionFunctionsGroup *df = dfs[devId];

  //         CUDA_RT_CALL(cudaDeviceReset());
  //     }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
