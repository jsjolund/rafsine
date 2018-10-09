#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <gtest/gtest.h>

#include "DistributedDFGroup.hpp"
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
  EXPECT_EQ(1 << divisions, topology.getNumPartitionsTotal());
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

  EXPECT_EQ(topology.getPartitionContaining(0, 0, 0), p0);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 128), p0);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 129), p1);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 256), p1);
  EXPECT_THROW(topology.getPartitionContaining(0, 0, 257), std::out_of_range);
}

TEST(BasicTopology, Three) {
  int nx = 64, ny = 64, nz = 2057;
  int divisions = 2;
  Topology topology(nx, ny, nz, divisions);
  Partition *p0 = topology.getPartition(0, 0, 0);
  Partition *p1 = topology.getPartition(0, 0, 1);
  Partition *p2 = topology.getPartition(0, 0, 2);
  Partition *p3 = topology.getPartition(0, 0, 3);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 0), p0);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 514), p0);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 515), p1);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 1028), p1);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 1029), p2);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 1542), p2);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 1543), p3);
  EXPECT_EQ(topology.getPartitionContaining(0, 0, 2056), p3);
  EXPECT_THROW(topology.getPartitionContaining(0, 0, 2057), std::out_of_range);
}

TEST(BasicTopology, Idt) {
  int nx = 64, ny = 64, nz = 2057;
  int divisions = 2;
  Topology topology0(nx, ny, nz, divisions);
  Partition *t0p0 = topology0.getPartition(0, 0, 0);
  Topology topology1(nx, ny, nz, divisions);
  Partition *t1p0 = topology1.getPartition(0, 0, 0);
  EXPECT_EQ(*t0p0, *t1p0);
}

TEST(BasicTopologyKernel, ArrayAccess) {
  int nq = 2, nx = 2, ny = 2, nz = 4, divisions = 1;
  DistributedDFGroup *df = new DistributedDFGroup(nq, nx, ny, nz, divisions);
  df->allocate(*df->getPartition(0, 0, 0));
  df->allocate(*df->getPartition(0, 0, 1));
  df->fill(0, 0);
  df->fill(1, 0);
  int i = 0;
  for (int q = 0; q < nq; ++q)
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          (*df)(q, x, y, z) = ++i;
        }
  std::cout << *df << std::endl;

  std::vector<glm::ivec3> srcPoints;
  std::vector<glm::ivec3> haloPoints;
  df->getPartition(0, 0, 0)->getHalo(glm::ivec3(0, 0, 1), &srcPoints,
                                     &haloPoints);
  for (glm::ivec3 halo : haloPoints) {
    std::cout << halo.x << "," << halo.y << "," << halo.z << std::endl;
  }

  // std::vector<Partition *> partitions = df->getPartitions();
  // for (Partition *partition : partitions) {
  // for (std::pair<glm::ivec3, Partition *> element : partition->m_neighbours)
  // {
  //   glm::ivec3 direction = element.first;
  //   Partition neighbour = *element.second;
  //   std::vector<glm::ivec3> srcPoints;
  //   std::vector<glm::ivec3> haloPoints;
  // }
  // }
}

// TEST(BasicTopologyKernel, One) {
//   int nx = 32, ny = 32, nz = 16;

//   int numDevices = 0;
//   CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));

//   // Create more or equal number of partitions as there are GPUs
//   int divisions = 0;
//   while (1 << divisions < numDevices) divisions++;

//   // Create as many DF groups as there are GPUs
//   DistributedDFGroup *dfs[numDevices - 1];
//   DistributedDFGroup *dfMaster =
//       new DistributedDFGroup(2, nx, ny, nz, divisions);

//   std::vector<Partition *> partitions = dfMaster->getPartitions();
//   int numPartitions = partitions.size();

//   // Create as many threads as there are GPUs
// #pragma omp parallel num_threads(numDevices)
//   {
//     int devId = omp_get_thread_num();
//     CUDA_RT_CALL(cudaSetDevice(devId));
//     CUDA_RT_CALL(cudaFree(0));
// #pragma omp barrier
//     dfs[devId] = (devId == 0)
//                      ? dfMaster
//                      : new DistributedDFGroup(2, nx, ny, nz, divisions);
//     DistributedDFGroup *df = dfs[devId];

//     for (int i = devId; i < numPartitions; i += numDevices) {
//       df->allocate(*partitions.at(i));
//     }
//     df->fill(0, 0);
//     df->fill(1, 1);

//     CUDA_RT_CALL(cudaDeviceReset());
//   }
// }

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
