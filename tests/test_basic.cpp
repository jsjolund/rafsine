
#include <gtest/gtest.h>

#include "PartitionTopology.hpp"

TEST(Topology, Volume) {
  int nx = 31, ny = 51, nz = 74;
  int divisions = 8;
  Topology topology(nx, ny, nz, divisions);
  int totalVol = 0;
  for (int x = 0; x < topology.getNumPartitions().x; x++)
    for (int y = 0; y < topology.getNumPartitions().y; y++)
      for (int z = 0; z < topology.getNumPartitions().z; z++) {
        Partition *p = topology.getPartition(x, y, z);
        totalVol += p->getLatticeDims().x * p->getLatticeDims().y *
                    p->getLatticeDims().z;
      }
  ASSERT_EQ(totalVol, topology.getLatticeDims().x *
                          topology.getLatticeDims().y *
                          topology.getLatticeDims().z);
  ASSERT_EQ(totalVol, topology.getLatticeSize());
  ASSERT_EQ(totalVol, nx * ny * nz);
  ASSERT_EQ(divisions, topology.getNumPartitions().x *
                                topology.getNumPartitions().y *
                                topology.getNumPartitions().z);
  ASSERT_EQ(divisions, topology.getNumPartitionsTotal());
}

TEST(Topology, One) {
  int nx = 52, ny = 51, nz = 50;
  int divisions = 0;
  Topology topology(nx, ny, nz, divisions);
  Partition *p0 = topology.getPartition(0, 0, 0);
  ASSERT_EQ(p0->getLatticeDims().x, 52);
  ASSERT_EQ(p0->getLatticeDims().y, 51);
  ASSERT_EQ(p0->getLatticeDims().z, 50);
}

// TEST(Topology, Two) {
//   int nx = 128, ny = 128, nz = 257;
//   int divisions = 1;
//   Topology topology(nx, ny, nz, divisions);
//   Partition *p0 = topology.getPartition(0, 0, 0);
//   ASSERT_EQ(p0->getLatticeDims().x, 128);
//   ASSERT_EQ(p0->getLatticeDims().y, 128);
//   ASSERT_EQ(p0->getLatticeDims().z, 129);
//   Partition *p1 = topology.getPartition(0, 0, 1);
//   ASSERT_EQ(p1->getLatticeDims().x, 128);
//   ASSERT_EQ(p1->getLatticeDims().y, 128);
//   ASSERT_EQ(p1->getLatticeDims().z, 128);
//   ASSERT_EQ(p0, p0);
//   EXPECT_NE(p0, p1);

//   ASSERT_EQ(topology.getPartitionContaining(0, 0, 0), p0);
//   ASSERT_EQ(topology.getPartitionContaining(0, 0, 128), p0);
//   ASSERT_EQ(topology.getPartitionContaining(0, 0, 129), p1);
//   ASSERT_EQ(topology.getPartitionContaining(0, 0, 256), p1);
//   ASSERT_THROW(topology.getPartitionContaining(0, 0, 257), std::out_of_range);
// }

TEST(Topology, Three) {
  int nx = 64, ny = 64, nz = 2057;
  int divisions = 2;
  Topology topology(nx, ny, nz, divisions);
  Partition *p0 = topology.getPartition(0, 0, 0);
  Partition *p1 = topology.getPartition(0, 0, 1);
  Partition *p2 = topology.getPartition(0, 0, 2);
  Partition *p3 = topology.getPartition(0, 0, 3);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 0), p0);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 514), p0);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 515), p1);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 1028), p1);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 1029), p2);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 1542), p2);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 1543), p3);
  ASSERT_EQ(topology.getPartitionContaining(0, 0, 2056), p3);
  ASSERT_THROW(topology.getPartitionContaining(0, 0, 2057), std::out_of_range);
}

TEST(Topology, Idt) {
  int nx = 64, ny = 64, nz = 2057;
  int divisions = 2;
  Topology topology0(nx, ny, nz, divisions);
  Partition *t0p0 = topology0.getPartition(0, 0, 0);
  Topology topology1(nx, ny, nz, divisions);
  Partition *t1p0 = topology1.getPartition(0, 0, 0);
  ASSERT_EQ(*t0p0, *t1p0);
}
