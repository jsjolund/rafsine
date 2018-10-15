
#include <gtest/gtest.h>

#include "PartitionTopology.hpp"

TEST(Topology, Volume) {
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

TEST(Topology, One) {
  int nx = 512, ny = 511, nz = 510;
  int divisions = 0;
  Topology topology(nx, ny, nz, divisions);
  Partition *p0 = topology.getPartition(0, 0, 0);
  EXPECT_EQ(p0->getLatticeSize().x, 512);
  EXPECT_EQ(p0->getLatticeSize().y, 511);
  EXPECT_EQ(p0->getLatticeSize().z, 510);
}

TEST(Topology, Two) {
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

TEST(Topology, Three) {
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

TEST(Topology, Idt) {
  int nx = 64, ny = 64, nz = 2057;
  int divisions = 2;
  Topology topology0(nx, ny, nz, divisions);
  Partition *t0p0 = topology0.getPartition(0, 0, 0);
  Topology topology1(nx, ny, nz, divisions);
  Partition *t1p0 = topology1.getPartition(0, 0, 0);
  EXPECT_EQ(*t0p0, *t1p0);
}
