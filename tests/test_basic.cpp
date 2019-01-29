
#include <gtest/gtest.h>

#include "Lattice.hpp"

TEST(Lattice, Volume) {
  int nx = 31, ny = 51, nz = 74;
  int divisions = 8;
  Lattice lattice(nx, ny, nz, divisions);
  int totalVol = 0;
  for (int x = 0; x < lattice.getNumSubLattices().x; x++)
    for (int y = 0; y < lattice.getNumSubLattices().y; y++)
      for (int z = 0; z < lattice.getNumSubLattices().z; z++) {
        SubLattice p = lattice.getSubLattice(x, y, z);
        totalVol +=
            p.getLatticeDims().x * p.getLatticeDims().y * p.getLatticeDims().z;
      }
  ASSERT_EQ(totalVol, lattice.getLatticeDims().x * lattice.getLatticeDims().y *
                          lattice.getLatticeDims().z);
  ASSERT_EQ(totalVol, lattice.getLatticeSize());
  ASSERT_EQ(totalVol, nx * ny * nz);
  ASSERT_EQ(divisions, lattice.getNumSubLattices().x *
                           lattice.getNumSubLattices().y *
                           lattice.getNumSubLattices().z);
  ASSERT_EQ(divisions, lattice.getNumSubLatticesTotal());
}

TEST(Lattice, One) {
  int nx = 52, ny = 51, nz = 50;
  int divisions = 0;
  Lattice lattice(nx, ny, nz, divisions);
  SubLattice p0 = lattice.getSubLattice(0, 0, 0);
  ASSERT_EQ(p0.getLatticeDims().x, 52);
  ASSERT_EQ(p0.getLatticeDims().y, 51);
  ASSERT_EQ(p0.getLatticeDims().z, 50);
}

// TEST(Lattice, Two) {
//   int nx = 128, ny = 128, nz = 257;
//   int divisions = 1;
//   Lattice lattice(nx, ny, nz, divisions);
//   SubLattice *p0 = lattice.getSubLattice(0, 0, 0);
//   ASSERT_EQ(p0.getLatticeDims().x, 128);
//   ASSERT_EQ(p0.getLatticeDims().y, 128);
//   ASSERT_EQ(p0.getLatticeDims().z, 129);
//   SubLattice *p1 = lattice.getSubLattice(0, 0, 1);
//   ASSERT_EQ(p1->getLatticeDims().x, 128);
//   ASSERT_EQ(p1->getLatticeDims().y, 128);
//   ASSERT_EQ(p1->getLatticeDims().z, 128);
//   ASSERT_EQ(p0, p0);
//   EXPECT_NE(p0, p1);

//   ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 0), p0);
//   ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 128), p0);
//   ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 129), p1);
//   ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 256), p1);
//   ASSERT_THROW(lattice.getSubLatticeContaining(0, 0, 257),
//   std::out_of_range);
// }

TEST(Lattice, Three) {
  int nx = 64, ny = 64, nz = 2057;
  int divisions = 4;
  Lattice lattice(nx, ny, nz, divisions);
  SubLattice p0 = lattice.getSubLattice(0, 0, 0);
  SubLattice p1 = lattice.getSubLattice(0, 0, 1);
  SubLattice p2 = lattice.getSubLattice(0, 0, 2);
  SubLattice p3 = lattice.getSubLattice(0, 0, 3);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 0), p0);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 514), p0);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 515), p1);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 1028), p1);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 1029), p2);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 1542), p2);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 1543), p3);
  ASSERT_EQ(lattice.getSubLatticeContaining(0, 0, 2056), p3);
  ASSERT_THROW(lattice.getSubLatticeContaining(0, 0, 2057), std::out_of_range);
}

TEST(Lattice, Idt) {
  int nx = 64, ny = 64, nz = 2057;
  int divisions = 2;
  Lattice lattice0(nx, ny, nz, divisions);
  SubLattice t0p0 = lattice0.getSubLattice(0, 0, 0);
  Lattice lattice1(nx, ny, nz, divisions);
  SubLattice t1p0 = lattice1.getSubLattice(0, 0, 0);
  ASSERT_EQ(t0p0, t1p0);
}
