
#include <gtest/gtest.h>

#include "Lattice.hpp"
#include "rapidcsv.h"

TEST(Lattice, Volume) {
  int nx = 31, ny = 51, nz = 74;
  int divisions = 8;
  Lattice lattice(nx, ny, nz, divisions);
  int totalVol = 0;
  for (int x = 0; x < lattice.getNumSubLattices().x; x++)
    for (int y = 0; y < lattice.getNumSubLattices().y; y++)
      for (int z = 0; z < lattice.getNumSubLattices().z; z++) {
        SubLattice p = lattice.getSubLattice(x, y, z);
        totalVol += p.getDims().x * p.getDims().y * p.getDims().z;
      }
  ASSERT_EQ(totalVol,
            lattice.getDims().x * lattice.getDims().y * lattice.getDims().z);
  ASSERT_EQ(totalVol, lattice.getSize());
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
  ASSERT_EQ(p0.getDims().x, 52);
  ASSERT_EQ(p0.getDims().y, 51);
  ASSERT_EQ(p0.getDims().z, 50);
}

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

TEST(Csv, Read) {
  rapidcsv::Document doc("problems/data_center/input.csv",
                         rapidcsv::LabelParams(0, -1));
  std::vector<std::string> columnNames = doc.GetColumnNames();
  std::cout << "Found " << columnNames.size() << " columns" << std::endl;
  for (unsigned int i = 0; i < columnNames.size(); i++) {
    std::cout << columnNames.at(i) << " | ";
  }
  std::cout << std::endl;
  std::vector<int> close = doc.GetColumn<int>("time0");
  std::cout << "Read time0 " << close.size() << " values." << std::endl;
  for (unsigned int i = 0; i < close.size(); i++) {
    std::cout << close.at(i) << std::endl;
  }
}
