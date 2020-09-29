#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "Partition.hpp"
#include "test_kernel.hpp"

TEST(PartitionTest, Intersect) {
  Partition lattice(vector3<int>(10, 10, 10), vector3<int>(20, 20, 20),
                    vector3<int>(0, 0, 0));
  vector3<int> min, max, iMin, iMax;
  int volume;

  min = vector3<int>(1, 1, 1);
  max = vector3<int>(9, 9, 9);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;

  min = vector3<int>(11, 11, 11);
  max = vector3<int>(19, 19, 19);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;

  min = vector3<int>(11, 11, 11);
  max = vector3<int>(30, 30, 30);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;

  min = vector3<int>(20, 20, 20);
  max = vector3<int>(30, 30, 30);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;
}

TEST(PartitionTest, GhostLayer) {
  DistributedLattice lattice(10, 10, 10, 2, 1, D3Q4::Y_AXIS);
  std::vector<Partition> partitions = lattice.getPartitions();
  Partition l0 = partitions.at(0);
  Partition l1 = partitions.at(1);
  GhostLayerParameters h0 = l0.getGhostLayer(vector3<int>(0, 1, 0), l1);
  std::cout << "ghostLayer0:\nsrc=" << h0.m_src << " \ndst=" << h0.m_dst
            << " \ndpitch=" << h0.m_dpitch << " \nsrc=" << h0.m_src
            << " \nspitch=" << h0.m_spitch << " \nwidth=" << h0.m_width
            << " \nheight=" << h0.m_height << std::endl;

  GhostLayerParameters h1 = l1.getGhostLayer(vector3<int>(0, -1, 0), l0);
  std::cout << "ghostLayer1:\nsrc=" << h1.m_src << " \ndst=" << h1.m_dst
            << " \ndpitch=" << h1.m_dpitch << " \nsrc=" << h1.m_src
            << " \nspitch=" << h1.m_spitch << " \nwidth=" << h1.m_width
            << " \nheight=" << h1.m_height << std::endl;
}
