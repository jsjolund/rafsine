#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "Partition.hpp"
#include "test_kernel.hpp"

TEST(PartitionTest, Intersect) {
  Partition lattice(glm::ivec3(10, 10, 10), glm::ivec3(20, 20, 20),
                    glm::ivec3(0, 0, 0));
  glm::ivec3 min, max, iMin, iMax;
  int volume;

  min = glm::ivec3(1, 1, 1);
  max = glm::ivec3(9, 9, 9);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;

  min = glm::ivec3(11, 11, 11);
  max = glm::ivec3(19, 19, 19);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;

  min = glm::ivec3(11, 11, 11);
  max = glm::ivec3(30, 30, 30);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;

  min = glm::ivec3(20, 20, 20);
  max = glm::ivec3(30, 30, 30);
  volume = lattice.intersect(min, max, &iMin, &iMax);
  std::cout << "vol=" << volume << " min=" << iMin << " max=" << iMax
            << std::endl;
}

TEST(PartitionTest, GhostLayer) {
  DistributedLattice lattice(10, 10, 10, 2, 1);
  std::vector<Partition> partitions = lattice.getPartitions();
  Partition l0 = partitions.at(0);
  Partition l1 = partitions.at(1);
  GhostLayerParameters h0 = l0.getGhostLayer(glm::ivec3(0, 1, 0), l1);
  std::cout << "ghostLayer0:\nsrc=" << h0.m_src << " \ndst=" << h0.m_dst
            << " \ndpitch=" << h0.m_dpitch << " \nsrc=" << h0.m_src
            << " \nspitch=" << h0.m_spitch << " \nwidth=" << h0.m_width
            << " \nheight=" << h0.m_height << std::endl;

  GhostLayerParameters h1 = l1.getGhostLayer(glm::ivec3(0, -1, 0), l0);
  std::cout << "ghostLayer1:\nsrc=" << h1.m_src << " \ndst=" << h1.m_dst
            << " \ndpitch=" << h1.m_dpitch << " \nsrc=" << h1.m_src
            << " \nspitch=" << h1.m_spitch << " \nwidth=" << h1.m_width
            << " \nheight=" << h1.m_height << std::endl;
}