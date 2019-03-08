#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "SubLattice.hpp"
#include "test_kernel.hpp"

TEST(SubLatticeTest, Intersect) {
  SubLattice lattice(glm::ivec3(10, 10, 10), glm::ivec3(20, 20, 20),
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
