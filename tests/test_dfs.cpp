#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "DistributionFunction.hpp"
#include "KernelInterface.hpp"
#include "test_kernel.hpp"

/**
 * @brief Compare a subLattice with a reference array
 */
template <size_t nx, size_t ny, size_t nz>
static int compareSubLattices(DistributionFunction *df, SubLattice p0,
                              real (&ref)[nx][ny][nz]) {
  size_t errors = 0;
  glm::ivec3 min = p0.getMin() - glm::ivec3(1, 1, 1);
  glm::ivec3 max = p0.getMax() + glm::ivec3(1, 1, 1);
  for (int hq = 0; hq < df->getQ(); hq++, hq++)
    for (int hz = min.z, rz = 0; hz < max.z; hz++, rz++)
      for (int hy = min.y, ry = 0; hy < max.y; hy++, ry++)
        for (int hx = min.x, rx = 0; hx < max.x; hx++, rx++) {
          real a = ref[rz][ry][rx];
          real b = (*df)(p0, hq, hx, hy, hz);
          // EXPECT_EQ(a, b);
          if (a != b) errors++;
        }
  return errors;
}

TEST(DistributedDFTest, HaloExchangeMultiGPU) {
  // int maxDevices = 2, nq = 19, nx = 3, ny = 5, nz = 2;
  int maxDevices = 2, nq = 19, nx = 5, ny = 3, nz = 2;

  int numDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));
}
