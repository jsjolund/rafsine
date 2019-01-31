#include "test_kernel.hpp"

/**
 * @brief Simple kernel which puts sequential numbers in array
 */
__global__ void TestKernel(real *__restrict__ df, glm::ivec3 pMin,
                           glm::ivec3 pMax, glm::ivec3 pHalo) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y;
  glm::ivec3 p(x, y, z);
  glm::ivec3 dfSize = pMax - pMin;
  if ((p.x >= dfSize.x) || (p.y >= dfSize.y) || (p.z >= dfSize.z)) return;
  real value = 1 + I3D(x, y, z, dfSize.x, dfSize.y, dfSize.z);
  glm::ivec3 arrSize = dfSize + pHalo;
  df[I4D(0, p.x, p.y, p.z, arrSize.x, arrSize.y, arrSize.z)] = value;
}

/**
 * @brief Launcher for the test kernel
 */
void runTestKernel(DistributionArray *df, SubLattice subLattice,
                   cudaStream_t stream) {
  glm::ivec3 n = subLattice.getLatticeDims();
  glm::ivec3 m = subLattice.getHalo();
  dim3 gridSize(n.y + m.y, n.z + m.z, 1);
  dim3 blockSize(n.x + m.x, 1, 1);
  for (int q = 0; q < df->getQ(); q++)
    TestKernel<<<gridSize, blockSize, 0, stream>>>(
        df->gpu_ptr(subLattice, q), subLattice.getLatticeMin(),
        subLattice.getLatticeMax(), subLattice.getHalo());
}
