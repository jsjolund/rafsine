#include "test_kernel.hpp"

/**
 * @brief Simple kernel which puts sequential numbers in array
 */
__global__ void TestKernel(real *__restrict__ df, glm::ivec3 pMin,
                           glm::ivec3 pMax, glm::ivec3 pHalo, int scl) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y;
  glm::ivec3 p0(x, y, z);
  glm::ivec3 pSize = pMax - pMin;
  if ((p0.x >= pSize.x) || (p0.y >= pSize.y) || (p0.z >= pSize.z)) return;
  glm::ivec3 p1 = p0 + pHalo;
  glm::ivec3 arrSize = pMax - pMin + pHalo * 2;
  real value = (1 + I3D(x, y, z, pSize.x, pSize.y, pSize.z)) * scl;
  df[I4D(0, p1.x, p1.y, p1.z, arrSize.x, arrSize.y, arrSize.z)] = value;
}

/**
 * @brief Launcher for the test kernel
 */
void runTestKernel(DistributionArray<real> *df, SubLattice subLattice, int scl,
                   cudaStream_t stream) {
  glm::ivec3 n = subLattice.getDims();
  glm::ivec3 m = subLattice.getHalo();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);

  for (int q = 0; q < df->getQ(); q++) {
    TestKernel<<<gridSize, blockSize, 0, stream>>>(
        df->gpu_ptr(subLattice, q), subLattice.getMin(), subLattice.getMax(),
        subLattice.getHalo(), scl * (q + 1));
    CUDA_CHECK_ERRORS("TestKernel");
  }
}

// /**
//  * @brief Launcher for the test kernel
//  */
// void runBorderTestKernel(DistributionArray<real> *df, SubLattice subLattice,
//                          int scl, cudaStream_t stream) {
//   glm::ivec3 n = subLattice.getDims();
//   glm::ivec3 m = subLattice.getHalo();
//   dim3 gridSize(n.y, n.z, 1);
//   dim3 blockSize(n.x, 1, 1);

//   for (int q = 0; q < df->getQ(); q++) {
//     TestKernel<<<gridSize, blockSize, 0, stream>>>(
//         df->gpu_ptr(subLattice, q), subLattice.getMin(),
//         subLattice.getMax(), subLattice.getHalo(), scl * (q + 1));
//     CUDA_CHECK_ERRORS("TestKernel");
//   }
// }