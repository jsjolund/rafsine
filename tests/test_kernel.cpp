#include "test_kernel.hpp"

__device__ void runKernel(const int x, const int y, const int z, const int nx,
                          const int ny, const int nz, real *__restrict__ df,
                          int offset) {
  real value = (1 + I3D(x, y, z, nx, ny, nz)) + offset;
  df[I4D(0, x, y, z, nx, ny, nz)] = value;
}

/**
 * @brief Simple kernel which puts sequential numbers in array
 */
__global__ void TestKernel(SubLattice subLattice, real *__restrict__ df,
                           int offset) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y;
  glm::ivec3 p0(x, y, z);
  glm::ivec3 pSize = subLattice.getDims();
  if ((p0.x >= pSize.x) || (p0.y >= pSize.y) || (p0.z >= pSize.z)) return;
  glm::ivec3 p1 = p0 + subLattice.getHalo();
  glm::ivec3 arrSize = subLattice.getArrayDims();
  runKernel(p1.x, p1.y, p1.z, arrSize.x, arrSize.y, arrSize.z, df, offset);
}

/**
 * @brief Simple kernel which puts sequential numbers in array
 */
__global__ void TestBoundaryKernel(SubLattice subLattice, real *__restrict__ df,
                                   int offset) {
  const int n = blockIdx.x;
  const int side = threadIdx.x;
  const int i = n + side * subLattice.getNumBoundaryElements() / 2;
  glm::ivec3 p0;
  subLattice.getBoundaryElement(i, &p0.x, &p0.y, &p0.z);
  // glm::ivec3 pSize = subLattice.getDims();
  // if ((p0.x >= pSize.x) || (p0.y >= pSize.y) || (p0.z >= pSize.z)) return;
  glm::ivec3 p1 = p0 + subLattice.getHalo();
  glm::ivec3 arrSize = subLattice.getArrayDims();
  runKernel(p1.x, p1.y, p1.z, arrSize.x, arrSize.y, arrSize.z, df, offset);
}

/**
 * @brief Launcher for the test kernel
 */
void runTestKernel(DistributionArray<real> *df, SubLattice subLattice,
                   int offset, cudaStream_t stream) {
  glm::ivec3 n = subLattice.getDims();
  dim3 gridSize(n.y, n.z, 1);
  dim3 blockSize(n.x, 1, 1);

  for (int q = 0; q < df->getQ(); q++) {
    TestKernel<<<gridSize, blockSize, 0, stream>>>(
        subLattice, df->gpu_ptr(subLattice, q), offset);
    CUDA_CHECK_ERRORS("TestKernel");
  }
}

/**
 * @brief Launcher for the test kernel
 */
void runBoundaryTestKernel(DistributionArray<real> *df, SubLattice subLattice,
                           int offset, cudaStream_t stream) {
  int n = subLattice.getNumBoundaryElements() / 2;
  dim3 gridSize(n, 1, 1);
  dim3 blockSize(2, 1, 1);

  for (int q = 0; q < df->getQ(); q++) {
    TestBoundaryKernel<<<gridSize, blockSize, 0, stream>>>(
        subLattice, df->gpu_ptr(subLattice, q), offset * (q + 1));
    CUDA_CHECK_ERRORS("TestKernel");
  }
}
