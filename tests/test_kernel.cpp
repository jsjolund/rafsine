#include "test_kernel.hpp"

__device__ void runKernel(const int x,
                          const int y,
                          const int z,
                          const int nx,
                          const int ny,
                          const int nz,
                          real_t* __restrict__ df,
                          int offset) {
  real_t value = (1 + I3D(x, y, z, nx, ny, nz)) + offset;
  df[I4D(0, x, y, z, nx, ny, nz)] = value;
}

/**
 * @brief Simple kernel which puts sequential numbers in array
 */
__global__ void TestKernel(Partition partition,
                           real_t* __restrict__ df,
                           int offset) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y;
  Vector3<size_t> p0(x, y, z);
  Vector3<size_t> pSize = partition.getExtents();
  if ((p0.x() >= pSize.x()) || (p0.y() >= pSize.y()) || (p0.z() >= pSize.z()))
    return;
  Vector3<size_t> p1 = p0 + partition.getGhostLayer();
  Vector3<size_t> arrSize = partition.getArrayExtents();
  runKernel(p1.x(), p1.y(), p1.z(), arrSize.x(), arrSize.y(), arrSize.z(), df,
            offset);
}

// /**
//  * @brief Simple kernel which puts sequential numbers in array
//  */
// __global__ void TestBoundaryKernel(Partition partition, real_t *__restrict__
// df,
//                                    int offset) {
//   const int n = blockIdx.x;
//   const int side = threadIdx.x;
//   const int i = n + side * partition.getNumBoundaryElements() / 2;
//   Vector3<int> p0;
//   partition.getBoundaryElement(i, &p0.x, &p0.y, &p0.z);
//   // Vector3<int> pSize = partition.getExtents();
//   // if ((p0.x >= pSize.x) || (p0.y >= pSize.y) || (p0.z >= pSize.z)) return;
//   Vector3<int> p1 = p0 + partition.getGhostLayer();
//   Vector3<int> arrSize = partition.getArrayExtents();
//   runKernel(p1.x, p1.y, p1.z, arrSize.x, arrSize.y, arrSize.z, df, offset);
// }

/**
 * @brief Launcher for the test kernel
 */
void runTestKernel(DistributionArray<real_t>* df,
                   Partition partition,
                   int offset,
                   cudaStream_t stream) {
  Vector3<size_t> n = partition.getExtents();
  dim3 gridSize(n.y(), n.z(), 1);
  dim3 blockSize(n.x(), 1, 1);

  for (size_t q = 0; q < df->getQ(); q++) {
    TestKernel<<<gridSize, blockSize, 0, stream>>>(
        partition, df->gpu_ptr(partition, q), offset);
    CUDA_CHECK_ERRORS("TestKernel");
  }
}

// /**
//  * @brief Launcher for the test kernel
//  */
// void runBoundaryTestKernel(DistributionArray<real_t> *df, Partition
// partition,
//                            int offset, cudaStream_t stream) {
//   int n = partition.getNumBoundaryElements() / 2;
//   dim3 gridSize(n, 1, 1);
//   dim3 blockSize(2, 1, 1);

//   for (int q = 0; q < df->getQ(); q++) {
//     TestBoundaryKernel<<<gridSize, blockSize, 0, stream>>>(
//         partition, df->gpu_ptr(partition, q), offset * (q + 1));
//     CUDA_CHECK_ERRORS("TestKernel");
//   }
// }
