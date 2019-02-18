#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <thrust/execution_policy.h>
#include <thrust/scatter.h>

#include "CudaUtils.hpp"
#include "Primitives.hpp"

TEST(CudaTest, ExplicitCopyArray) {
  int numDevices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  ASSERT_GE(numDevices, 2);

  const int srcDev = 0, dstDev = 1;
  const size_t numChars = 1000;
  char *srcVec, *dstVec, *hostVec;

  // Allocate memory on gpu0 and set it to some value
  CUDA_RT_CALL(cudaSetDevice(srcDev));
  CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
  CUDA_RT_CALL(cudaMalloc(&srcVec, numChars * sizeof(char)));
  CUDA_RT_CALL(cudaMemset(srcVec, 'y', numChars * sizeof(char)));
  // Allocate memory on gpu1 and set it to some other value
  CUDA_RT_CALL(cudaSetDevice(dstDev));
  CUDA_RT_CALL(cudaDeviceEnablePeerAccess(srcDev, 0));
  CUDA_RT_CALL(cudaMalloc(&dstVec, numChars * sizeof(char)));
  CUDA_RT_CALL(cudaMemset(dstVec, 'n', numChars * sizeof(char)));
  // Copy P2P
  CUDA_RT_CALL(
      cudaMemcpyPeer(dstVec, dstDev, srcVec, srcDev, numChars * sizeof(char)));
  // Allocate memory on host, copy from gpu1 to host, and verify P2P copy
  CUDA_RT_CALL(cudaMallocHost(&hostVec, numChars * sizeof(char)));
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaMemcpy(hostVec, dstVec, numChars * sizeof(char),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < numChars; i++) ASSERT_EQ(hostVec[i], 'y');

  CUDA_RT_CALL(cudaSetDevice(srcDev));
  CUDA_RT_CALL(cudaDeviceDisablePeerAccess(dstDev));
  CUDA_RT_CALL(cudaSetDevice(dstDev));
  CUDA_RT_CALL(cudaDeviceDisablePeerAccess(srcDev));
}

TEST(CudaTest, ExplicitCopyThrustArray) {
  int numDevices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  ASSERT_GE(numDevices, 2);

  const int srcDev = 0, dstDev = 1;
  const size_t numInts = 1000;

  // Allocate memory on gpu0 and set it to some values
  CUDA_RT_CALL(cudaSetDevice(srcDev));
  CUDA_RT_CALL(cudaDeviceEnablePeerAccess(dstDev, 0));
  thrust::device_vector<int> srcVec(numInts);
  thrust::sequence(srcVec.begin(), srcVec.end(), 23);
  int *srcVecptr = thrust::raw_pointer_cast(&srcVec[0]);

  // Allocate memory on gpu1 and set it to some other values
  CUDA_RT_CALL(cudaSetDevice(dstDev));
  CUDA_RT_CALL(cudaDeviceEnablePeerAccess(srcDev, 0));
  thrust::device_vector<int> dstVec(numInts);
  thrust::fill(dstVec.begin(), dstVec.end(), 0);
  int *dstVecptr = thrust::raw_pointer_cast(&dstVec[0]);

  // Copy P2P
  CUDA_RT_CALL(cudaMemcpyPeer(dstVecptr, dstDev, srcVecptr, srcDev,
                              numInts * sizeof(int)));

  // Allocate memory on host, copy from gpu1 to host, and verify P2P copy
  CUDA_RT_CALL(cudaDeviceSynchronize());
  thrust::host_vector<int> hostVec(numInts);
  hostVec = dstVec;
  for (int i = 0; i < numInts; i++) ASSERT_EQ(hostVec[i], 23 + i);

  CUDA_RT_CALL(cudaSetDevice(srcDev));
  CUDA_RT_CALL(cudaDeviceDisablePeerAccess(dstDev));
  CUDA_RT_CALL(cudaSetDevice(dstDev));
  CUDA_RT_CALL(cudaDeviceDisablePeerAccess(srcDev));
}

TEST(CudaTest, GradientTransform) {
  real min = -100;
  real max = 120;
  int sizeX = 10;
  int sizeY = 3;
  thrust::host_vector<real> plot(sizeX * sizeY);
  // Calculate ticks between min and max value
  real Dx = (max - min) / (real)(sizeX * sizeY - 1);
  std::cout << Dx << std::endl;
  if (min != max) {
    // Draw the gradient plot
    thrust::transform(thrust::make_counting_iterator(min / Dx),
                      thrust::make_counting_iterator((max + Dx) / Dx),
                      thrust::make_constant_iterator(Dx), plot.begin(),
                      thrust::multiplies<real>());
  }
  thrust::copy(plot.begin(), plot.end(),
               std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;
  ASSERT_EQ(plot[0], -100);
  ASSERT_EQ(plot[sizeX * sizeY - 1], 120);
  for (int i = 0; i < sizeX * sizeY - 1; i++) {
    ASSERT_LT(plot[i], plot[i + 1]);
  }
}

TEST(CudaTest, RemoveIfNaN) {
  CUDA_RT_CALL(cudaSetDevice(0));
  thrust::device_vector<real> gpuVec(6);
  gpuVec[0] = -1;
  gpuVec[1] = NaN;
  gpuVec[2] = 2;
  gpuVec[3] = 99;
  gpuVec[4] = NaN;
  gpuVec[5] = 100;
  real max = *thrust::max_element(
      gpuVec.begin(),
      thrust::remove_if(gpuVec.begin(), gpuVec.end(), CUDA_isNaN()));
  real min = *thrust::min_element(
      gpuVec.begin(),
      thrust::remove_if(gpuVec.begin(), gpuVec.end(), CUDA_isNaN()));
  ASSERT_EQ(max, 100);
  ASSERT_EQ(min, -1);
}
