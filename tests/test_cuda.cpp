#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "CudaUtils.hpp"

TEST(DistributedDF, ExplicitCopyArray) {
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

TEST(DistributedDF, ExplicitCopyThrustArray) {
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

TEST(DistributedDF, ExplicitCopyThrustArrayAllToAll) {
  int numDevices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, 8);  // Limit to 8 for this test
  ASSERT_GE(numDevices, 2);

  typedef thrust::device_vector<int> dvec;
  std::vector<dvec *> deviceVectors(numDevices);

  bool success = true;

#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));
#pragma omp barrier
    for (int dstDev = 0; dstDev < numDevices; dstDev++) {
      if (dstDev != srcDev) {
        int canAccessPeer = 0;
        cudaError_t cudaPeerAccessStatus;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, srcDev, dstDev));
        if (canAccessPeer) {
          cudaPeerAccessStatus = cudaDeviceEnablePeerAccess(dstDev, 0);
        }
        if (!cudaDeviceCanAccessPeer || cudaPeerAccessStatus != cudaSuccess) {
#pragma omp critical
          {
            if (success) success = false;
          }
        }
      }
    }
    dvec *srcVec = new dvec(numDevices);
    deviceVectors.at(srcDev) = srcVec;
    thrust::fill(srcVec->begin(), srcVec->end(), -1);
#pragma omp barrier
    (*srcVec)[srcDev] = srcDev;
    int *srcPtr = thrust::raw_pointer_cast(&(*srcVec)[srcDev]);
    for (int dstDev = 0; dstDev < numDevices; dstDev++) {
      dvec *dstVec = deviceVectors.at(dstDev);
      int *dstPtr = thrust::raw_pointer_cast(&(*dstVec)[srcDev]);
      if (dstDev == srcDev) {
        CUDA_RT_CALL(
            cudaMemcpy(dstPtr, srcPtr, sizeof(int), cudaMemcpyDeviceToDevice));
      } else {
        CUDA_RT_CALL(
            cudaMemcpyPeer(dstPtr, dstDev, srcPtr, srcDev, sizeof(int)));
      }
    }
#pragma omp barrier
    {
      for (int i = 0; i < numDevices; i++) {
        if (success && (*srcVec)[i] != i) success = false;
      }
    }
#pragma omp barrier
#pragma omp master
    {
      delete srcVec;
      CUDA_RT_CALL(cudaDeviceReset());
    }
  }
  ASSERT_TRUE(success);
}
