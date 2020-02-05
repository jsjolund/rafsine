#include "test_cuda.hpp"

#include "cuda_histogram.h"

namespace cudatest {

TEST_F(CudaTest, Histogram) {
  CUDA_RT_CALL(cudaSetDevice(0));
  // Allocate an array on CPU-side to fill data of 10 indices
  float h_data[] = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};
  // Allocate an array on GPU-memory to hold the input
  float* d_data = NULL;
  CUDA_RT_CALL(cudaMalloc(&d_data, 20 * sizeof(float)));
  // Copy the input-data to the GPU
  CUDA_RT_CALL(
      cudaMemcpy(d_data, h_data, 20 * sizeof(float), cudaMemcpyHostToDevice));

  // Init the result-array on host-side to zero (we have 6 bins, from 0 to 5):
  int res[4] = {0};

  // Create the necessary function objects and run histogram using them
  // 1 result per input index, 6 bins, 10 inputs
  test_xform xform;
  xform.m_min = 10.0;
  xform.m_max = 19.0;
  xform.m_nbins = 4;
  test_sumfun sum;
  callHistogramKernel<histogram_atomic_inc, 1>(d_data, xform, sum, 0, 20, 0,
                                               &res[0], 4);
  printf("Results: [%d, %d, %d, %d]\n", res[0], res[1], res[2], res[3]);
  // Done: Let OS + GPU-driver+runtime worry about resource-cleanup
}

TEST_F(CudaTest, ThrustGather1) {
  CUDA_RT_CALL(cudaSetDevice(0));
  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  // gather all even indices into the first half of the range
  // and odd indices to the last half of the range
  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10);
  thrust::gather(thrust::device, d_map.begin(), d_map.end(), d_values.begin(),
                 d_output.begin());
  thrust::host_vector<int> h_output = d_output;
  for (int i = 0; i < h_output.size(); i++) std::cout << h_output[i] << ", ";
  std::cout << std::endl;
}

TEST_F(CudaTest, ThrustGather2) {
  CUDA_RT_CALL(cudaSetDevice(0));
  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  int map[10] = {0, 2, 4, 6, 8};
  thrust::device_vector<int> d_map(map, map + 5);
  thrust::device_vector<int> d_output(5);
  thrust::gather(thrust::device, d_map.begin(), d_map.end(), d_values.begin(),
                 d_output.begin());
  thrust::host_vector<int> h_output = d_output;
  for (int i = 0; i < h_output.size(); i++) std::cout << h_output[i] << ", ";
  std::cout << std::endl;
}

TEST_F(CudaTest, ExplicitCopyArray) {
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

TEST_F(CudaTest, ExplicitCopyThrustArray) {
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
  int* srcVecptr = thrust::raw_pointer_cast(&srcVec[0]);

  // Allocate memory on gpu1 and set it to some other values
  CUDA_RT_CALL(cudaSetDevice(dstDev));
  CUDA_RT_CALL(cudaDeviceEnablePeerAccess(srcDev, 0));
  thrust::device_vector<int> dstVec(numInts);
  thrust::fill(dstVec.begin(), dstVec.end(), 0);
  int* dstVecptr = thrust::raw_pointer_cast(&dstVec[0]);

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

TEST_F(CudaTest, GradientTransform) {
  real min = -100;
  real max = 120;
  int sizeX = 10;
  int sizeY = 3;
  thrust::host_vector<real> plot(sizeX * sizeY);
  // Calculate ticks between min and max value
  real Dx = (max - min) / (real)(sizeX * sizeY - 1);
  if (min != max) {
    // Draw the gradient plot
    thrust::transform(thrust::make_counting_iterator(min / Dx),
                      thrust::make_counting_iterator((max + Dx) / Dx),
                      thrust::make_constant_iterator(Dx), plot.begin(),
                      thrust::multiplies<real>());
  }
  // std::cout << Dx << std::endl;
  // thrust::copy(plot.begin(), plot.end(),
  //              std::ostream_iterator<float>(std::cout, " "));
  // std::cout << std::endl;
  ASSERT_EQ(plot[0], -100);
  ASSERT_EQ(plot[sizeX * sizeY - 1], 120);
  for (int i = 0; i < sizeX * sizeY - 1; i++) {
    ASSERT_LT(plot[i], plot[i + 1]);
  }
}

TEST_F(CudaTest, RemoveIfNaN) {
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

template <typename T>
struct division : public thrust::unary_function<T, T> {
  const T m_arg;
  __host__ __device__ T operator()(const T& x) const { return x / m_arg; }
  explicit division(T arg) : m_arg(arg) {}
};

TEST_F(CudaTest, Average) {
  float data[6] = {10.0, 20.0, 10.0, 20.0, 10.0, 20.0};
  float result =
      thrust::transform_reduce(thrust::host, data, data + 6, division<float>(6),
                               static_cast<float>(0), thrust::plus<float>());
  ASSERT_EQ(result, 15.0);
}

}  // namespace cudatest
