#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "DistributionFunction.hpp"
#include "KernelInterface.hpp"
#include "Primitives.hpp"
#include "test_kernel.hpp"

TEST(DistributionArray, CopyTest) {
  const int maxDevices = 9, nq = 1, nx = 5, ny = 18, nz = 3;
  int numDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));
  P2PLattice lattice(nx, ny, nz, numDevices);

  DistributionArray<real> *arrays[maxDevices];

  VoxelArea area("testArea", vec3<int>(1, 1, 1), vec3<int>(4, 17, 2),
                 vec3<real>(0, 0, 0), vec3<real>(0, 0, 0));
  glm::ivec3 adims = area.getDims();
  DistributionArray<real> *areaArray =
      new DistributionArray<real>(1, adims.x, adims.y, adims.z);
  areaArray->allocate(areaArray->getSubLattice(0, 0, 0));
  areaArray->fill(0);

#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    const SubLattice s = lattice.getDeviceSubLattice(srcDev);
    const SubLattice subLattice(s.getMin(), s.getMax(), glm::ivec3(0, 0, 0));
    const glm::ivec3 dims = subLattice.getDims();

    DistributionArray<real> *array =
        new DistributionArray<real>(nq, nx, ny, nz, numDevices);
    arrays[srcDev] = array;
    array->allocate(subLattice);
    array->fill(0);

    runTestKernel(array, subLattice, srcDev * dims.x * dims.y * dims.z);

    array->gather(area.getMin(), area.getMax(), 0, 0, subLattice, areaArray,
                  areaArray->getSubLattice(0, 0, 0));
  }

  for (int i = 0; i < numDevices; i++) {
    arrays[i]->download();
    std::cout << "Device " << i << std::endl;
    std::cout << *arrays[i] << std::endl;
  }
  areaArray->download();
  std::cout << "Area" << std::endl;
  std::cout << *areaArray << std::endl;
}
