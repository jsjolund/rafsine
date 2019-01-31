#include <gtest/gtest.h>

#include <omp.h>

#include "DistributionArray.hpp"
#include "KernelInterface.hpp"
#include "test_kernel.hpp"

TEST(DistributionArrayTest, GatherInto) {
  int numDevices = 2, nq = 2, nx = 3, ny = 4, nz = 2;

  int maxDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&maxDevices));
  numDevices = min(numDevices, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));

  // Create the full array on GPU0
  DistributionArray *fullArray = new DistributionArray(nq, nx, ny, nz);
  SubLattice fullLattice = fullArray->getSubLattices().at(0);
  fullArray->allocate(fullLattice);

  // Create as many DF groups as there are GPUs
  DistributionFunction *arrays[numDevices];

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction *df =
        new DistributionFunction(nq, nx, ny, nz, numDevices);
    arrays[srcDev] = df;
    SubLattice subLattice = df->getSubLatticeFromDevice(srcDev);
    df->allocate(subLattice);
    for (int q = 0; q < df->getQ(); q++) df->fill(q, 0);
    runTestKernel(df, subLattice, srcDev + 1);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    std::vector<bool> peerAccessList(numDevices);
    enablePeerAccess(srcDev, 0, &peerAccessList);
    df->gatherInto(fullArray);
    disablePeerAccess(srcDev, &peerAccessList);
  }

  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));

    DistributionFunction *df = arrays[srcDev];
    df->download();

    std::cout << "######################## Device " << srcDev << std::endl;
    std::cout << *df << std::endl;
  }

  CUDA_RT_CALL(cudaSetDevice(0));
  fullArray->download();
  std::cout << "######################## Full Array " << std::endl;
  std::cout << *fullArray << std::endl;
  delete fullArray;
}
