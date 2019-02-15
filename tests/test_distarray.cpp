#include <gtest/gtest.h>

#include <omp.h>

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"
#include "KernelInterface.hpp"
#include "test_kernel.hpp"

TEST(DistributionArrayTest, ScatterGather) {
  int numDevices = 10, nq = 3, nx = 50, ny = 165, nz = 30;

  int maxDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&maxDevices));
  numDevices = min(numDevices, maxDevices);

  // Create a large array on GPU0 and fill it with some numbers
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real> *fullArray =
      new DistributionArray<real>(nq, nx, ny, nz);
  SubLattice fullLattice = fullArray->getSubLattice(0, 0, 0);
  fullArray->allocate(fullLattice);
  for (int q = 0; q < fullArray->getQ(); q++) fullArray->fill(q, -10);
  runTestKernel(fullArray, fullLattice, 1);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Create as many DF groups as there are GPUs
  DistributionFunction *arrays[numDevices];

  // Scatter the large array to partitions
#pragma omp parallel num_threads(CUDA_MAX_P2P_DEVS)
#pragma omp for
  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction *df =
        new DistributionFunction(nq, nx, ny, nz, numDevices);
    arrays[srcDev] = df;
    SubLattice subLattice = df->getDeviceSubLattice(srcDev);
    df->allocate(subLattice);
    for (int q = 0; q < df->getQ(); q++) df->fill(q, -srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->scatter(*fullArray, subLattice);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Create a second large empty array
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real> *newFullArray =
      new DistributionArray<real>(nq, nx, ny, nz);
  SubLattice newFullLattice = newFullArray->getSubLattice(0, 0, 0);
  newFullArray->allocate(newFullLattice);
  for (int q = 0; q < newFullArray->getQ(); q++) newFullArray->fill(q, -20);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Gather the partitions into the new large array
#pragma omp parallel num_threads(CUDA_MAX_P2P_DEVS)
#pragma omp for
  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction *df = arrays[srcDev];
    SubLattice subLattice = df->getDeviceSubLattice(srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->gather(subLattice, newFullArray);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Compare the new and old large array
  CUDA_RT_CALL(cudaSetDevice(0));
  fullArray->download();
  newFullArray->download();

  for (int q = 0; q < nq; q++)
    for (int x = 0; x < nx; x++)
      for (int y = 0; y < ny; y++)
        for (int z = 0; z < nz; z++) {
          real a = (*fullArray)(fullLattice, q, x, y, z);
          real b = (*newFullArray)(newFullLattice, q, x, y, z);
          if (a != b)
            FAIL() << "Distribution function not equal at q=" << q
                   << ", x=" << x << ", y=" << y << ", z=" << z
                   << ", orig=" << a << ", new=" << b << std::endl;
        }

  delete fullArray;
  delete newFullArray;
  for (int i = 0; i < numDevices; i++) delete arrays[i];
}

TEST(DistributionArrayTest, ScatterGatherSlice) {
  int numDevices = 2, nq = 1, nx = 6, ny = 5, nz = 4;
  glm::ivec3 slicePos(2, 2, 2);

  int maxDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&maxDevices));
  numDevices = min(numDevices, maxDevices);

  // Create a large array on GPU0 and fill it with some numbers
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real> *fullArray =
      new DistributionArray<real>(nq, nx, ny, nz);
  SubLattice fullLattice = fullArray->getSubLattice(0, 0, 0);
  fullArray->allocate(fullLattice);
  for (int q = 0; q < fullArray->getQ(); q++) fullArray->fill(q, 0);
  runTestKernel(fullArray, fullLattice, 1);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Create as many DF groups as there are GPUs
  DistributionFunction *arrays[numDevices];

  // Scatter the large array to partitions
#pragma omp parallel num_threads(CUDA_MAX_P2P_DEVS)
#pragma omp for
  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction *df =
        new DistributionFunction(nq, nx, ny, nz, numDevices);
    arrays[srcDev] = df;
    SubLattice subLattice = df->getDeviceSubLattice(srcDev);
    df->allocate(subLattice);
    for (int q = 0; q < df->getQ(); q++) df->fill(q, -srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->scatter(*fullArray, subLattice);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Create a second large empty array
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real> *newFullArray =
      new DistributionArray<real>(nq, nx, ny, nz);
  SubLattice newFullLattice = newFullArray->getSubLattice(0, 0, 0);
  newFullArray->allocate(newFullLattice);
  for (int q = 0; q < newFullArray->getQ(); q++) newFullArray->fill(q, 0);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Gather the partitions into the new large array
#pragma omp parallel num_threads(CUDA_MAX_P2P_DEVS)
#pragma omp for
  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction *df = arrays[srcDev];
    SubLattice subLattice = df->getDeviceSubLattice(srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->gatherSlice(slicePos, 0, 0, subLattice, newFullArray);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Compare the new and old large array
  CUDA_RT_CALL(cudaSetDevice(0));
  fullArray->download();
  newFullArray->download();

  std::cout << *newFullArray << std::endl;

  // for (int q = 0; q < nq; q++)
  //   for (int x = 0; x < nx; x++)
  //     for (int y = 0; y < ny; y++)
  //       for (int z = 0; z < nz; z++) {
  //         real a = (*fullArray)(fullLattice, q, x, y, z);
  //         real b = (*newFullArray)(newFullLattice, q, x, y, z);
  //         if (a != b)
  //           FAIL() << "Distribution function not equal at q=" << q
  //                  << ", x=" << x << ", y=" << y << ", z=" << z
  //                  << ", orig=" << a << ", new=" << b << std::endl;
  //       }

  // delete fullArray;
  // delete newFullArray;
  // for (int i = 0; i < numDevices; i++) delete arrays[i];
}
