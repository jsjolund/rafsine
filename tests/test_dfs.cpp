#include "test_cuda.hpp"

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"
#include "DistributionFunction.hpp"
#include "KernelInterface.hpp"
#include "Primitives.hpp"
#include "test_kernel.hpp"

namespace cudatest {
class DistributionArrayTest : public CudaTest {};

TEST_F(DistributionArrayTest, GatherTest2) {
  VoxelVolume vol1("test", vec3<int>(1, 1, 1), vec3<int>(2, 2, 2));
  VoxelVolumeArray *avgAreas = new VoxelVolumeArray();
  avgAreas->push_back(vol1);

  int avgsTotalSize = 0;
  for (int i = 0; i < avgAreas->size(); i++) {
    VoxelVolume avg = avgAreas->at(i);
    glm::ivec3 dims = avg.getDims();
    avgsTotalSize += dims.x * dims.y * dims.z;
  }
  DistributionArray<real> avgArray(4, avgsTotalSize, 1, 1);
  avgArray.allocate();
  avgArray.fill(1);
  avgArray.download();

  ASSERT_EQ(avgArray.size(avgArray.getSubLattice()), 4 * avgsTotalSize);

  std::cout << avgArray << std::endl;
}

TEST_F(DistributionArrayTest, GatherTest) {
  const int maxDevices = 9, nq = 1, nx = 4, ny = 20, nz = 4;
  int numDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));
  P2PLattice lattice(nx, ny, nz, numDevices);

  DistributionArray<real> *arrays[maxDevices];

  VoxelVolume area("testArea", vec3<int>(1, 1, 1), vec3<int>(3, 19, 3),
                   vec3<real>(0, 0, 0), vec3<real>(0, 0, 0));
  glm::ivec3 adims = area.getDims();
  DistributionArray<real> *areaArray =
      new DistributionArray<real>(nq, adims.x, adims.y, adims.z);
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

    for (int q = 0; q < nq; q++)
      array->gather(area.getMin(), area.getMax(), q, q, subLattice, areaArray,
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

TEST_F(DistributionArrayTest, ScatterGather) {
  int numDevices = 9;
  const int nq = 3, nx = 50, ny = 65, nz = 11;

  int maxDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&maxDevices));
  numDevices = min(numDevices, maxDevices);

  // Create a large array on GPU0 and fill it with some numbers
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real> *fullArray =
      new DistributionArray<real>(nq, nx, ny, nz);
  SubLattice fullLattice = fullArray->getSubLattice(0, 0, 0);
  fullArray->allocate(fullLattice);
  fullArray->fill(-10);
  runTestKernel(fullArray, fullLattice, 1);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Create as many DF groups as there are GPUs
  DistributionFunction *arrays[numDevices];

  // Scatter the large array to partitions
#pragma omp parallel num_threads(9)
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
    df->fill(-srcDev);

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
  newFullArray->fill(-20);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Gather the partitions into the new large array
#pragma omp parallel num_threads(9)
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

TEST_F(DistributionArrayTest, ScatterGatherSlice) {
  int numDevices = 2;
  const int nq = 1, nx = 6, ny = 5, nz = 4;
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
  fullArray->fill(0);
  runTestKernel(fullArray, fullLattice, 1);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Create as many DF groups as there are GPUs
  DistributionFunction *arrays[numDevices];

  // Scatter the large array to partitions
#pragma omp parallel num_threads(9)
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
    df->fill(-srcDev);

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
  newFullArray->fill(0);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Gather the partitions into the new large array
#pragma omp parallel num_threads(9)
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

  for (int q = 0; q < nq; q++) {
    const int x = slicePos.x;
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        real a = (*fullArray)(fullLattice, q, x, y, z);
        real b = (*newFullArray)(newFullLattice, q, x, y, z);
        if (a != b) {
          FAIL() << "Distribution function not equal at q=" << q << ", x=" << x
                 << ", y=" << y << ", z=" << z << ", orig=" << a
                 << ", new=" << b << std::endl;
        }
      }
    }
  }
  for (int q = 0; q < nq; q++) {
    for (int x = 0; x < nx; x++) {
      const int y = slicePos.y;
      for (int z = 0; z < nz; z++) {
        real a = (*fullArray)(fullLattice, q, x, y, z);
        real b = (*newFullArray)(newFullLattice, q, x, y, z);
        if (a != b) {
          FAIL() << "Distribution function not equal at q=" << q << ", x=" << x
                 << ", y=" << y << ", z=" << z << ", orig=" << a
                 << ", new=" << b << std::endl;
        }
      }
    }
  }
  for (int q = 0; q < nq; q++) {
    for (int x = 0; x < nx; x++) {
      for (int y = 0; y < ny; y++) {
        const int z = slicePos.z;
        real a = (*fullArray)(fullLattice, q, x, y, z);
        real b = (*newFullArray)(newFullLattice, q, x, y, z);
        if (a != b) {
          FAIL() << "Distribution function not equal at q=" << q << ", x=" << x
                 << ", y=" << y << ", z=" << z << ", orig=" << a
                 << ", new=" << b << std::endl;
        }
      }
    }
  }
  for (int q = 0; q < nq; q++) {
    for (int x = 0; x < nx; x++) {
      for (int y = 0; y < ny; y++) {
        for (int z = 0; z < nz; z++) {
          if (x == slicePos.x || y == slicePos.y || z == slicePos.z) continue;
          real a = (*newFullArray)(newFullLattice, q, x, y, z);
          if (a != 0) {
            FAIL() << "Distribution function not equal at q=" << q
                   << ", x=" << x << ", y=" << y << ", z=" << z
                   << ", orig=" << 0 << ", new=" << a << std::endl;
          }
        }
      }
    }
  }

  delete fullArray;
  delete newFullArray;
  for (int i = 0; i < numDevices; i++) delete arrays[i];
}

}  // namespace cudatest
