#include "test_cuda.hpp"

#include <thrust/gather.h>

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"
#include "DistributionFunction.hpp"
#include "KernelInterface.hpp"
#include "Primitives.hpp"
#include "test_kernel.hpp"

namespace cudatest {
class DistributionArrayTest : public CudaTest {};

TEST_F(DistributionArrayTest, GatherTest2) {
  // Create lattice
  const int maxDevices = 10, nq = 2, nx = 4, ny = 20, nz = 4;
  int numDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // Initialize lattice
  P2PLattice lattice(nx, ny, nz, numDevices);
  DistributionFunction *arrays[numDevices];

  // Define some averaging areas
  VoxelVolumeArray avgVols;
  VoxelVolume vol1("test1", vec3<int>(1, 1, 1), vec3<int>(2, 5, 2));
  VoxelVolume vol2("test2", vec3<int>(nx - 1, ny - 1, nz - 1),
                   vec3<int>(nx, ny, nz));
  avgVols.push_back(vol1);
  avgVols.push_back(vol2);

  // Combine averaging areas into one array
  int avgSizeTotal = 0;
  for (int avgIdx = 0; avgIdx < avgVols.size(); avgIdx++) {
    glm::ivec3 aDims = avgVols.at(avgIdx).getDims();
    avgSizeTotal += aDims.x * aDims.y * aDims.z;
  }
  DistributionArray<real> avgArray(nq, avgSizeTotal, 1, 1);
  avgArray.allocate();
  avgArray.fill(0);
  ASSERT_EQ(avgArray.size(avgArray.getPartition()), nq * avgSizeTotal);

  // Create maps and stencils for gather_if
  std::vector<int> *maps[numDevices];
  std::vector<int> *stencils[numDevices];
  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    maps[srcDev] = new std::vector<int>(nq * avgSizeTotal, 0);
    stencils[srcDev] = new std::vector<int>(nq * avgSizeTotal, 0);
  }
  int avgArrayIdx = 0;
  for (int avgIdx = 0; avgIdx < avgVols.size(); avgIdx++) {
    VoxelVolume avg = avgVols.at(avgIdx);
    glm::ivec3 aDims = avg.getDims();
    glm::ivec3 aMin = avg.getMin();
    glm::ivec3 aMax = avg.getMax();

    for (int z = aMin.z; z < aMax.z; z++)
      for (int y = aMin.y; y < aMax.y; y++)
        for (int x = aMin.x; x < aMax.x; x++) {
          glm::ivec3 avgVox = glm::ivec3(x, y, z);

          for (int srcDev = 0; srcDev < numDevices; srcDev++) {
            const Partition partition = lattice.getDevicePartition(srcDev);
            const glm::ivec3 pMin = partition.getMin();
            const glm::ivec3 pMax = partition.getMax();
            const glm::ivec3 pDims = partition.getDims();
            const glm::ivec3 pArrDims = partition.getArrayDims();
            const glm::ivec3 pHalo = partition.getHalo();

            if ((pMin.x <= avgVox.x && avgVox.x < pMax.x) &&
                (pMin.y <= avgVox.y && avgVox.y < pMax.y) &&
                (pMin.z <= avgVox.z && avgVox.z < pMax.z)) {
              glm::ivec3 srcPos = avgVox - pMin + pHalo;
              for (int q = 0; q < nq; q++) {
                int srcIndex = I4D(q, srcPos.x, srcPos.y, srcPos.z, pArrDims.x,
                                   pArrDims.y, pArrDims.z);
                int mapIdx = q * avgSizeTotal + avgArrayIdx;
                maps[srcDev]->at(mapIdx) = srcIndex;
                stencils[srcDev]->at(mapIdx) = 1;
              }
              break;
            }
          }
          avgArrayIdx++;
        }
  }
  // Print maps and stencils
  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    std::vector<int> *map = maps[srcDev];
    std::vector<int> *sten = stencils[srcDev];
    std::ostringstream ss;
    ss << "Device " << srcDev << std::endl;
    ss << "Map ";
    std::copy(map->begin(), map->end() - 1,
              std::ostream_iterator<int>(ss, ","));
    ss << map->back() << std::endl;
    ss << "Stn ";
    std::copy(sten->begin(), sten->end() - 1,
              std::ostream_iterator<int>(ss, ","));
    ss << sten->back();
    std::cout << ss.str() << std::endl;
  }

#pragma omp parallel num_threads(numDevices)
  {
    // Run test kernel, average and check the array results
    std::stringstream ss;
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    const Partition partition = lattice.getDevicePartition(srcDev);
    const glm::ivec3 pDims = partition.getArrayDims();

    DistributionFunction *array =
        new DistributionFunction(nq, nx, ny, nz, numDevices);
    arrays[srcDev] = array;
    array->allocate(partition);
    array->fill(0);

    runTestKernel(array, partition, srcDev * pDims.x * pDims.y * pDims.z);

    thrust::device_vector<real> *d_values = array->getDeviceVector(partition);
    thrust::device_vector<int> d_map(maps[srcDev]->begin(),
                                     maps[srcDev]->end());
    thrust::device_vector<int> d_stencil(stencils[srcDev]->begin(),
                                         stencils[srcDev]->end());
    thrust::device_vector<real> *d_output =
        avgArray.getDeviceVector(avgArray.getPartition());

    thrust::gather_if(thrust::device, d_map.begin(), d_map.end(),
                      d_stencil.begin(), d_values->begin(), d_output->begin());
  }
  for (int i = 0; i < numDevices; i++) {
    arrays[i]->download();
    std::cout << "Device " << i << std::endl;
    std::cout << *arrays[i] << std::endl;
  }
  avgArray.download();
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

  DistributionArray<real> *arrays[numDevices];

  VoxelVolume area("testArea", vec3<int>(1, 1, 1), vec3<int>(3, 19, 3),
                   vec3<real>(0, 0, 0), vec3<real>(0, 0, 0));
  glm::ivec3 adims = area.getDims();
  DistributionArray<real> *areaArray =
      new DistributionArray<real>(nq, adims.x, adims.y, adims.z);
  areaArray->allocate(areaArray->getPartition(0, 0, 0));
  areaArray->fill(0);

#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    const Partition partition = lattice.getDevicePartition(srcDev);
    const Partition partitionNoHalo(partition.getMin(), partition.getMax(),
                                    glm::ivec3(0, 0, 0));
    const glm::ivec3 pDims = partitionNoHalo.getDims();

    DistributionArray<real> *array =
        new DistributionArray<real>(nq, nx, ny, nz, numDevices);
    arrays[srcDev] = array;
    array->allocate(partitionNoHalo);
    array->fill(0);

    runTestKernel(array, partitionNoHalo, srcDev * pDims.x * pDims.y * pDims.z);

    for (int q = 0; q < nq; q++)
      array->gather(area.getMin(), area.getMax(), q, q, partitionNoHalo,
                    areaArray, areaArray->getPartition(0, 0, 0));
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
  Partition fullLattice = fullArray->getPartition(0, 0, 0);
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
    Partition partitionNoHalo = df->getDevicePartition(srcDev);
    df->allocate(partitionNoHalo);
    df->fill(-srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->scatter(*fullArray, partitionNoHalo);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Create a second large empty array
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real> *newFullArray =
      new DistributionArray<real>(nq, nx, ny, nz);
  Partition newFullLattice = newFullArray->getPartition(0, 0, 0);
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
    Partition partitionNoHalo = df->getDevicePartition(srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->gather(partitionNoHalo, newFullArray);
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
  Partition fullLattice = fullArray->getPartition(0, 0, 0);
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
    Partition partitionNoHalo = df->getDevicePartition(srcDev);
    df->allocate(partitionNoHalo);
    df->fill(-srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->scatter(*fullArray, partitionNoHalo);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Create a second large empty array
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real> *newFullArray =
      new DistributionArray<real>(nq, nx, ny, nz);
  Partition newFullLattice = newFullArray->getPartition(0, 0, 0);
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
    Partition partitionNoHalo = df->getDevicePartition(srcDev);

    std::vector<bool> p2pList(numDevices);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->gatherSlice(slicePos, 0, 0, partitionNoHalo, newFullArray);
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
