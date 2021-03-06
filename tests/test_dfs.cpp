#include "test_cuda.hpp"

#include <thrust/gather.h>

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"
#include "DistributionFunction.hpp"
#include "KernelInterface.hpp"
#include "test_kernel.hpp"

namespace cudatest {
class DistributionArrayTest : public CudaTest {};

// TEST_F(DistributionArrayTest, GatherTest2) {
//   // Create lattice
//   const int maxDevices = 9, nq = 2, nx = 4, ny = 20, nz = 4;
//   int nd;
//   CUDA_RT_CALL(cudaGetDeviceCount(&nd));
//   nd = min(nd, maxDevices);
//   CUDA_RT_CALL(cudaSetDevice(0));
//   CUDA_RT_CALL(cudaFree(0));

//   // Initialize lattice
//   P2PLattice lattice(nx, ny, nz, nd, D3Q4::Y_AXIS);
//   DistributionFunction* arrays[nd];

//   // Define some averaging areas
//   VoxelCuboidArray avgVols;
//   VoxelCuboid vol1("test1", Vector3<int>(1, 1, 1), Vector3<int>(2, 5, 2));
//   VoxelCuboid vol2("test2", Vector3<int>(nx - 1, ny - 1, nz - 1),
//                    Vector3<int>(nx, ny, nz));
//   avgVols.push_back(vol1);
//   avgVols.push_back(vol2);

//   // Combine averaging areas into one array
//   int avgSizeTotal = 0;
//   for (int avgIdx = 0; avgIdx < avgVols.size(); avgIdx++) {
//     Vector3<int> aExtents = avgVols.at(avgIdx).getExtents();
//     avgSizeTotal += aExtents.x() * aExtents.y() * aExtents.z();
//   }
//   DistributionArray<real_t> avgArray(nq, avgSizeTotal, 1, 1, 1, 0, D3Q4::Y_AXIS);
//   avgArray.allocate();
//   avgArray.fill(0);
//   ASSERT_EQ(avgArray.size(avgArray.getPartition()), nq * avgSizeTotal);

//   // Create maps and stencils for gather_if
//   std::vector<int>* maps[nd];
//   std::vector<int>* stencils[nd];
//   for (int srcDev = 0; srcDev < nd; srcDev++) {
//     maps[srcDev] = new std::vector<int>(nq * avgSizeTotal, 0);
//     stencils[srcDev] = new std::vector<int>(nq * avgSizeTotal, 0);
//   }
//   int avgArrayIdx = 0;
//   for (int avgIdx = 0; avgIdx < avgVols.size(); avgIdx++) {
//     VoxelCuboid avg = avgVols.at(avgIdx);
//     Vector3<int> aExtents = avg.getExtents();
//     Vector3<int> aMin = avg.getMin();
//     Vector3<int> aMax = avg.getMax();

//     for (int z = aMin.z(); z < aMax.z(); z++)
//       for (int y = aMin.y(); y < aMax.y(); y++)
//         for (int x = aMin.x(); x < aMax.x(); x++) {
//           Vector3<int> avgVox = Vector3<int>(x, y, z);

//           for (int srcDev = 0; srcDev < nd; srcDev++) {
//             const Partition partition = lattice.getDevicePartition(srcDev);
//             const Vector3<unsigned int> pMin = partition.getMin();
//             const Vector3<unsigned int> pMax = partition.getMax();
//             const Vector3<size_t> pExtents = partition.getExtents();
//             const Vector3<size_t> pArrExtents = partition.getArrayExtents();
//             const Vector3<int> pGhostLayer = partition.getGhostLayer();

//             if ((pMin.x() <= avgVox.x() && avgVox.x() < pMax.x()) &&
//                 (pMin.y() <= avgVox.y() && avgVox.y() < pMax.y()) &&
//                 (pMin.z() <= avgVox.z() && avgVox.z() < pMax.z())) {
//               Vector3<int> srcPos = avgVox - pMin + pGhostLayer;
//               for (int q = 0; q < nq; q++) {
//                 int srcIndex =
//                     I4D(q, srcPos.x(), srcPos.y(), srcPos.z(), pArrExtents.x(),
//                         pArrExtents.y(), pArrExtents.z());
//                 int mapIdx = q * avgSizeTotal + avgArrayIdx;
//                 maps[srcDev]->at(mapIdx) = srcIndex;
//                 stencils[srcDev]->at(mapIdx) = 1;
//               }
//               break;
//             }
//           }
//           avgArrayIdx++;
//         }
//   }
//   // Print maps and stencils
//   for (int srcDev = 0; srcDev < nd; srcDev++) {
//     std::vector<int>* map = maps[srcDev];
//     std::vector<int>* sten = stencils[srcDev];
//     std::ostringstream ss;
//     ss << "Device " << srcDev << std::endl;
//     ss << "Map ";
//     std::copy(map->begin(), map->end() - 1,
//               std::ostream_iterator<int>(ss, ","));
//     ss << map->back() << std::endl;
//     ss << "Stn ";
//     std::copy(sten->begin(), sten->end() - 1,
//               std::ostream_iterator<int>(ss, ","));
//     ss << sten->back();
//     std::cout << ss.str() << std::endl;
//   }

// #pragma omp parallel num_threads(nd)
//   {
//     // Run test kernel, average and check the array results
//     std::stringstream ss;
//     const int srcDev = omp_get_thread_num();
//     CUDA_RT_CALL(cudaSetDevice(srcDev));
//     CUDA_RT_CALL(cudaFree(0));

//     const Partition partition = lattice.getDevicePartition(srcDev);
//     const Vector3<int> pExtents = partition.getArrayExtents();

//     DistributionFunction* array =
//         new DistributionFunction(nq, nx, ny, nz, nd, D3Q4::Y_AXIS);
//     arrays[srcDev] = array;
//     array->allocate(partition);
//     array->fill(0);

//     runTestKernel(array, partition,
//                   srcDev * pExtents.x() * pExtents.y() * pExtents.z());

//     thrust::device_vector<real_t>* d_values = array->getDeviceVector(partition);
//     thrust::device_vector<int> d_map(maps[srcDev]->begin(),
//                                      maps[srcDev]->end());
//     thrust::device_vector<int> d_stencil(stencils[srcDev]->begin(),
//                                          stencils[srcDev]->end());
//     thrust::device_vector<real_t>* d_output =
//         avgArray.getDeviceVector(avgArray.getPartition());

//     thrust::gather_if(thrust::device, d_map.begin(), d_map.end(),
//                       d_stencil.begin(), d_values->begin(), d_output->begin());
//   }
//   for (int i = 0; i < nd; i++) {
//     arrays[i]->download();
//     std::cout << "Device " << i << std::endl;
//     std::cout << *arrays[i] << std::endl;
//   }
//   avgArray.download();
//   std::cout << avgArray << std::endl;
// }

TEST_F(DistributionArrayTest, GatherTest) {
  const int maxDevices = 9, nq = 1, nx = 4, ny = 20, nz = 4;
  int nd;
  CUDA_RT_CALL(cudaGetDeviceCount(&nd));
  nd = min(nd, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));
  P2PLattice lattice(nx, ny, nz, nd, D3Q4::Y_AXIS);

  DistributionArray<real_t>* arrays[nd];

  VoxelCuboid area("testArea", Vector3<int>(1, 1, 1),
                   Vector3<int>(3, 19, 3), Vector3<real_t>(0, 0, 0),
                   Vector3<real_t>(0, 0, 0));
  Vector3<int> aexts = area.getExtents();
  DistributionArray<real_t>* areaArray = new DistributionArray<real_t>(
      nq, aexts.x(), aexts.y(), aexts.z(), 1, 0, D3Q4::Y_AXIS);
  areaArray->allocate(areaArray->getPartition(0, 0, 0));
  areaArray->fill(0);

#pragma omp parallel num_threads(nd)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    const Partition partition = lattice.getDevicePartition(srcDev);
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), Vector3<int>(0, 0, 0));
    const Vector3<size_t> pExtents = partitionNoGhostLayer.getExtents();

    DistributionArray<real_t>* array =
        new DistributionArray<real_t>(nq, nx, ny, nz, nd, 0, D3Q4::Y_AXIS);
    arrays[srcDev] = array;
    array->allocate(partitionNoGhostLayer);
    array->fill(0);

    runTestKernel(array, partitionNoGhostLayer,
                  srcDev * pExtents.x() * pExtents.y() * pExtents.z());

    for (int q = 0; q < nq; q++)
      array->gather(area.getMin(), area.getMax(), q, q, partitionNoGhostLayer,
                    areaArray, areaArray->getPartition(0, 0, 0));
  }

  for (int i = 0; i < nd; i++) {
    arrays[i]->download();
    std::cout << "Device " << i << std::endl;
    std::cout << *arrays[i] << std::endl;
  }
  areaArray->download();
  std::cout << "Area" << std::endl;
  std::cout << *areaArray << std::endl;
}

TEST_F(DistributionArrayTest, ScatterGather) {
  int nd = 9;
  const int nq = 3, nx = 50, ny = 65, nz = 11;

  int maxDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&maxDevices));
  nd = min(nd, maxDevices);

  // Create a large array on GPU0 and fill it with some numbers
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real_t>* fullArray =
      new DistributionArray<real_t>(nq, nx, ny, nz, 1, 0, D3Q4::Y_AXIS);
  Partition fullLattice = fullArray->getPartition(0, 0, 0);
  fullArray->allocate(fullLattice);
  fullArray->fill(-10);
  runTestKernel(fullArray, fullLattice, 1);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Create as many DF groups as there are GPUs
  DistributionFunction* arrays[nd];

  // Scatter the large array to partitions
#pragma omp parallel num_threads(9)
#pragma omp for
  for (int srcDev = 0; srcDev < nd; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction* df =
        new DistributionFunction(nq, nx, ny, nz, nd, D3Q4::Y_AXIS);
    arrays[srcDev] = df;
    Partition partitionNoGhostLayer = df->getDevicePartition(srcDev);
    df->allocate(partitionNoGhostLayer);
    df->fill(-srcDev);

    std::vector<bool> p2pList(nd);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->scatter(*fullArray, partitionNoGhostLayer);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Create a second large empty array
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real_t>* newFullArray =
      new DistributionArray<real_t>(nq, nx, ny, nz, 1, 0, D3Q4::Y_AXIS);
  Partition newFullLattice = newFullArray->getPartition(0, 0, 0);
  newFullArray->allocate(newFullLattice);
  newFullArray->fill(-20);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Gather the partitions into the new large array
#pragma omp parallel num_threads(9)
#pragma omp for
  for (int srcDev = 0; srcDev < nd; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction* df = arrays[srcDev];
    Partition partitionNoGhostLayer = df->getDevicePartition(srcDev);

    std::vector<bool> p2pList(nd);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->gather(partitionNoGhostLayer, newFullArray);
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
          real_t a = (*fullArray)(fullLattice, q, x, y, z);
          real_t b = (*newFullArray)(newFullLattice, q, x, y, z);
          if (a != b)
            FAIL() << "Distribution function not equal at q=" << q
                   << ", x=" << x << ", y=" << y << ", z=" << z
                   << ", orig=" << a << ", new=" << b << std::endl;
        }

  delete fullArray;
  delete newFullArray;
  for (int i = 0; i < nd; i++) delete arrays[i];
}

TEST_F(DistributionArrayTest, ScatterGatherSlice) {
  int nd = 2;
  const int nq = 1, nx = 6, ny = 5, nz = 4;
  Vector3<int> slicePos(2, 2, 2);

  int maxDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&maxDevices));
  nd = min(nd, maxDevices);

  // Create a large array on GPU0 and fill it with some numbers
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real_t>* fullArray =
      new DistributionArray<real_t>(nq, nx, ny, nz, 1, 0, D3Q4::Y_AXIS);
  Partition fullLattice = fullArray->getPartition(0, 0, 0);
  fullArray->allocate(fullLattice);
  fullArray->fill(0);
  runTestKernel(fullArray, fullLattice, 1);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Create as many DF groups as there are GPUs
  DistributionFunction* arrays[nd];

  // Scatter the large array to partitions
#pragma omp parallel num_threads(9)
#pragma omp for
  for (int srcDev = 0; srcDev < nd; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction* df =
        new DistributionFunction(nq, nx, ny, nz, nd, D3Q4::Y_AXIS);
    arrays[srcDev] = df;
    Partition partitionNoGhostLayer = df->getDevicePartition(srcDev);
    df->allocate(partitionNoGhostLayer);
    df->fill(-srcDev);

    std::vector<bool> p2pList(nd);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->scatter(*fullArray, partitionNoGhostLayer);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Create a second large empty array
  CUDA_RT_CALL(cudaSetDevice(0));
  DistributionArray<real_t>* newFullArray =
      new DistributionArray<real_t>(nq, nx, ny, nz, 1, 0, D3Q4::Y_AXIS);
  Partition newFullLattice = newFullArray->getPartition(0, 0, 0);
  newFullArray->allocate(newFullLattice);
  newFullArray->fill(0);
  CUDA_RT_CALL(cudaDeviceSynchronize());

  // Gather the partitions into the new large array
#pragma omp parallel num_threads(9)
#pragma omp for
  for (int srcDev = 0; srcDev < nd; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    // Allocate a sub lattice on GPUx
    DistributionFunction* df = arrays[srcDev];
    Partition partitionNoGhostLayer = df->getDevicePartition(srcDev);

    std::vector<bool> p2pList(nd);
    enablePeerAccess(srcDev, 0, &p2pList);
    df->gatherSlice(slicePos, 0, 0, partitionNoGhostLayer, newFullArray);
    disableAllPeerAccess(srcDev, &p2pList);

    CUDA_RT_CALL(cudaDeviceSynchronize());
  }

  // Compare the new and old large array
  CUDA_RT_CALL(cudaSetDevice(0));
  fullArray->download();
  newFullArray->download();

  std::cout << *newFullArray << std::endl;

  for (int q = 0; q < nq; q++) {
    const int x = slicePos.x();
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        real_t a = (*fullArray)(fullLattice, q, x, y, z);
        real_t b = (*newFullArray)(newFullLattice, q, x, y, z);
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
      const int y = slicePos.y();
      for (int z = 0; z < nz; z++) {
        real_t a = (*fullArray)(fullLattice, q, x, y, z);
        real_t b = (*newFullArray)(newFullLattice, q, x, y, z);
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
        const int z = slicePos.z();
        real_t a = (*fullArray)(fullLattice, q, x, y, z);
        real_t b = (*newFullArray)(newFullLattice, q, x, y, z);
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
          if (x == slicePos.x() || y == slicePos.y() || z == slicePos.z())
            continue;
          real_t a = (*newFullArray)(newFullLattice, q, x, y, z);
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
  for (int i = 0; i < nd; i++) delete arrays[i];
}

}  // namespace cudatest
