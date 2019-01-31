#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "DistributionFunction.hpp"
#include "KernelInterface.hpp"
#include "test_kernel.hpp"

// // Reference for empty DF group
// static real pEmpty[4][4][4] = {{                 //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0}},   //
//                                {                 //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0}},   //
//                                {                 //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0}},   //
//                                {                 //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0},    //
//                                 {0, 0, 0, 0}}};  //
// // Reference for initial values in lattice, before halo exchange
// static real pBefore[4][4][4] = {{                 //
//                                  {0, 0, 0, 0},    //
//                                  {0, 0, 0, 0},    //
//                                  {0, 0, 0, 0},    //
//                                  {0, 0, 0, 0}},   //
//                                 {                 //
//                                  {0, 0, 0, 0},    //
//                                  {0, 1, 2, 0},    //
//                                  {0, 3, 4, 0},    //
//                                  {0, 0, 0, 0}},   //
//                                 {                 //
//                                  {0, 0, 0, 0},    //
//                                  {0, 5, 6, 0},    //
//                                  {0, 7, 8, 0},    //
//                                  {0, 0, 0, 0}},   //
//                                 {                 //
//                                  {0, 0, 0, 0},    //
//                                  {0, 0, 0, 0},    //
//                                  {0, 0, 0, 0},    //
//                                  {0, 0, 0, 0}}};  //
// // Reference for values after halo exchange
// static real pAfter[4][4][4] = {{                 //
//                                 {8, 7, 8, 7},    //
//                                 {6, 5, 6, 5},    //
//                                 {8, 7, 8, 7},    //
//                                 {6, 5, 6, 5}},   //
//                                {                 //
//                                 {4, 3, 4, 3},    //
//                                 {2, 1, 2, 1},    //
//                                 {4, 3, 4, 3},    //
//                                 {2, 1, 2, 1}},   //
//                                {                 //
//                                 {8, 7, 8, 7},    //
//                                 {6, 5, 6, 5},    //
//                                 {8, 7, 8, 7},    //
//                                 {6, 5, 6, 5}},   //
//                                {                 //
//                                 {4, 3, 4, 3},    //
//                                 {2, 1, 2, 1},    //
//                                 {4, 3, 4, 3},    //
//                                 {2, 1, 2, 1}}};  //

/**
 * @brief Compare a subLattice with a reference array
 */
template <size_t nx, size_t ny, size_t nz>
static int compareSubLattices(DistributionFunction *df, SubLattice p0,
                              real (&ref)[nx][ny][nz]) {
  size_t errors = 0;
  glm::ivec3 min = p0.getLatticeMin() - glm::ivec3(1, 1, 1);
  glm::ivec3 max = p0.getLatticeMax() + glm::ivec3(1, 1, 1);
  for (int hq = 0; hq < df->getQ(); hq++, hq++)
    for (int hz = min.z, rz = 0; hz < max.z; hz++, rz++)
      for (int hy = min.y, ry = 0; hy < max.y; hy++, ry++)
        for (int hx = min.x, rx = 0; hx < max.x; hx++, rx++) {
          real a = ref[rz][ry][rx];
          real b = (*df)(p0, hq, hx, hy, hz);
          // EXPECT_EQ(a, b);
          if (a != b) errors++;
        }
  return errors;
}

// // TEST(DistributedDFTest, HaloExchangeCPU) {
// //   int nq = 7, nx = 2, ny = 2, nz = 4, divisions = 2;
// //   DistributionFunction *df =
// //       new DistributionFunction(nq, nx, ny, nz, divisions);

// //   std::vector<SubLattice> subLattices = df->getSubLattices();
// //   for (SubLattice p : subLattices) {
// //     df->allocate(p);
// //   }
// //   df->fill(0, 0);

// //   int i = 0;
// //   for (int q = 0; q < nq; ++q)
// //     for (int z = 0; z < nz; ++z)
// //       for (int y = 0; y < ny; ++y)
// //         for (int x = 0; x < nx; ++x) {
// //           (*df)(q, x, y, z) = 1 + (i++ % 8);
// //         }
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(0), pBefore));
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(1), pBefore));
// //   for (std::pair<SubLattice, std::unordered_map<SubLattice,
// HaloParamsLocal
// *>>
// //            element1 : df->m_haloData) {
// //     SubLattice subLattice = element1.first;
// //     std::unordered_map<SubLattice, HaloParamsLocal *> neighboursMap =
// //         element1.second;

// //     for (std::pair<SubLattice, HaloParamsLocal *> element2 :
// neighboursMap)
// {
// //       SubLattice neighbour = element2.first;

// //       std::vector<glm::ivec3> pSrc, nSrc, pDst, nDst;
// //       for (int i = 0; i < 27; i++) {
// //         glm::ivec3 direction = D3Q27[i];

// //         subLattice.getHalo(direction, &pSrc, &nDst);
// //         neighbour.getHalo(-direction, &nSrc, &pDst);
// //         ASSERT_EQ(pSrc.size(), nDst.size());
// //         ASSERT_EQ(pSrc.size(), nSrc.size());
// //         ASSERT_EQ(pSrc.size(), pDst.size());

// //         for (int j = 0; j < pSrc.size(); j++) {
// //           glm::ivec3 src = pSrc.at(j);
// //           glm::ivec3 dst = pDst.at(j);

// //           for (int q = 0; q < nq; ++q) {
// //             (*df)(neighbour, q, dst.x, dst.y, dst.z) =
// //                 (*df)(subLattice, q, src.x, src.y, src.z);
// //           }
// //         }
// //       }
// //     }
// //   }
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(0), pAfter));
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(1), pAfter));
// // }

// // TEST(DistributedDFTest, SingleGPUKernelSubLattice) {
// //   const int nq = 27, nx = 2, ny = 2, nz = 4, divisions = 0;
// //   CUDA_RT_CALL(cudaSetDevice(0));
// //   DistributionFunction *df =
// //       new DistributionFunction(nq, nx, ny, nz, divisions);
// //   for (SubLattice subLattice : df->getSubLattices())
// df->allocate(subLattice);
// //   for (int q = 0; q < nq; q++) df->fill(q, 0);
// //   df->upload();
// //   cudaStream_t computeStream;
// //   CUDA_RT_CALL(cudaStreamCreate(&computeStream));
// //   std::vector<SubLattice> subLattices = df->getSubLattices();
// //   for (SubLattice subLattice : subLattices) {
// //     runTestKernel(df, subLattice, computeStream);
// //   }
// //   df->download();
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(0), pBefore));
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(1), pBefore));
// //   CUDA_RT_CALL(cudaStreamDestroy(computeStream));
// //   CUDA_RT_CALL(cudaDeviceReset());
// // }

// // TEST(DistributedDFTest, SingleGPUKernelSwapAndEquals) {
// //   const int nq = 1, nx = 2, ny = 2, nz = 4, divisions = 2;
// //   CUDA_RT_CALL(cudaSetDevice(0));
// //   DistributionFunction *df, *dfTmp;
// //   df = new DistributionFunction(nq, nx, ny, nz, divisions);
// //   dfTmp = new DistributionFunction(nq, nx, ny, nz, divisions);
// //   std::vector<SubLattice> subLattices = df->getSubLattices();
// //   for (SubLattice subLattice : subLattices) {
// //     df->allocate(subLattice);
// //     dfTmp->allocate(subLattice);
// //   }
// //   for (int q = 0; q < nq; q++) {
// //     df->fill(q, 0);
// //     dfTmp->fill(q, 0);
// //   }
// //   df->upload();
// //   dfTmp->upload();
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(0), pEmpty));
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(1), pEmpty));
// //   ASSERT_TRUE(compareSubLattices(dfTmp, subLattices.at(0), pEmpty));
// //   ASSERT_TRUE(compareSubLattices(dfTmp, subLattices.at(1), pEmpty));
// //   cudaStream_t computeStream;
// //   CUDA_RT_CALL(cudaStreamCreate(&computeStream));
// //   for (SubLattice subLattice : subLattices) {
// //     runTestKernel(df, subLattice, computeStream);
// //   }
// //   DistributionFunction::swap(df, dfTmp);
// //   df->download();
// //   dfTmp->download();
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(0), pEmpty));
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(1), pEmpty));
// //   ASSERT_TRUE(compareSubLattices(dfTmp, subLattices.at(0), pBefore));
// //   ASSERT_TRUE(compareSubLattices(dfTmp, subLattices.at(1), pBefore));
// //   df = dfTmp;
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(0), pBefore));
// //   ASSERT_TRUE(compareSubLattices(df, subLattices.at(1), pBefore));
// //   ASSERT_TRUE(compareSubLattices(dfTmp, subLattices.at(0), pBefore));
// //   ASSERT_TRUE(compareSubLattices(dfTmp, subLattices.at(1), pBefore));
// //   CUDA_RT_CALL(cudaStreamDestroy(computeStream));
// //   CUDA_RT_CALL(cudaDeviceReset());
// // }

TEST(DistributedDFTest, HaloExchangeMultiGPU) {
  // int maxDevices = 2, nq = 19, nx = 3, ny = 5, nz = 2;
  int maxDevices = 2, nq = 19, nx = 5, ny = 3, nz = 2;

  int numDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));

  // Create as many DF groups as there are GPUs
  DistributionFunction *dfs[numDevices];

  // Distribute the workload
  std::unordered_map<SubLattice, int> subLatticeDeviceMap;
  std::vector<SubLattice> deviceSubLatticeMap(numDevices);

  // Calculate sub lattices and assign them to GPUs
  {
    DistributionFunction df(1, nx, ny, nz, numDevices);
    std::vector<SubLattice> subLattices = df.getSubLattices();
    for (int i = 0; i < subLattices.size(); i++) {
      SubLattice subLattice = subLattices.at(i);
      int devIndex = i % numDevices;
      subLatticeDeviceMap[subLattice] = devIndex;
      deviceSubLatticeMap.at(devIndex) = subLattice;
    }
  }
  // bool success = true;

  // Create one CPU thread per GPU
#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaFree(0));

    DistributionFunction *df =
        new DistributionFunction(nq, nx, ny, nz, numDevices);
    SubLattice subLattice = deviceSubLatticeMap.at(srcDev);
    df->allocate(subLattice);
    dfs[srcDev] = df;
    for (int q = 0; q < nq; q++) df->fill(q, q + srcDev * 10);
    df->upload();
    CUDA_RT_CALL(cudaDeviceSynchronize());
    // Wait for all threads to create distribution functions...

#pragma omp barrier
    // Enable P2P access between GPUs
    std::vector<bool> peerAccessList(numDevices);
    for (int nIdx = 0; nIdx < df->getQ(); nIdx++) {
      SubLattice neighbour = df->getNeighbour(subLattice, D3Q27[nIdx]);
      const int dstDev = subLatticeDeviceMap[neighbour];
      enablePeerAccess(srcDev, dstDev, &peerAccessList);
    }

    // Setup streams
    cudaStream_t computeStream;
    cudaStream_t dfExchangeStream;
    CUDA_RT_CALL(
        cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_RT_CALL(
        cudaStreamCreateWithFlags(&dfExchangeStream, cudaStreamNonBlocking));

    runTestKernel(df, subLattice, srcDev, computeStream);
    CUDA_RT_CALL(cudaStreamSynchronize(computeStream));
    CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
    {
      SubLattice neighbour = df->getNeighbour(subLattice, D3Q27[1]);
      DistributionFunction *ndf = dfs[subLatticeDeviceMap[neighbour]];
      SubLatticeSegment segment =
          df->getSubLatticeSegment(subLattice, neighbour, D3Q7::X_AXIS_POS);
      for (int q = 0; q < df->getQ(); q++) {
        real *dfPtr = df->gpu_ptr(subLattice, q, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z);
        real *ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfExchangeStream));
      }
    }
    {
      SubLattice neighbour = df->getNeighbour(subLattice, D3Q27[2]);
      DistributionFunction *ndf = dfs[subLatticeDeviceMap[neighbour]];
      SubLatticeSegment segment =
          df->getSubLatticeSegment(subLattice, neighbour, D3Q7::X_AXIS_NEG);
      for (int q = 0; q < df->getQ(); q++) {
        real *dfPtr = df->gpu_ptr(subLattice, q, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z);
        real *ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfExchangeStream));
      }
    }
    CUDA_RT_CALL(cudaStreamSynchronize(dfExchangeStream));
#pragma omp barrier
    {
      SubLattice neighbour = df->getNeighbour(subLattice, D3Q27[3]);
      DistributionFunction *ndf = dfs[subLatticeDeviceMap[neighbour]];
      SubLatticeSegment segment =
          df->getSubLatticeSegment(subLattice, neighbour, D3Q7::Y_AXIS_POS);
      for (int q = 0; q < df->getQ(); q++) {
        real *dfPtr = df->gpu_ptr(subLattice, q, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z);
        real *ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfExchangeStream));
      }
    }
    {
      SubLattice neighbour = df->getNeighbour(subLattice, D3Q27[4]);
      DistributionFunction *ndf = dfs[subLatticeDeviceMap[neighbour]];
      SubLatticeSegment segment =
          df->getSubLatticeSegment(subLattice, neighbour, D3Q7::Y_AXIS_NEG);
      for (int q = 0; q < df->getQ(); q++) {
        real *dfPtr = df->gpu_ptr(subLattice, q, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z);
        real *ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfExchangeStream));
      }
    }
    CUDA_RT_CALL(cudaStreamSynchronize(dfExchangeStream));
#pragma omp barrier
    {
      SubLattice neighbour = df->getNeighbour(subLattice, D3Q27[5]);
      DistributionFunction *ndf = dfs[subLatticeDeviceMap[neighbour]];
      SubLatticeSegment segment =
          df->getSubLatticeSegment(subLattice, neighbour, D3Q7::Z_AXIS_POS);
      for (int q = 0; q < df->getQ(); q++) {
        real *dfPtr = df->gpu_ptr(subLattice, q, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z);
        real *ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfExchangeStream));
      }
    }
    {
      SubLattice neighbour = df->getNeighbour(subLattice, D3Q27[6]);
      DistributionFunction *ndf = dfs[subLatticeDeviceMap[neighbour]];
      SubLatticeSegment segment =
          df->getSubLatticeSegment(subLattice, neighbour, D3Q7::Z_AXIS_NEG);
      for (int q = 0; q < df->getQ(); q++) {
        real *dfPtr = df->gpu_ptr(subLattice, q, segment.m_src.x,
                                  segment.m_src.y, segment.m_src.z);
        real *ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x,
                                    segment.m_dst.y, segment.m_dst.z);
        CUDA_RT_CALL(cudaMemcpy2DAsync(
            ndfPtr, segment.m_dstStride, dfPtr, segment.m_srcStride,
            segment.m_segmentLength, segment.m_numSegments, cudaMemcpyDefault,
            dfExchangeStream));
      }
    }
    CUDA_RT_CALL(cudaStreamSynchronize(dfExchangeStream));

#pragma omp barrier

    CUDA_RT_CALL(cudaDeviceSynchronize());

    // for (SubLattice subLattice : deviceSubLatticeMap.at(srcDev)) {
    //   // Merge subLattice array into device 0
    //   if (srcDev != 0) {
    //     DistributionFunction *dstDf = dfs[0];
    //     cudaStream_t cpyStream = cpyStreams[0];
    //     df->pushSubLattice(srcDev, subLattice, 0, dstDf, cpyStream);
    //   }
    // }
  }

  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    DistributionFunction *df = dfs[srcDev];
    df->download();

    std::cout << "######################## Device " << srcDev << std::endl;
    std::cout << *df << std::endl;
  }

  // for (int srcDev = 0; srcDev < numDevices; srcDev++) {
  //   CUDA_RT_CALL(cudaSetDevice(srcDev));
  //   CUDA_RT_CALL(cudaDeviceSynchronize());

  //   DistributionFunction *df = dfs[srcDev];
  //   df->download();

  //   // Check after halo exchange
  //   for (SubLattice subLattice : df->getAllocatedSubLattices()) {
  //     std::stringstream ss;
  //     ss << "Checking subLattice " << subLattice << " on GPU" << srcDev
  //        << std::endl;
  //     std::cout << ss.str();
  //     ss.str("");
  //     int errors = compareSubLattices(df, subLattice, pAfter);
  //     ss << "Device " << srcDev << " failed with " << errors << std::endl;
  //     EXPECT_EQ(errors, 0) << ss.str();
  //   }
  //   delete df;
  //   CUDA_RT_CALL(cudaDeviceReset());
  // }
  // ASSERT_TRUE(success);
}
