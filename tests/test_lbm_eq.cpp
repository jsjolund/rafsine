#include "test_cuda.hpp"

#include <QObject>
#include <QString>
#include <QThread>

#include <stdio.h>
#include <iostream>

#include "DomainData.hpp"
#include "LbmFile.hpp"
#include "SimulationWorker.hpp"

namespace cudatest {
class LbmTest : public CudaTest {};

TEST_F(LbmTest, SingleMultiEq) {
  const int iterations = 1000;
  const QString lbmFilePath =
      QObject::tr("/home/ubuntu/rafsine/problems/data_center/data_center.lbm");
  const LbmFile lbmFile(lbmFilePath);

  // Run on single GPU
  int numDevices = 1;
  CUDA_RT_CALL(cudaSetDevice(0));
  SimulationWorker *simWorker =
      new SimulationWorker(lbmFile, iterations, numDevices);
  std::shared_ptr<VoxelGeometry> voxGeo = simWorker->getVoxelGeometry();
  simWorker->run();
  int nx = voxGeo->getNx(), ny = voxGeo->getNy(), nz = voxGeo->getNz();
  DistributionFunction *df0 = simWorker->getDomainData()->m_kernel->getDf(0);
  DistributionFunction *singleGpuDf = new DistributionFunction(19, nx, ny, nz);
  const Partition fullLattice = df0->getPartition();
  singleGpuDf->allocate(fullLattice);
  df0->gather(fullLattice, singleGpuDf);
  delete simWorker;
  singleGpuDf->download();

  // Run on multiple GPUs
  numDevices = 9;
  DistributionFunction *multiGpuDf = new DistributionFunction(19, nx, ny, nz);
  multiGpuDf->allocate(fullLattice);
  simWorker = new SimulationWorker(lbmFile, iterations, numDevices);
  simWorker->run();
#pragma omp parallel num_threads(numDevices)
  {
    const int srcDev = omp_get_thread_num() % numDevices;
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    DistributionFunction *srcDf =
        simWorker->getDomainData()->m_kernel->getDf(srcDev);
    const Partition partition = srcDf->getAllocatedPartitions().at(0);
    srcDf->gather(partition, multiGpuDf);
  }
  multiGpuDf->download();

  // Compare
  int numOk[19];
  for (int q = 0; q < 19; q++) numOk[q] = 0;
#pragma omp parallel num_threads(19)
  {
    int q = omp_get_thread_num();
    for (int x = 0; x < nx; x++)
      for (int y = 0; y < ny; y++)
        for (int z = 0; z < nz; z++) {
          real a = singleGpuDf->read(fullLattice, q, x, y, z);
          real b = multiGpuDf->read(fullLattice, q, x, y, z);
          if (a == b) {
            numOk[q] += 1;
          } else {
            std::cerr << "Fail at (" << q << ", " << x << ", " << y << ", " << z
                      << ")" << std::endl;
          }
        }
  }
  int sumOk = 0;
  for (int q = 0; q < 19; q++) sumOk += numOk[q];
  std::cout << "Read " << sumOk << " values out of " << 19 * nx * ny * nz
            << std::endl;
  ASSERT_EQ(sumOk, 19 * nx * ny * nz);
}

}  // namespace cudatest
