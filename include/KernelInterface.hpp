#pragma once

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>

#include <cuda_profiler_api.h>

#include <limits.h>
#include <omp.h>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Average.hpp"
#include "BoundaryCondition.hpp"
#include "ComputeKernelParams.hpp"
#include "DistributionFunction.hpp"
#include "InitKernel.hpp"
#include "Kernel.hpp"
#include "P2PLattice.hpp"
#include "SliceRenderKernel.hpp"

/**
 * @brief Class responsible for calling the CUDA kernel
 *
 */
class KernelInterface : public P2PLattice {
 private:
  //! Cuda LBM kernel parameters
  std::vector<ComputeParams *> m_params;
  DistributionArray<real> *m_plot;
  bool m_resetAvg;
  int m_bufferIndex;
  DistributionArray<real> *m_avgs;
  std::unordered_map<VoxelVolume, int> m_avgOffsets;

  void runInitKernel(DistributionFunction *df, DistributionFunction *dfT,
                     Partition partition, float rho, float vx, float vy,
                     float vz, float T);

  void runComputeKernelInterior(Partition partition, ComputeParams *kp,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t computeStream = 0);

  void runComputeKernelBoundary(D3Q4::Enum direction, const Partition partition,
                                ComputeParams *params,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t stream = 0);

  std::vector<cudaStream_t> exchange(int srcDev, Partition partition,
                                     D3Q7::Enum direction);

 public:
  DistributionFunction *getDf(int srcDev) { return m_params.at(srcDev)->df; }
  DistributionFunction *getDfT(int srcDev) { return m_params.at(srcDev)->dfT; }

  void getMinMax(real *min, real *max);
  void uploadBCs(std::shared_ptr<BoundaryConditions> bcs);
  void resetDfs();
  void compute(
      DisplayQuantity::Enum displayQuantity = DisplayQuantity::TEMPERATURE,
      glm::ivec3 slicePos = glm::ivec3(-1, -1, -1), real *sliceX = NULL,
      real *sliceY = NULL, real *sliceZ = NULL);
  void plot(thrust::device_vector<real> *plot);

  LatticeAverage getAverage(VoxelVolume area, uint64_t deltaTicks);

  inline void resetAverages() { m_resetAvg = true; }

  KernelInterface(const int nx, const int ny, const int nz,
                  const std::shared_ptr<ComputeParams> params,
                  const std::shared_ptr<BoundaryConditions> bcs,
                  const std::shared_ptr<VoxelArray> voxels,
                  const std::shared_ptr<VoxelVolumeArray> avgVols,
                  const int numDevices);

  ~KernelInterface() {
    delete m_plot;
    delete m_avgs;
    for (ComputeParams *param : m_params) delete param;
  }
};
