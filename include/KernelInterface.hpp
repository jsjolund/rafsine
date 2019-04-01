#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_profiler_api.h>

#include <omp.h>

#include <limits.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Average.hpp"
#include "BoundaryCondition.hpp"
#include "ComputeKernelParams.hpp"
#include "DistributionFunction.hpp"
#include "Kernel.hpp"
#include "P2PLattice.hpp"

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
  std::unordered_map<VoxelArea, DistributionArray<real> *> m_avgs;

  void runInitKernel(DistributionFunction *df, DistributionFunction *dfT,
                     SubLattice subLattice, float rho, float vx, float vy,
                     float vz, float T);

  void runComputeKernelInterior(SubLattice subLattice, ComputeParams *kp,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t computeStream = 0);

  void runComputeKernelBoundary(D3Q4::Enum direction,
                                const SubLattice subLattice,
                                ComputeParams *params,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t stream = 0);

  std::vector<cudaStream_t> exchange(int srcDev, SubLattice subLattice,
                                     D3Q7::Enum direction);

 public:
  void getMinMax(real *min, real *max);
  void uploadBCs(std::vector<BoundaryCondition> *bcs);
  void resetDfs();
  void compute(DisplayQuantity::Enum displayQuantity,
               glm::ivec3 slicePos = glm::ivec3(-1, -1, -1));
  void plot(thrust::device_vector<real> *plot);

  Average getAverage(VoxelArea area, uint64_t deltaTicks);

  inline real *gpu_ptr() {
    return m_plot->gpu_ptr(m_plot->getSubLattice(0, 0, 0), m_bufferIndex);
  }
  inline void resetAverages() { m_resetAvg = true; }

  KernelInterface(const int nx, const int ny, const int nz,
                  const ComputeParams *params,
                  const std::vector<BoundaryCondition> *bcs,
                  const VoxelArray *voxels,
                  const std::vector<VoxelArea> *avgAreas, const int numDevices);
  ~KernelInterface() {
    for (ComputeParams *param : m_params) delete param;
    for (std::pair<VoxelArea, DistributionArray<real> *> element : m_avgs)
      delete element.second;
  }
};
