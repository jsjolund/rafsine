#pragma once

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <limits.h>
#include <omp.h>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Average.hpp"
#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "GatherKernel.hpp"
#include "InitKernel.hpp"
#include "Kernel.hpp"
#include "LatticeHistogram.hpp"
#include "P2PLattice.hpp"
#include "SimulationParams.hpp"
#include "SimulationState.hpp"
#include "SliceRenderKernel.hpp"

/**
 * @brief Class responsible for calling the CUDA kernel
 *
 */
class KernelInterface : public P2PLattice {
 private:
  //! Cuda LBM kernel parameters
  std::vector<SimulationParams*> m_params;
  //! Cuda LBM distribution function states
  std::vector<SimulationState*> m_state;
  //! Array for storing visualization plot
  DistributionArray<real>* m_plot;
  //! Plot array back buffer
  DistributionArray<real>* m_plot_tmp;
  //! Stores averages gathered from multiple GPUs
  DistributionArray<real>* m_avgs;
  //! Index offsets for different volumes in averaging array
  std::unordered_map<VoxelVolume, int> m_avgOffsets;
  //! When true, averaging array will be reset on next compute
  bool m_resetAvg;
  //! Length of time step in seconds
  real m_dt;
  //! LBM method
  LBM::Enum m_method;

  void runInitKernel(DistributionFunction* df,
                     DistributionFunction* dfT,
                     Partition partition,
                     float rho,
                     float vx,
                     float vy,
                     float vz,
                     float T);

  void runComputeKernelInterior(Partition partition,
                                SimulationParams* kp,
                                SimulationState* state,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t computeStream = 0);

  void runComputeKernelBoundary(D3Q4::Enum direction,
                                const Partition partition,
                                SimulationParams* params,
                                SimulationState* state,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t stream = 0);

  std::vector<cudaStream_t> exchange(int srcDev,
                                     Partition partition,
                                     D3Q7::Enum direction);

 public:
  DistributionFunction* getDf(int srcDev) { return m_state.at(srcDev)->df; }

  DistributionFunction* getDfT(int srcDev) { return m_state.at(srcDev)->dfT; }

  void uploadBCs(std::shared_ptr<BoundaryConditions> bcs);

  void resetDfs();

  void compute(
      DisplayQuantity::Enum displayQuantity = DisplayQuantity::TEMPERATURE,
      Eigen::Vector3i slicePos = Eigen::Vector3i(-1, -1, -1),
      real* sliceX = NULL,
      real* sliceY = NULL,
      real* sliceZ = NULL,
      bool runSimulation = true);

  LatticeAverage getAverage(VoxelVolume area, uint64_t deltaTicks);

  inline void resetAverages() { m_resetAvg = true; }

  void calculateAverages();

  void getMinMax(real* min, real* max, thrust::host_vector<real>* histogram);

  KernelInterface(const int nx,
                  const int ny,
                  const int nz,
                  const real dt,
                  const std::shared_ptr<SimulationParams> params,
                  const std::shared_ptr<BoundaryConditions> bcs,
                  const std::shared_ptr<VoxelArray> voxels,
                  const std::shared_ptr<VoxelVolumeArray> avgVols,
                  const int nd,
                  const LBM::Enum method,
                  const D3Q4::Enum partitioning);

  ~KernelInterface() {
    delete m_plot;
    delete m_plot_tmp;
    delete m_avgs;
    for (SimulationParams* param : m_params) delete param;
    for (SimulationState* state : m_state) delete state;
  }
};
