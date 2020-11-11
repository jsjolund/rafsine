#pragma once

#include <cuda_profiler_api.h>
#include <limits.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

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
#include "LBM_BGK.hpp"
#include "LatticeHistogram.hpp"
#include "P2PLattice.hpp"
#include "SimulationParams.hpp"
#include "SimulationState.hpp"
#include "SliceRenderKernel.hpp"
#include "Vector3.hpp"

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
  DistributionArray<real_t>* m_plot;
  //! Plot array back buffer
  DistributionArray<real_t>* m_plot_tmp;
  //! Stores averages gathered from multiple GPUs
  DistributionArray<real_t>* m_avgs;
  //! Index offsets for different volumes in averaging array
  std::unordered_map<VoxelCuboid, int> m_avgOffsets;
  //! When true, averaging array will be reset on next compute
  bool m_resetAvg;
  //! Length of time step in seconds
  real_t m_dt;
  //! LBM method
  LBM::Enum m_method;

  /**
   * @brief Initialize the distribution functions
   *
   * @param df Velocity distribution function
   * @param dfT Temperature distribution function
   * @param partition Partition on lattice
   * @param rho Initial density
   * @param vx Initial velocity X-axis
   * @param vy Initial velocity Y-axis
   * @param vz Initial velocity Z-axis
   * @param T Initial temperature
   */
  void runInitKernel(DistributionFunction* df,
                     DistributionFunction* dfT,
                     Partition partition,
                     float rho,
                     float vx,
                     float vy,
                     float vz,
                     float T);

  /**
   * @brief Compute stream and collide for interior lattice sites
   *
   * @param partition
   * @param params
   * @param state
   * @param displayQuantity
   * @param computeStream
   */
  void runComputeKernelInterior(Partition partition,
                                SimulationParams* params,
                                SimulationState* state,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t computeStream = 0);

  /**
   * @brief Compute stream and collide for boundary lattice sites (adjacent to
   * ghost layers)
   *
   * @param direction
   * @param partition
   * @param params
   * @param state
   * @param displayQuantity
   * @param stream
   */
  void runComputeKernelBoundary(D3Q4::Enum direction,
                                const Partition partition,
                                SimulationParams* params,
                                SimulationState* state,
                                DisplayQuantity::Enum displayQuantity,
                                cudaStream_t stream = 0);

  /**
   * @brief Exchange ghost layers between distribution functions
   *
   * @param srcDev
   * @param partition
   * @param direction
   * @return std::vector<cudaStream_t>
   */
  std::vector<cudaStream_t> exchange(unsigned int srcDev,
                                     Partition partition,
                                     D3Q7::Enum direction);

 public:
  /**
   * @brief Upload boundary conditions from host to devices
   *
   * @param bcs
   */
  void uploadBCs(std::shared_ptr<BoundaryConditions> bcs);

  /**
   * @brief Reset distribution functions to initial conditions
   *
   */
  void resetDfs();

  /**
   * @brief Compute one stream and collide time steps using all GPUs
   *
   * @param displayQuantity The macroscopic quantity to plot
   * @param slicePos Current display slice positions on X, Y, Z-axis
   * @param sliceX Pointer to display slice memory on X-axis
   * @param sliceY Pointer to display slice memory on Y-axis
   * @param sliceZ Pointer to display slice memory on Z-axis
   * @param runSimulation If false, only render display slices without
   * simulating
   */
  void compute(
      DisplayQuantity::Enum displayQuantity = DisplayQuantity::TEMPERATURE,
      Vector3<int> slicePos = Vector3<int>(-1, -1, -1),
      real_t* sliceX = NULL,
      real_t* sliceY = NULL,
      real_t* sliceZ = NULL,
      bool runSimulation = true);

  /**
   * @brief Get the time averaged macroscopic values for an area
   *
   * @param area
   * @param deltaTicks
   * @return LatticeAverage
   */
  LatticeAverage getAverage(VoxelCuboid area, uint64_t deltaTicks);

  /**
   * @brief Reset time averaged macroscopic values on next time step
   *
   */
  inline void resetAverages() { m_resetAvg = true; }

  /**
   * @brief Gathers the averages to make them available for
   * KernelInterface::getAverage
   *
   */
  void calculateAverages();

  /**
   * @brief Get the minimum and maximum of the currently plotted macroscopic
   * value
   *
   * @param min
   * @param max
   * @param histogram
   */
  void getMinMax(real_t* min,
                 real_t* max,
                 thrust::host_vector<real_t>* histogram);

  /**
   * @brief Construct a new Kernel Interface object
   *
   * @param nx Lattice size on X-axis
   * @param ny Lattice size on Y-axis
   * @param nz Lattice size on Z-axis
   * @param dt Time step in seconds
   * @param params Required simulation physical parameters
   * @param bcs Boundary conditions
   * @param voxels Voxel array
   * @param avgVols Volumes to perform time averaged mesurements over
   * @param nd Number of CUDA devices
   * @param method LBM algorithm
   * @param partitioning Lattice partitioning
   */
  KernelInterface(const size_t nx,
                  const size_t ny,
                  const size_t nz,
                  const real_t dt,
                  const std::shared_ptr<SimulationParams> params,
                  const std::shared_ptr<BoundaryConditions> bcs,
                  const std::shared_ptr<VoxelArray> voxels,
                  const std::shared_ptr<VoxelCuboidArray> avgVols,
                  const size_t nd,
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
