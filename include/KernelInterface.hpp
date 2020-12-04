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
 protected:
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
  virtual void resetDfs() = 0;

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
  virtual void compute(
      DisplayQuantity::Enum displayQuantity = DisplayQuantity::TEMPERATURE,
      Vector3<int> slicePos = Vector3<int>(-1, -1, -1), real_t* sliceX = NULL,
      real_t* sliceY = NULL, real_t* sliceZ = NULL,
      bool runSimulation = true) = 0;

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
  void getMinMax(real_t* min, real_t* max,
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
  KernelInterface(const size_t nx, const size_t ny, const size_t nz,
                  const size_t nd,  const real_t dt, const D3Q4::Enum partitioning)
      : P2PLattice(nx, ny, nz, nd, partitioning),
        m_params(nd),
        m_state(nd),
        m_resetAvg(false),
        m_dt(dt) {}

  ~KernelInterface() {
    delete m_plot;
    delete m_plot_tmp;
    delete m_avgs;
    for (SimulationParams* param : m_params) delete param;
    for (SimulationState* state : m_state) delete state;
  }
};
