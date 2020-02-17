#pragma once

#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "VoxelArray.hpp"

class SimulationState {
 public:
  /**
   * Contains the macroscopic temperature, velocity (x,y,z components)
   * integrated in time, so divide by number of time steps to get average).
   * 0 -> temperature
   * 1 -> x-component of velocity
   * 2 -> y-component of velocity
   * 3 -> z-component of velocity
   */
  DistributionArray<real>* avg;
  //! Average array (for swap)
  DistributionArray<real>* avg_tmp;
  //! Value map for average gathering
  thrust::device_vector<int>* avgMap;
  //! Stencil for average gathering
  thrust::host_vector<int>* avgStencil;
  //! Stores results gathered from the averaging
  DistributionArray<real>* avgResult;
  //! Velocity distribution functions
  DistributionFunction* df;
  //! Velocity distribution functions (for swap)
  DistributionFunction* df_tmp;
  //! Temperature distribution functions
  DistributionFunction* dfT;
  //! Temp. distribution functions (for swap)
  DistributionFunction* dfT_tmp;
  //! Internal temperature distribution functions
  DistributionFunction* dfTeff;
  //! Internal temperature distribution functions (for swap)
  DistributionFunction* dfTeff_tmp;
  //! Plot array for slice renderer
  DistributionArray<real>* plot;
  //! Plot array (for swap)
  DistributionArray<real>* plot_tmp;
  //! The array of voxels
  VoxelArray* voxels;
  //! The boundary conditions
  thrust::device_vector<BoundaryCondition>* bcs;

  ~SimulationState() {
    delete df;
    delete df_tmp;
    delete dfT;
    delete dfT_tmp;
    delete dfTeff;
    delete dfTeff_tmp;
    delete avg;
    delete avg_tmp;
    delete avgMap;
    delete avgStencil;
    delete avgResult;
    delete plot;
    delete plot_tmp;
    delete voxels;
  }

  SimulationState()
      : df(nullptr),
        df_tmp(nullptr),
        dfT(nullptr),
        dfT_tmp(nullptr),
        dfTeff(nullptr),
        dfTeff_tmp(nullptr),
        avg(nullptr),
        avg_tmp(nullptr),
        avgMap(nullptr),
        avgStencil(nullptr),
        avgResult(nullptr),
        plot(nullptr),
        plot_tmp(nullptr),
        voxels(nullptr),
        bcs(nullptr) {}

  explicit SimulationState(const SimulationState& kp)
      : df(kp.df),
        df_tmp(kp.df_tmp),
        dfT(kp.dfT),
        dfT_tmp(kp.dfT_tmp),
        dfTeff(kp.dfTeff),
        dfTeff_tmp(kp.dfTeff_tmp),
        avg(kp.avg),
        avg_tmp(kp.avg_tmp),
        avgMap(kp.avgMap),
        avgStencil(kp.avgStencil),
        avgResult(kp.avgResult),
        plot(kp.plot),
        plot_tmp(kp.plot_tmp),
        voxels(kp.voxels),
        bcs(kp.bcs) {}
};
