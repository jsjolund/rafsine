#pragma once

#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "VoxelArray.hpp"

class SimulationState {
 public:
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
  /**
   * Contains the macroscopic temperature, velocity (x,y,z components)
   * integrated in time, so divide by number of time steps to get average).
   * 0 -> temperature
   * 1 -> x-component of velocity
   * 2 -> y-component of velocity
   * 3 -> z-component of velocity
   */
  DistributionArray<real_t>* avg;
  //! Average array (for swap)
  DistributionArray<real_t>* avg_tmp;
  //! Value map for average gathering
  thrust::device_vector<int>* avgMap;
  //! Stencil for average gathering
  thrust::host_vector<int>* avgStencil;
  //! Stores results gathered from the averaging
  DistributionArray<real_t>* avgResult;
  //! Plot array for slice renderer
  DistributionArray<real_t>* plot;
  //! Plot array (for swap)
  DistributionArray<real_t>* plot_tmp;
  //! The array of voxels
  VoxelArray* voxels;
  //! Boundary condition IDs
  thrust::device_vector<voxel_t>* bcs_id;
  //! Boundary condition types, e.g. walls, inlets, outlets
  thrust::device_vector<VoxelType::Enum>* bcs_type;
  //! Boundary condition constant temperatures
  thrust::device_vector<real_t>* bcs_temperature;
  //! Boundary condition constant velocity
  thrust::device_vector<real3_t>* bcs_velocity;
  //! Boundary condition normals
  thrust::device_vector<int3>* bcs_normal;
  //! Boundary condition relative positions for velocity and temperature
  thrust::device_vector<int3>* bcs_rel_pos;
  //! Boundary condition temperature constant 1
  thrust::device_vector<real_t>* bcs_tau1;
  //! Boundary condition temperature constant 2
  thrust::device_vector<real_t>* bcs_tau2;
  //! Boundary condition temperature constant 3
  thrust::device_vector<real_t>* bcs_lambda;

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
    // delete bcs_id;
    // delete bcs_type;
    // delete bcs_temperature;
    // delete bcs_velocity;
    // delete bcs_normal;
    // delete bcs_rel_pos;
    // delete bcs_tau1;
    // delete bcs_tau2;
    // delete bcs_lambda;
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
        bcs_id(nullptr),
        bcs_type(nullptr),
        bcs_temperature(nullptr),
        bcs_velocity(nullptr),
        bcs_normal(nullptr),
        bcs_rel_pos(nullptr),
        bcs_tau1(nullptr),
        bcs_tau2(nullptr),
        bcs_lambda(nullptr) {}

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
        bcs_id(kp.bcs_id),
        bcs_type(kp.bcs_type),
        bcs_temperature(kp.bcs_temperature),
        bcs_velocity(kp.bcs_velocity),
        bcs_normal(kp.bcs_normal),
        bcs_rel_pos(kp.bcs_rel_pos),
        bcs_tau1(kp.bcs_tau1),
        bcs_tau2(kp.bcs_tau2),
        bcs_lambda(kp.bcs_lambda) {}
};
