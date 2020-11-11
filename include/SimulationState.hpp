#pragma once

#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "VoxelArray.hpp"

/**
 * @brief Holds pointers to distribution functions, plots, averaging and
 * boundary conditions for the current simulation
 */
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

  /**
   * @brief Copy constructor
   *
   * @param state
   */
  explicit SimulationState(const SimulationState& state)
      : df(state.df),
        df_tmp(state.df_tmp),
        dfT(state.dfT),
        dfT_tmp(state.dfT_tmp),
        dfTeff(state.dfTeff),
        dfTeff_tmp(state.dfTeff_tmp),
        avg(state.avg),
        avg_tmp(state.avg_tmp),
        avgMap(state.avgMap),
        avgStencil(state.avgStencil),
        avgResult(state.avgResult),
        plot(state.plot),
        plot_tmp(state.plot_tmp),
        voxels(state.voxels),
        bcs_id(state.bcs_id),
        bcs_type(state.bcs_type),
        bcs_temperature(state.bcs_temperature),
        bcs_velocity(state.bcs_velocity),
        bcs_normal(state.bcs_normal),
        bcs_rel_pos(state.bcs_rel_pos),
        bcs_tau1(state.bcs_tau1),
        bcs_tau2(state.bcs_tau2),
        bcs_lambda(state.bcs_lambda) {}
};
