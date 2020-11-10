#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "KernelInterface.hpp"
#include "LuaData.hpp"
#include "LuaGeometry.hpp"
#include "SimulationTimer.hpp"

/**
 * @brief Stores the data of a CFD problem
 *
 */
class DomainData : public LuaData {
 public:
  //! Interface to CUDA kernel
  std::shared_ptr<KernelInterface> m_kernel;
  //! An ordered list of boundary condition details
  std::shared_ptr<BoundaryConditions> m_bcs;
  //! Areas on which to perform temperature and velocity averaging
  std::shared_ptr<VoxelCuboidArray> m_avgs;
  //! Timer counting time passed in the simulation
  std::shared_ptr<SimulationTimer> m_timer;

  /**
   * @brief Loads the previous class members from Lua script
   *
   * @param nd Number of CUDA devices
   * @param buildGeometryPath Path to geometry.lua
   * @param settingsPath Path to settings.lua
   */
  void loadSimulation(int nd,
                      std::string buildGeometryPath,
                      std::string settingsPath);

  DomainData() {}

  ~DomainData() { std::cout << "Destroying domain data" << std::endl; }
};
