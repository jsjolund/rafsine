#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <LuaContext.hpp>

#include "KernelInterface.hpp"
#include "LuaGeometry.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"

class LuaData {
 public:
  int m_nx, m_ny, m_nz;
  //! The real-to-lbm unit converter loaded from Lua
  std::shared_ptr<UnitConverter> m_unitConverter;
  //! Voxel/lattice geometry loaded from Lua script
  std::shared_ptr<VoxelGeometry> m_voxGeo;
  //! Some parameters for the CUDA kernel
  std::shared_ptr<ComputeParams> m_param;
  //! Averaging period
  float m_avgPeriod;

  void loadFromLua(std::string buildGeometryPath, std::string settingsPath);
};

/**
 * @brief Stores the data of a specific CFD problem
 *
 */
class DomainData : public LuaData {
 public:
  //! Interface to CUDA kernel
  std::shared_ptr<KernelInterface> m_kernel;
  //! An ordered list of boundary condition details
  std::shared_ptr<BoundaryConditions> m_bcs;
  //! Areas on which to perform temperature and velocity averaging
  std::shared_ptr<VoxelVolumeArray> m_avgs;
  //! Timer counting time passed in the simulation
  std::shared_ptr<SimulationTimer> m_timer;

  /**
   * @brief Loads the previous class members from Lua script
   *
   * @param buildGeometryPath
   * @param settingsPath
   */
  void loadFromLua(int numDevices,
                   std::string buildGeometryPath,
                   std::string settingsPath);

  DomainData() {}

  ~DomainData() { std::cout << "Destroying domain data" << std::endl; }
};
