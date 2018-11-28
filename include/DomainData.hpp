#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <LuaContext.hpp>

#include "KernelInterface.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"
#include "VoxelGeometry.hpp"

/**
 * @brief Stores the data of a specific CFD problem
 *
 */
class DomainData {
 private:
  int m_numDevices;

 public:
  std::shared_ptr<UnitConverter>
      m_unitConverter;  //!< The real-to-lbm unit converter loaded from Lua

  std::shared_ptr<VoxelGeometry>
      m_voxGeo;  //!< Voxel/lattice geometry loaded from Lua script

  KernelInterface *m_kernel;  //!< Interface to CUDA kernel

  ComputeKernelParams *m_param;  //!< Some parameters for the CUDA kernel

  BoundaryConditionsArray
      *m_bcs;  //!< An ordered list of boundary condition details

  SimulationTimer *m_timer;  //!< Timer counting time passed in the simulation

  /**
   * @brief Loads the previous class members from Lua script
   *
   * @param buildGeometryPath
   * @param settingsPath
   */
  void loadFromLua(std::string buildGeometryPath, std::string settingsPath);

  int getNumDevices() { return m_numDevices; }

  DomainData(int numDevices);
  ~DomainData();
};
