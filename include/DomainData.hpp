#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "KernelData.hpp"
#include "LuaContext.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"
#include "VoxelGeometry.hpp"

/**
 * @brief Stores the data of a specific CFD problem
 *
 */
class DomainData {
 public:
  // The real-to-lbm unit converter loaded from Lua script
  std::shared_ptr<UnitConverter> m_unitConverter;
  // Voxel/lattice geometry loaded from Lua script
  std::shared_ptr<VoxelGeometry> m_voxGeo;
  // Interface to CUDA kernel
  KernelData *m_kernelData;
  // Some parameters for the CUDA kernel
  KernelParameters *m_kernelParam;
  // An ordered list of boundary condition details
  BoundaryConditionsArray *m_bcs;
  // Timer counting time passed in the simulation
  SimulationTimer *m_simTimer;

  /**
   * @brief Loads the previous class members from Lua script
   * 
   * @param buildGeometryPath 
   * @param settingsPath 
   */
  void loadFromLua(std::string buildGeometryPath, std::string settingsPath);

  DomainData();
  ~DomainData();
};
