#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <LuaContext.hpp>

#include "LuaGeometry.hpp"
#include "SimulationParams.hpp"
#include "UnitConverter.hpp"

/**
 * @brief Load data from lua code for a CFD problem
 *
 */
class LuaData {
 private:
  template <typename T>
  void readNumber(const std::string var, T* dst, LuaContext* lua);

 public:
  int m_nx, m_ny, m_nz;
  //! The real-to-lbm unit converter loaded from Lua
  std::shared_ptr<UnitConverter> m_unitConverter;
  //! Voxel/lattice geometry loaded from Lua script
  std::shared_ptr<VoxelGeometry> m_voxGeo;
  //! Some parameters for the CUDA kernel
  std::shared_ptr<SimulationParams> m_param;
  //! Averaging period
  float m_avgPeriod;

  void loadSimulation(const std::string buildGeometryPath,
                      const std::string settingsPath);
};
