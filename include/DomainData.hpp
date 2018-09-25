#pragma once

#include <fstream>
#include <iostream>
#include <memory>

#include "KernelData.hpp"
#include "LuaContext.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"
#include "VoxelGeometry.hpp"

class DomainData
{
private:
public:
  std::shared_ptr<UnitConverter> m_unitConverter;
  std::shared_ptr<VoxelGeometry> m_voxGeo;
  KernelData *m_kernelData;
  KernelParameters *m_kernelParam;
  BoundaryConditionsArray *m_bcs;
  SimulationTimer *m_simTimer;

  void loadFromLua(std::string buildGeometryPath, std::string settingsPath);

  DomainData();
  ~DomainData();
};