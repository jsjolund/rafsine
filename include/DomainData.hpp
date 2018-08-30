#pragma once

#include <memory>
#include <iostream>
#include <fstream>

#include "LuaContext.hpp"

#include "VoxelGeometry.hpp"
#include "UnitConverter.hpp"
#include "KernelData.hpp"
#include "SimulationTimer.hpp"

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

  void loadFromLua(std::string settingsPath, std::string buildGeometryPath);

  DomainData();
  ~DomainData() {}
};