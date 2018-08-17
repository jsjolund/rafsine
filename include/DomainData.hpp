#pragma once

#include <memory>
#include <iostream>
#include <fstream>

#include "LuaContext.hpp"

#include "VoxelGeometry.hpp"
#include "UnitConverter.hpp"
#include "KernelData.hpp"

class DomainData
{
private:
public:
  std::shared_ptr<UnitConverter> m_unitConverter;
  std::shared_ptr<VoxelGeometry> m_voxGeo;
  KernelData *m_kernelData;
  KernelParameters *m_kernelParam;
  BoundaryConditions *m_bcs;

  void buildKernel(std::string settingsPath, std::string buildGeometryPath);

  DomainData();
  ~DomainData() {}
};