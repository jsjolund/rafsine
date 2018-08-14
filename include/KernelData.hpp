#pragma once

#include "VoxelGeometry.hpp"
#include "SimConstants.hpp"
#include "UnitConverter.hpp"

class KernelData
{
public:
  UnitConverter *uc;
  SimConstants *sc;
  UserConstants *c;
  VoxelGeometry *vox;
  std::vector<VoxelGeometryGroup *> *geo;

  KernelData();
  KernelData(UnitConverter *uc, SimConstants *sc, UserConstants *c, VoxelGeometry *vox);
  ~KernelData() { delete geo; }
};