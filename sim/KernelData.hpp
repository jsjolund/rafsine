#pragma once

#include "../geo/VoxelGeometry.hpp"
#include "../sim/SimConstants.hpp"
#include "../sim/UnitConverter.hpp"

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