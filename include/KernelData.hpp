#pragma once

#include <memory>
#include <iostream>
#include <fstream>

#include "LuaContext.hpp"

#include "VoxelGeometry.hpp"
#include "UnitConverter.hpp"

class KernelData
{
public:
  std::shared_ptr<UnitConverter> uc;
  std::shared_ptr<VoxelGeometry> vox;

  KernelData();

  ~KernelData() {}
};