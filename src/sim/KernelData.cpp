#include "KernelData.hpp"

void KernelData::buildKernel(std::string settingsPath, std::string buildGeometryPath)
{

  LuaContext lua;

  uc = std::make_shared<UnitConverter>();
  lua.writeVariable("ucAdapter", uc);
  lua.registerFunction("round", &UnitConverter::round);
  lua.registerFunction("set", &UnitConverter::set);
  lua.registerFunction("m_to_lu", (int (UnitConverter::*)(real))(&UnitConverter::m_to_lu));
  lua.registerFunction("m_to_LUA", (int (UnitConverter::*)(real))(&UnitConverter::m_to_LUA));
  lua.registerFunction("ms_to_lu", &UnitConverter::ms_to_lu);
  lua.registerFunction("Q_to_Ulu", &UnitConverter::Q_to_Ulu);
  lua.registerFunction("Nu_to_lu", &UnitConverter::Nu_to_lu);
  lua.registerFunction("Nu_to_tau", &UnitConverter::Nu_to_tau);
  lua.registerFunction("N_to_s", &UnitConverter::N_to_s);
  lua.registerFunction("s_to_N", &UnitConverter::s_to_N);
  lua.registerFunction("Temp_to_lu", &UnitConverter::Temp_to_lu);
  lua.registerFunction("gBetta_to_lu", &UnitConverter::gBetta_to_lu);

  std::ifstream script = std::ifstream{"problems/data_center/settings.lua"};
  try
  {
    lua.executeCode(script);
  }
  catch (const LuaContext::ExecutionErrorException &e)
  {
    std::cout << e.what() << std::endl; // prints an error message

    try
    {
      std::rethrow_if_nested(e);
    }
    catch (const std::runtime_error &e)
    {
      // e is the exception that was thrown from inside the lambda
      std::cout << e.what() << std::endl; // prints "Problem"
    }
  }
  script.close();

  float nx = lua.readVariable<float>("nx");
  float ny = lua.readVariable<float>("ny");
  float nz = lua.readVariable<float>("nz");

  vox = std::make_shared<VoxelGeometry>(nx, ny, nz, uc);
  lua.writeVariable("voxGeoAdapter", vox);
  lua.registerFunction("addWallXmin", &VoxelGeometry::addWallXmin);
  lua.registerFunction("addWallYmin", &VoxelGeometry::addWallYmin);
  lua.registerFunction("addWallZmin", &VoxelGeometry::addWallZmin);
  lua.registerFunction("addWallXmax", &VoxelGeometry::addWallXmax);
  lua.registerFunction("addWallYmax", &VoxelGeometry::addWallYmax);
  lua.registerFunction("addWallZmax", &VoxelGeometry::addWallZmax);
  lua.registerFunction("addQuadBC", &VoxelGeometry::createAddQuadBC);
  lua.registerFunction("addSolidBox", &VoxelGeometry::createAddSolidBox);
  lua.registerFunction("makeHollow",
                       (void (VoxelGeometry::*)(real, real, real,
                                                real, real, real,
                                                bool, bool, bool,
                                                bool, bool, bool))(&VoxelGeometry::makeHollow));

  script = std::ifstream{"problems/data_center/buildGeometry.lua"};

  try
  {
    lua.executeCode(script);
  }
  catch (const LuaContext::ExecutionErrorException &e)
  {
    std::cout << e.what() << std::endl;
    try
    {
      std::rethrow_if_nested(e);
    }
    catch (const std::runtime_error &e)
    {
      // e is the exception that was thrown from inside the lambda
      std::cout << e.what() << std::endl; // prints "Problem"
    }
  }
  script.close();

  std::cout << "Number of lattice site types: " << vox->getNumTypes() << std::endl;
}

KernelData::KernelData()
{
}