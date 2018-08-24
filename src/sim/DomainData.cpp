#include "DomainData.hpp"

void DomainData::buildKernel(std::string settingsPath, std::string buildGeometryPath)
{
  LuaContext lua;

  m_unitConverter = std::make_shared<UnitConverter>();
  lua.writeVariable("ucAdapter", m_unitConverter);
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

  std::ifstream settingsScript = std::ifstream{settingsPath};
  try
  {
    lua.executeCode(settingsScript);
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
      std::cout << e.what() << std::endl;
    }
  }
  m_kernelParam = new KernelParameters();
  try
  {
    // TODO: Put load() in subclass?
    m_kernelParam->nx = lua.readVariable<float>("nx");
    m_kernelParam->ny = lua.readVariable<float>("ny");
    m_kernelParam->nz = lua.readVariable<float>("nz");
    m_kernelParam->nu = lua.readVariable<float>("nu");
    m_kernelParam->C = lua.readVariable<float>("C");
    m_kernelParam->nuT = lua.readVariable<float>("nuT");
    m_kernelParam->Pr = lua.readVariable<float>("Pr");
    m_kernelParam->Pr_t = lua.readVariable<float>("Pr_t");
    m_kernelParam->gBetta = lua.readVariable<float>("gBetta");
    m_kernelParam->Tinit = lua.readVariable<float>("Tinit");
    m_kernelParam->Tref = lua.readVariable<float>("Tref");
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
      std::cout << e.what() << std::endl;
    }
  }
  settingsScript.close();

  m_voxGeo = std::make_shared<VoxelGeometry>(m_kernelParam->nx,
                                             m_kernelParam->ny,
                                             m_kernelParam->nz,
                                             m_unitConverter);
  lua.writeVariable("voxGeoAdapter", m_voxGeo);
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

  std::ifstream buildScript = std::ifstream{buildGeometryPath};
  try
  {
    lua.executeCode(buildScript);
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
      std::cout << e.what() << std::endl;
    }
  }
  buildScript.close();

  m_bcs = &(m_voxGeo->voxdetail);

  std::cout << "Number of lattice site types: " << m_voxGeo->getNumTypes() << std::endl;

  m_kernelData = new KernelData(
      m_kernelParam,
      m_bcs,
      m_voxGeo->data);
}

DomainData::DomainData()
{
  buildKernel("problems/data_center/settings.lua", "problems/data_center/buildGeometry.lua");
  // buildKernel("problems/hospital/settings.lua", "problems/hospital/buildGeometry.lua");
}