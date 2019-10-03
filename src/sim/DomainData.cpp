#include "DomainData.hpp"

void LuaData::loadFromLua(std::string buildGeometryPath,
                          std::string settingsPath) {
  LuaContext lua;

  m_unitConverter = std::make_shared<UnitConverter>();
  lua.writeVariable("ucAdapter", m_unitConverter);
  lua.registerFunction("round", &UnitConverter::round);
  lua.registerFunction("set", &UnitConverter::set);
  lua.registerFunction("m_to_lu",
                       (int (UnitConverter::*)(real))(&UnitConverter::m_to_lu));
  lua.registerFunction(
      "m_to_LUA", (int (UnitConverter::*)(real))(&UnitConverter::m_to_LUA));
  lua.registerFunction("ms_to_lu", &UnitConverter::ms_to_lu);
  lua.registerFunction("Q_to_Ulu", &UnitConverter::Q_to_Ulu);
  lua.registerFunction("Nu_to_lu", &UnitConverter::Nu_to_lu);
  lua.registerFunction("Nu_to_tau", &UnitConverter::Nu_to_tau);
  lua.registerFunction("N_to_s", &UnitConverter::N_to_s);
  lua.registerFunction("s_to_N", &UnitConverter::s_to_N);
  lua.registerFunction("Temp_to_lu", &UnitConverter::Temp_to_lu);
  lua.registerFunction("gBetta_to_lu", &UnitConverter::gBetta_to_lu);
  lua.registerFunction("C_L", &UnitConverter::C_L);
  lua.registerFunction("C_U", &UnitConverter::C_U);
  lua.registerFunction("C_T", &UnitConverter::C_T);

  std::ifstream settingsScript = std::ifstream{settingsPath};
  try {
    lua.executeCode(settingsScript);
  } catch (const LuaContext::ExecutionErrorException &e) {
    std::cout << e.what() << std::endl;
    try {
      std::rethrow_if_nested(e);
    } catch (const std::runtime_error &e) {
      std::cout << e.what() << std::endl;
    }
  }
  m_param = std::make_shared<ComputeParams>();
  try {
    m_nx = lua.readVariable<float>("nx");
    m_ny = lua.readVariable<float>("ny");
    m_nz = lua.readVariable<float>("nz");
    m_param->nu = lua.readVariable<float>("nu");
    m_param->C = lua.readVariable<float>("C");
    m_param->nuT = lua.readVariable<float>("nuT");
    m_param->Pr = lua.readVariable<float>("Pr");
    m_param->Pr_t = lua.readVariable<float>("Pr_t");
    m_param->gBetta = lua.readVariable<float>("gBetta");
    m_param->Tinit = lua.readVariable<float>("Tinit");
    m_param->Tref = lua.readVariable<float>("Tref");
    m_avgPeriod = lua.readVariable<float>("avgPeriod");
  } catch (const LuaContext::ExecutionErrorException &e) {
    std::cout << e.what() << std::endl;
    try {
      std::rethrow_if_nested(e);
    } catch (const std::runtime_error &e) {
      std::cout << e.what() << std::endl;
    }
  }
  settingsScript.close();

  m_voxGeo = std::make_shared<LuaGeometry>(m_nx, m_ny, m_nz, m_unitConverter);
  lua.writeVariable("voxGeoAdapter",
                    std::static_pointer_cast<LuaGeometry>(m_voxGeo));
  lua.registerFunction("addWallXmin", &LuaGeometry::addWallXmin);
  lua.registerFunction("addWallYmin", &LuaGeometry::addWallYmin);
  lua.registerFunction("addWallZmin", &LuaGeometry::addWallZmin);
  lua.registerFunction("addWallXmax", &LuaGeometry::addWallXmax);
  lua.registerFunction("addWallYmax", &LuaGeometry::addWallYmax);
  lua.registerFunction("addWallZmax", &LuaGeometry::addWallZmax);
  lua.registerFunction("addQuadBC", &LuaGeometry::addQuadBC);
  lua.registerFunction("addSensor", &LuaGeometry::addSensor);
  lua.registerFunction("addSolidBox", &LuaGeometry::addSolidBox);
  // makeHollow is overloaded, so specify parameters of the one to use
  lua.registerFunction(
      "makeHollow", (void (LuaGeometry::*)(real, real, real, real, real, real,
                                           bool, bool, bool, bool, bool,
                                           bool))(&LuaGeometry::makeHollow));

  std::ifstream buildScript = std::ifstream{buildGeometryPath};
  try {
    lua.executeCode(buildScript);
  } catch (const LuaContext::ExecutionErrorException &e) {
    std::cout << e.what() << std::endl;
    try {
      std::rethrow_if_nested(e);
    } catch (const std::runtime_error &e) {
      std::cout << e.what() << std::endl;
    }
  }
  buildScript.close();
}

void DomainData::loadFromLua(int numDevices, std::string buildGeometryPath,
                             std::string settingsPath) {
  LuaData::loadFromLua(buildGeometryPath, settingsPath);

  m_bcs = m_voxGeo->getBoundaryConditions();
  m_avgs = m_voxGeo->getSensors();
  if (m_avgPeriod <= 0.0) m_avgs->clear();

  std::cout << "Number of lattice site types: " << m_voxGeo->getNumTypes()
            << std::endl;

  std::cout << "Allocating GPU resources" << std::endl;

  std::shared_ptr<VoxelArray> voxArray = m_voxGeo->getVoxelArray();
  voxArray->upload();
  m_kernel = std::make_shared<KernelInterface>(m_nx, m_ny, m_nz, m_param, m_bcs,
                                               voxArray, m_avgs, numDevices);
  voxArray->deallocate(MemoryType::DEVICE_MEMORY);

  m_timer = std::make_shared<SimulationTimer>(m_nx * m_ny * m_nz,
                                              m_unitConverter->N_to_s(1));
}
