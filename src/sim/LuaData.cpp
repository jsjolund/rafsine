#include "LuaData.hpp"

template <typename S, typename D>
void LuaData::readVariable(const std::string var, D* dst, LuaContext* lua) {
  try {
    *dst = lua->readVariable<S>(var);
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    throw std::runtime_error("Error reading simulation input parameter " + var);
  }
}

void LuaData::loadSimulation(const std::string buildGeometryPath,
                             const std::string settingsPath) {
  LuaContext lua;

  // Register Lua functions for settings.lua
  m_unitConverter = std::make_shared<UnitConverter>();
  lua.writeVariable("ucAdapter", m_unitConverter);
  lua.registerFunction("round", &UnitConverter::round);
  lua.registerFunction("set", &UnitConverter::set);
  lua.registerFunction("m_to_lu",
                       (int (UnitConverter::*)(real_t))(&UnitConverter::m_to_lu));
  lua.registerFunction(
      "m_to_LUA", (int (UnitConverter::*)(real_t))(&UnitConverter::m_to_LUA));
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

  // Execute settings.lua
  std::ifstream settingsScript = std::ifstream{settingsPath};
  try {
    lua.executeCode(settingsScript);
  } catch (const LuaContext::ExecutionErrorException& e) {
    std::cerr << e.what() << std::endl;
    try {
      std::rethrow_if_nested(e);
    } catch (const std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }
  }

  // Read required parameters from settings.lua
  m_param = std::make_shared<SimulationParams>();
  readVariable<float, int>("nx", &m_nx, &lua);
  readVariable<float, int>("ny", &m_ny, &lua);
  readVariable<float, int>("nz", &m_nz, &lua);
  readVariable<float, float>("nu", &m_param->nu, &lua);
  readVariable<float, float>("C", &m_param->C, &lua);
  readVariable<float, float>("nuT", &m_param->nuT, &lua);
  readVariable<float, float>("Pr_t", &m_param->Pr_t, &lua);
  readVariable<float, float>("gBetta", &m_param->gBetta, &lua);
  readVariable<float, float>("Tinit", &m_param->Tinit, &lua);
  readVariable<float, float>("Tref", &m_param->Tref, &lua);
  readVariable<float, float>("avgPeriod", &m_avgPeriod, &lua);

  std::string lbmMethod;
  readVariable<std::string, std::string>("method", &lbmMethod, &lua);
  if (lbmMethod.compare("MRT") == 0)
    m_method = LBM::MRT;
  else if (lbmMethod.compare("BGK") == 0)
    m_method = LBM::BGK;
  else
    std::cerr << "Invalid LBM method" << std::endl;

  std::string partitioning;
  readVariable<std::string, std::string>("partitioning", &partitioning, &lua);
  if (partitioning.compare("X") == 0)
    m_partitioning = D3Q4::X_AXIS;
  else if (partitioning.compare("Y") == 0)
    m_partitioning = D3Q4::Y_AXIS;
  else if (partitioning.compare("Z") == 0)
    m_partitioning = D3Q4::Z_AXIS;
  else
    std::cerr << "Invalid partitioning axis" << std::endl;
  settingsScript.close();

  // Register functions for geometry.lua
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
      "makeHollow", (void (LuaGeometry::*)(real_t, real_t, real_t, real_t, real_t, real_t,
                                           bool, bool, bool, bool, bool,
                                           bool))(&LuaGeometry::makeHollow));
  // Execute geometry.lua
  std::ifstream buildScript = std::ifstream{buildGeometryPath};
  try {
    lua.executeCode(buildScript);
  } catch (const LuaContext::ExecutionErrorException& e) {
    std::cerr << e.what() << std::endl;
    try {
      std::rethrow_if_nested(e);
    } catch (const std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }
    throw std::runtime_error("Error executing " + buildGeometryPath);
  }
  buildScript.close();

  QCryptographicHash hash(QCryptographicHash::Sha1);
  QFile geometryFile(QString::fromStdString(buildGeometryPath));
  geometryFile.open(QFile::ReadOnly);
  hash.addData(&geometryFile);
  hash.addData(partitioning.c_str(), partitioning.length());
  QByteArray arr;
  QDataStream stream(&arr, QIODevice::WriteOnly);
  stream << m_nx;
  stream << m_ny;
  stream << m_nz;
  hash.addData(arr);
  m_hash = QString(hash.result().toHex()).toUtf8().constData();
}
