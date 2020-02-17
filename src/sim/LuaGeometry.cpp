#include "LuaGeometry.hpp"

void LuaGeometry::addQuadBCNodeUnits(VoxelQuad* quad) {
  storeType(&(quad->m_bc), quad->m_name);

  Eigen::Vector3i origin = quad->m_voxOrigin;
  Eigen::Vector3i dir1 = quad->m_voxDir1;
  Eigen::Vector3i dir2 = quad->m_voxDir2;
  int l1 =
      static_cast<int>(sqrt(dir1.x() * dir1.x()) + sqrt(dir1.y() * dir1.y()) +
                       sqrt(dir1.z() * dir1.z()));
  int l2 =
      static_cast<int>(sqrt(dir2.x() * dir2.x()) + sqrt(dir2.y() * dir2.y()) +
                       sqrt(dir2.z() * dir2.z()));
  Eigen::Vector3i dir1n =
      Eigen::Vector3i(sgn(dir1.x()), sgn(dir1.y()), sgn(dir1.z()));
  Eigen::Vector3i dir2n =
      Eigen::Vector3i(sgn(dir2.x()), sgn(dir2.y()), sgn(dir2.z()));

  for (int i = 0; i <= l1; i++) {
    for (int j = 0; j <= l2; j++) {
      Eigen::Vector3i p = origin + i * dir1n + j * dir2n;
      set(p, quad->m_bc, quad->m_mode, quad->m_name);
    }
  }
  if (m_incompatible > 0)
    std::cout << "Warning: Intersecting incompatible boundary conditions ("
              << m_incompatible << " voxels) in geometry '" << quad->m_name
              << "'!" << std::endl;
  m_nameToQuadMap[quad->m_name].insert(*quad);
}

void LuaGeometry::addQuadBC(std::string name,
                            std::string mode,
                            real originX,
                            real originY,
                            real originZ,
                            real dir1X,
                            real dir1Y,
                            real dir1Z,
                            real dir2X,
                            real dir2Y,
                            real dir2Z,
                            int normalX,
                            int normalY,
                            int normalZ,
                            std::string typeBC,
                            std::string temperatureType,
                            real temperature,
                            real velocityX,
                            real velocityY,
                            real velocityZ,
                            real rel_pos,
                            real tau1,
                            real tau2,
                            real lambda) {
  NodeMode::Enum modeEnum;
  if (mode.compare("overwrite") == 0)
    modeEnum = NodeMode::OVERWRITE;
  else if (mode.compare("intersect") == 0)
    modeEnum = NodeMode::INTERSECT;
  else if (mode.compare("fill") == 0)
    modeEnum = NodeMode::FILL;
  else
    throw std::runtime_error(ErrorFormat() << mode << " is unknown mode");

  VoxelType::Enum typeBcEnum;
  if (typeBC.compare("empty") == 0) {
    typeBcEnum = VoxelType::EMPTY;
  } else if (typeBC.compare("fluid") == 0) {
    typeBcEnum = VoxelType::FLUID;
  } else if (typeBC.compare("wall") == 0) {
    typeBcEnum = VoxelType::WALL;
  } else if (typeBC.compare("freeSlip") == 0) {
    typeBcEnum = VoxelType::FREE_SLIP;
  } else if (typeBC.compare("inlet") == 0) {
    if (temperatureType.compare("constant") == 0)
      typeBcEnum = VoxelType::INLET_CONSTANT;
    else if (temperatureType.compare("zeroGradient") == 0)
      typeBcEnum = VoxelType::INLET_ZERO_GRADIENT;
    else if (temperatureType.compare("relative") == 0)
      typeBcEnum = VoxelType::INLET_RELATIVE;
    else
      throw std::runtime_error(ErrorFormat() << temperatureType
                                             << " is unknown temperature type");
  } else {
    throw std::runtime_error(ErrorFormat()
                             << typeBC << " is unknown boundary condition");
  }
  if (name.length() == 0) name = DEFAULT_GEOMETRY_NAME;

  Eigen::Vector3f origin(originX, originY, originZ);
  Eigen::Vector3f dir1(dir1X, dir1Y, dir1Z);
  Eigen::Vector3f dir2(dir2X, dir2Y, dir2Z);
  Eigen::Vector3i normal(normalX, normalY, normalZ);
  Eigen::Vector3i voxOrigin = m_uc->m_to_LUA_vec(origin);
  Eigen::Vector3i voxDir1 = m_uc->m_to_LUA_vec(origin + dir1) - voxOrigin;
  Eigen::Vector3i voxDir2 = m_uc->m_to_LUA_vec(origin + dir2) - voxOrigin;

  Eigen::Vector3i relPosV(0, 0, 0);
  if (!std::isnan(rel_pos)) relPosV = -(1 + m_uc->m_to_lu(rel_pos)) * normal;

  VoxelQuad quad(name, modeEnum, voxOrigin, voxDir1, voxDir2, normal,
                 typeBcEnum, temperature, tau1, tau2, lambda,
                 Eigen::Vector3f(velocityX, velocityY, velocityZ), relPosV,
                 origin, dir1, dir2);

  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addSensor(std::string name,
                            real minX,
                            real minY,
                            real minZ,
                            real maxX,
                            real maxY,
                            real maxZ) {
  Eigen::Vector3f min(minX, minY, minZ);
  Eigen::Vector3f max(maxX, maxY, maxZ);
  Eigen::Vector3i voxMin = m_uc->m_to_lu_vec(min);
  Eigen::Vector3i voxMax = m_uc->m_to_lu_vec(max);
  if (voxMax.x() == voxMin.x()) voxMax.x() += 1;
  if (voxMax.y() == voxMin.y()) voxMax.y() += 1;
  if (voxMax.z() == voxMin.z()) voxMax.z() += 1;
  VoxelVolume sensorArea(name, voxMin, voxMax, min, max);
  m_sensorArray->push_back(sensorArea);
}

void LuaGeometry::addWallXmin() {
  Eigen::Vector3i n(1, 0, 0);
  Eigen::Vector3i origin(1, 1, 1);
  Eigen::Vector3i dir1(0, getSizeY() - 1, 0);
  Eigen::Vector3i dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallXmax() {
  Eigen::Vector3i n(-1, 0, 0);
  Eigen::Vector3i origin(getSizeX(), 1, 1);
  Eigen::Vector3i dir1(0, getSizeY() - 1, 0);
  Eigen::Vector3i dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallYmin() {
  Eigen::Vector3i n(0, 1, 0);
  Eigen::Vector3i origin(1, 1, 1);
  Eigen::Vector3i dir1(getSizeX() - 1, 0, 0);
  Eigen::Vector3i dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallYmax() {
  Eigen::Vector3i n(0, -1, 0);
  Eigen::Vector3i origin(1, getSizeY(), 1);
  Eigen::Vector3i dir1(getSizeX() - 1, 0, 0);
  Eigen::Vector3i dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallZmin() {
  Eigen::Vector3i n(0, 0, 1);
  Eigen::Vector3i origin(1, 1, 1);
  Eigen::Vector3i dir1(getSizeX() - 1, 0, 0);
  Eigen::Vector3i dir2(0, getSizeY() - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallZmax() {
  Eigen::Vector3i n(0, 0, -1);
  Eigen::Vector3i origin(1, 1, getSizeZ());
  Eigen::Vector3i dir1(getSizeX() - 1, 0, 0);
  Eigen::Vector3i dir2(0, getSizeY() - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::makeHollow(Eigen::Vector3f min,
                             Eigen::Vector3f max,
                             bool minXface,
                             bool minYface,
                             bool minZface,
                             bool maxXface,
                             bool maxYface,
                             bool maxZface) {
  Eigen::Vector3i imin = m_uc->m_to_LUA_vec(min);
  Eigen::Vector3i imax = m_uc->m_to_LUA_vec(max);
  imin += Eigen::Vector3i(1, 1, 1);
  imax -= Eigen::Vector3i(1, 1, 1);
  if (minXface) imin.x()--;
  if (minYface) imin.y()--;
  if (minZface) imin.z()--;
  if (maxXface) imax.x()++;
  if (maxYface) imax.y()++;
  if (maxZface) imax.z()++;
  for (int x = imin.x(); x <= imax.x(); x++)
    for (int y = imin.y(); y <= imax.y(); y++)
      for (int z = imin.z(); z <= imax.z(); z++)
        set(x, y, z, VoxelType::Enum::EMPTY);
}

void LuaGeometry::makeHollow(real minX,
                             real minY,
                             real minZ,
                             real maxX,
                             real maxY,
                             real maxZ,
                             bool minXface,
                             bool minYface,
                             bool minZface,
                             bool maxXface,
                             bool maxYface,
                             bool maxZface) {
  makeHollow(Eigen::Vector3f(minX, minY, minZ),
             Eigen::Vector3f(maxX, maxY, maxZ), minXface, minYface, minZface,
             maxXface, maxYface, maxZface);
}

void LuaGeometry::addSolidBox(std::string name,
                              real minX,
                              real minY,
                              real minZ,
                              real maxX,
                              real maxY,
                              real maxZ,
                              real temperature) {
  if (name.length() == 0) name = DEFAULT_GEOMETRY_NAME;

  Eigen::Vector3f min(minX, minY, minZ);
  Eigen::Vector3f max(maxX, maxY, maxZ);
  Eigen::Vector3i voxMin = m_uc->m_to_LUA_vec(min);
  Eigen::Vector3i voxMax = m_uc->m_to_LUA_vec(max);
  VoxelBox box(name, voxMin, voxMax, min, max, temperature);
  for (int i = 0; i < box.m_quads.size(); i++)
    addQuadBCNodeUnits(&(box.m_quads.at(i)));

  makeHollow(box.m_min.x(), box.m_min.y(), box.m_min.z(), box.m_max.x(),
             box.m_max.y(), box.m_max.z(), voxMin.x() <= 1, voxMin.y() <= 1,
             voxMin.z() <= 1, voxMax.x() >= getSizeX(),
             voxMax.y() >= getSizeY(), voxMax.z() >= getSizeZ());
}
