#include "LuaGeometry.hpp"

void LuaGeometry::addQuadBCNodeUnits(VoxelQuad* quad) {
  storeType(&(quad->m_bc), quad->m_name);

  vector3<int> origin = quad->m_voxOrigin;
  vector3<int> dir1 = quad->m_voxDir1;
  vector3<int> dir2 = quad->m_voxDir2;
  int l1 =
      static_cast<int>(sqrt(dir1.x() * dir1.x()) + sqrt(dir1.y() * dir1.y()) +
                       sqrt(dir1.z() * dir1.z()));
  int l2 =
      static_cast<int>(sqrt(dir2.x() * dir2.x()) + sqrt(dir2.y() * dir2.y()) +
                       sqrt(dir2.z() * dir2.z()));
  vector3<int> dir1n =
      vector3<int>(sgn(dir1.x()), sgn(dir1.y()), sgn(dir1.z()));
  vector3<int> dir2n =
      vector3<int>(sgn(dir2.x()), sgn(dir2.y()), sgn(dir2.z()));

  for (int i = 0; i <= l1; i++) {
    for (int j = 0; j <= l2; j++) {
      vector3<int> p = origin + dir1n * i + dir2n * j;
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
                            real_t originX,
                            real_t originY,
                            real_t originZ,
                            real_t dir1X,
                            real_t dir1Y,
                            real_t dir1Z,
                            real_t dir2X,
                            real_t dir2Y,
                            real_t dir2Z,
                            int normalX,
                            int normalY,
                            int normalZ,
                            std::string typeBC,
                            std::string temperatureType,
                            real_t temperature,
                            real_t velocityX,
                            real_t velocityY,
                            real_t velocityZ,
                            real_t rel_pos,
                            real_t tau1,
                            real_t tau2,
                            real_t lambda) {
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

  vector3<real_t> origin(originX, originY, originZ);
  vector3<real_t> dir1(dir1X, dir1Y, dir1Z);
  vector3<real_t> dir2(dir2X, dir2Y, dir2Z);
  vector3<int> normal(normalX, normalY, normalZ);
  vector3<int> voxOrigin = m_uc->m_to_LUA_vec(origin);
  vector3<int> voxDir1 = m_uc->m_to_LUA_vec(origin + dir1) - voxOrigin;
  vector3<int> voxDir2 = m_uc->m_to_LUA_vec(origin + dir2) - voxOrigin;

  vector3<int> relPosV(0, 0, 0);
  if (!std::isnan(rel_pos)) relPosV = normal * (-(1 + m_uc->m_to_lu(rel_pos)));

  VoxelQuad quad(name, modeEnum, voxOrigin, voxDir1, voxDir2, normal,
                 typeBcEnum, temperature, tau1, tau2, lambda,
                 vector3<real_t>(velocityX, velocityY, velocityZ), relPosV,
                 origin, dir1, dir2);

  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addSensor(std::string name,
                            real_t minX,
                            real_t minY,
                            real_t minZ,
                            real_t maxX,
                            real_t maxY,
                            real_t maxZ) {
  vector3<real_t> min(minX, minY, minZ);
  vector3<real_t> max(maxX, maxY, maxZ);
  vector3<int> voxMin = m_uc->m_to_lu_vec(min);
  vector3<int> voxMax = m_uc->m_to_lu_vec(max);
  if (voxMax.x() == voxMin.x()) voxMax.x() += 1;
  if (voxMax.y() == voxMin.y()) voxMax.y() += 1;
  if (voxMax.z() == voxMin.z()) voxMax.z() += 1;
  VoxelCuboid sensorArea(name, voxMin, voxMax, min, max);
  m_sensorArray->push_back(sensorArea);
}

void LuaGeometry::addWallXmin() {
  vector3<int> n(1, 0, 0);
  vector3<int> origin(1, 1, 1);
  vector3<int> dir1(0, getSizeY() - 1, 0);
  vector3<int> dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallXmax() {
  vector3<int> n(-1, 0, 0);
  vector3<int> origin(getSizeX(), 1, 1);
  vector3<int> dir1(0, getSizeY() - 1, 0);
  vector3<int> dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallYmin() {
  vector3<int> n(0, 1, 0);
  vector3<int> origin(1, 1, 1);
  vector3<int> dir1(getSizeX() - 1, 0, 0);
  vector3<int> dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallYmax() {
  vector3<int> n(0, -1, 0);
  vector3<int> origin(1, getSizeY(), 1);
  vector3<int> dir1(getSizeX() - 1, 0, 0);
  vector3<int> dir2(0, 0, getSizeZ() - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallZmin() {
  vector3<int> n(0, 0, 1);
  vector3<int> origin(1, 1, 1);
  vector3<int> dir1(getSizeX() - 1, 0, 0);
  vector3<int> dir2(0, getSizeY() - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallZmax() {
  vector3<int> n(0, 0, -1);
  vector3<int> origin(1, 1, getSizeZ());
  vector3<int> dir1(getSizeX() - 1, 0, 0);
  vector3<int> dir2(0, getSizeY() - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::makeHollow(vector3<real_t> min,
                             vector3<real_t> max,
                             bool minXface,
                             bool minYface,
                             bool minZface,
                             bool maxXface,
                             bool maxYface,
                             bool maxZface) {
  vector3<int> imin = m_uc->m_to_LUA_vec(min);
  vector3<int> imax = m_uc->m_to_LUA_vec(max);
  imin += vector3<int>(1, 1, 1);
  imax -= vector3<int>(1, 1, 1);
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

void LuaGeometry::makeHollow(real_t minX,
                             real_t minY,
                             real_t minZ,
                             real_t maxX,
                             real_t maxY,
                             real_t maxZ,
                             bool minXface,
                             bool minYface,
                             bool minZface,
                             bool maxXface,
                             bool maxYface,
                             bool maxZface) {
  makeHollow(vector3<real_t>(minX, minY, minZ),
             vector3<real_t>(maxX, maxY, maxZ), minXface, minYface, minZface,
             maxXface, maxYface, maxZface);
}

void LuaGeometry::addSolidBox(std::string name,
                              real_t minX,
                              real_t minY,
                              real_t minZ,
                              real_t maxX,
                              real_t maxY,
                              real_t maxZ,
                              real_t temperature) {
  if (name.length() == 0) name = DEFAULT_GEOMETRY_NAME;

  vector3<real_t> min(minX, minY, minZ);
  vector3<real_t> max(maxX, maxY, maxZ);
  vector3<int> voxMin = m_uc->m_to_LUA_vec(min);
  vector3<int> voxMax = m_uc->m_to_LUA_vec(max);
  VoxelBox box(name, voxMin, voxMax, min, max, temperature);
  for (size_t i = 0; i < box.m_quads.size(); i++)
    addQuadBCNodeUnits(&(box.m_quads.at(i)));

  makeHollow(box.m_min.x(), box.m_min.y(), box.m_min.z(), box.m_max.x(),
             box.m_max.y(), box.m_max.z(), voxMin.x() <= 1, voxMin.y() <= 1,
             voxMin.z() <= 1, voxMax.x() >= static_cast<int>(getSizeX()),
             voxMax.y() >= static_cast<int>(getSizeY()),
             voxMax.z() >= static_cast<int>(getSizeZ()));
}

void LuaGeometry::addSolidSphere(std::string name,
                                 real_t originX,
                                 real_t originY,
                                 real_t originZ,
                                 real_t radius,
                                 real_t temperature) {
  if (name.length() == 0) name = DEFAULT_GEOMETRY_NAME;
  vector3<real_t> origin(originX, originY, originZ);
  vector3<int> voxOrigin = m_uc->m_to_LUA_vec(origin);
  int voxRadius = m_uc->m_to_lu(radius);
  VoxelSphere sphere(name, voxOrigin, origin, voxRadius, temperature);
  // unsigned int voxR = sphere.getRadius();
  vector3<int> offset(voxRadius, voxRadius, voxRadius);

  VoxelType::Enum type = VoxelType::Enum::WALL;
  vector3<real_t> velocity(NaN, NaN, NaN);
  if (!std::isnan(temperature)) {
    type = VoxelType::Enum::INLET_CONSTANT;
    velocity.x() = 0;
    velocity.y() = 0;
    velocity.z() = 0;
  }

  for (unsigned int x = 0; x < sphere.getSizeX(); x++)
    for (unsigned int y = 0; y < sphere.getSizeY(); y++)
      for (unsigned int z = 0; z < sphere.getSizeZ(); z++) {
        vector3<int> p = voxOrigin - offset + vector3<int>(x, y, z);
        SphereVoxel::Enum voxel = sphere.getVoxel(x, y, z);
        vector3<int> normal = sphere.getNormal(x, y, z);
        BoundaryCondition bc(0, type, temperature, velocity, normal,
                             vector3<int>(0, 0, 0), 0, 0, 0);
        switch (voxel) {
          case SphereVoxel::Enum::INSIDE:
            set(p, VoxelType::Enum::EMPTY);
            break;
          case SphereVoxel::Enum::SURFACE:
          // [[fallthrough]];
          case SphereVoxel::Enum::CORNER:
            storeType(&bc, name);
            set(p, bc, NodeMode::OVERWRITE, name);
            break;
          case SphereVoxel::Enum::OUTSIDE:
            break;
          default:
            break;
        }
      }
}
