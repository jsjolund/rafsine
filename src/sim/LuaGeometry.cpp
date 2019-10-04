#include "LuaGeometry.hpp"

void LuaGeometry::addQuadBCNodeUnits(VoxelQuad *quad) {
  // std::cout << "adding quad:" << std::endl << *quad << std::endl;
  storeType(&(quad->m_bc), quad->m_name);

  m_voxNameMap[quad->m_bc.m_id].insert(quad->m_name);

  glm::ivec3 origin = quad->m_voxOrigin;
  glm::ivec3 dir1 = quad->m_voxDir1;
  glm::ivec3 dir2 = quad->m_voxDir2;
  int l1 = static_cast<int>(sqrt(dir1.x * dir1.x) + sqrt(dir1.y * dir1.y) +
                            sqrt(dir1.z * dir1.z));
  int l2 = static_cast<int>(sqrt(dir2.x * dir2.x) + sqrt(dir2.y * dir2.y) +
                            sqrt(dir2.z * dir2.z));
  glm::ivec3 dir1n = glm::ivec3(sgn(dir1.x), sgn(dir1.y), sgn(dir1.z));
  glm::ivec3 dir2n = glm::ivec3(sgn(dir2.x), sgn(dir2.y), sgn(dir2.z));

  int intersecting = 0;

  for (int i = 0; i <= l1; i++) {
    for (int j = 0; j <= l2; j++) {
      glm::ivec3 p = origin + i * dir1n + j * dir2n;
      if (get(p) == VoxelType::Enum::EMPTY ||
          get(p) == VoxelType::Enum::FLUID) {
        // Replacing empty voxel
        set(p, quad->m_bc.m_id);
      } else if (quad->m_mode == NodeMode::Enum::OVERWRITE) {
        // Overwrite whatever type was there
        set(p, quad->m_bc.m_id);
      } else if (quad->m_mode == NodeMode::Enum::INTERSECT) {
        // There is a boundary already
        voxel_t vox1 = get(p);
        BoundaryCondition oldBc = m_bcsArray->at(vox1);
        // normal of the exiting voxel
        glm::ivec3 n1 = oldBc.m_normal;
        // normal of the new boundary
        glm::ivec3 n2 = quad->m_bc.m_normal;
        // build a new vector, sum of the two vectors
        glm::ivec3 n = n1 + n2;
        // if the boundaries are opposite, they cannot be compatible, so
        // overwrite with the new boundary
        if (n1.x == -n2.x && n1.y == -n2.y && n1.z == -n2.z) n = n2;
        // TODO(this suppose they have the same boundary type)
        if (quad->m_bc.m_type != oldBc.m_type) intersecting++;

        BoundaryCondition mergeBc(&quad->m_bc);
        mergeBc.m_normal = n;
        storeType(&mergeBc, quad->m_name);
        set(p, mergeBc.m_id);
        m_voxNameMap[mergeBc.m_id].insert(quad->m_name);
      } else if (quad->m_mode == NodeMode::Enum::FILL) {
        // Not empty, do nothing
      }
    }
  }
  if (intersecting > 0)
    std::cout << "Warning: Intersecting incompatible boundary conditions ("
              << intersecting << " voxels) in geometry '" << quad->m_name
              << "'!" << std::endl;
  m_nameQuadMap[quad->m_name].insert(*quad);
}

void LuaGeometry::addQuadBC(std::string name, std::string mode, real originX,
                            real originY, real originZ, real dir1X, real dir1Y,
                            real dir1Z, real dir2X, real dir2Y, real dir2Z,
                            int normalX, int normalY, int normalZ,
                            std::string typeBC, std::string temperatureType,
                            real temperature, real velocityX, real velocityY,
                            real velocityZ, real rel_pos) {
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
  int relPosX = 0;
  int relPosY = 0;
  int relPosZ = 0;

  if (!std::isnan(rel_pos)) {
    relPosX = -(1 + m_uc->m_to_lu(rel_pos)) * normalX;
    relPosY = -(1 + m_uc->m_to_lu(rel_pos)) * normalY;
    relPosZ = -(1 + m_uc->m_to_lu(rel_pos)) * normalZ;
  }
  if (name.length() == 0) name = DEFAULT_GEOMETRY_NAME;

  glm::vec3 origin(originX, originY, originZ);
  glm::vec3 dir1(dir1X, dir1Y, dir1Z);
  glm::vec3 dir2(dir2X, dir2Y, dir2Z);
  glm::ivec3 normal(normalX, normalY, normalZ);
  glm::ivec3 voxOrigin = m_uc->m_to_LUA_vec(origin);
  glm::ivec3 voxDir1 = m_uc->m_to_LUA_vec(origin + dir1) - voxOrigin;
  glm::ivec3 voxDir2 = m_uc->m_to_LUA_vec(origin + dir2) - voxOrigin;

  VoxelQuad quad(name, modeEnum, voxOrigin, voxDir1, voxDir2, normal,
                 typeBcEnum, temperature,
                 glm::vec3(velocityX, velocityY, velocityZ),
                 glm::ivec3(relPosX, relPosY, relPosZ), origin, dir1, dir2);

  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addSensor(std::string name, real minX, real minY, real minZ,
                            real maxX, real maxY, real maxZ) {
  glm::vec3 min(minX, minY, minZ);
  glm::vec3 max(maxX, maxY, maxZ);
  glm::ivec3 voxMin = m_uc->m_to_lu_vec(min);
  glm::ivec3 voxMax = m_uc->m_to_lu_vec(max);
  if (voxMax.x == voxMin.x) voxMax.x += 1;
  if (voxMax.y == voxMin.y) voxMax.y += 1;
  if (voxMax.z == voxMin.z) voxMax.z += 1;
  VoxelVolume sensorArea(name, voxMin, voxMax, min, max);
  m_sensorArray->push_back(sensorArea);
}

void LuaGeometry::addWallXmin() {
  glm::ivec3 n(1, 0, 0);
  glm::ivec3 origin(1, 1, 1);
  glm::ivec3 dir1(0, m_ny - 1, 0);
  glm::ivec3 dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallXmax() {
  glm::ivec3 n(-1, 0, 0);
  glm::ivec3 origin(m_nx, 1, 1);
  glm::ivec3 dir1(0, m_ny - 1, 0);
  glm::ivec3 dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallYmin() {
  glm::ivec3 n(0, 1, 0);
  glm::ivec3 origin(1, 1, 1);
  glm::ivec3 dir1(m_nx - 1, 0, 0);
  glm::ivec3 dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallYmax() {
  glm::ivec3 n(0, -1, 0);
  glm::ivec3 origin(1, m_ny, 1);
  glm::ivec3 dir1(m_nx - 1, 0, 0);
  glm::ivec3 dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallZmin() {
  glm::ivec3 n(0, 0, 1);
  glm::ivec3 origin(1, 1, 1);
  glm::ivec3 dir1(m_nx - 1, 0, 0);
  glm::ivec3 dir2(0, m_ny - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::addWallZmax() {
  glm::ivec3 n(0, 0, -1);
  glm::ivec3 origin(1, 1, m_nz);
  glm::ivec3 dir1(m_nx - 1, 0, 0);
  glm::ivec3 dir2(0, m_ny - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
}

void LuaGeometry::makeHollow(glm::vec3 min, glm::vec3 max, bool minXface,
                             bool minYface, bool minZface, bool maxXface,
                             bool maxYface, bool maxZface) {
  glm::ivec3 imin = m_uc->m_to_LUA_vec(min);
  glm::ivec3 imax = m_uc->m_to_LUA_vec(max);
  imin += glm::ivec3(1, 1, 1);
  imax -= glm::ivec3(1, 1, 1);
  if (minXface) imin.x--;
  if (minYface) imin.y--;
  if (minZface) imin.z--;
  if (maxXface) imax.x++;
  if (maxYface) imax.y++;
  if (maxZface) imax.z++;
  for (int x = imin.x; x <= imax.x; x++)
    for (int y = imin.y; y <= imax.y; y++)
      for (int z = imin.z; z <= imax.z; z++)
        set(x, y, z, VoxelType::Enum::EMPTY);
}

void LuaGeometry::makeHollow(real minX, real minY, real minZ, real maxX,
                             real maxY, real maxZ, bool minXface, bool minYface,
                             bool minZface, bool maxXface, bool maxYface,
                             bool maxZface) {
  makeHollow(glm::vec3(minX, minY, minZ), glm::vec3(maxX, maxY, maxZ), minXface,
             minYface, minZface, maxXface, maxYface, maxZface);
}

void LuaGeometry::addSolidBox(std::string name, real minX, real minY, real minZ,
                              real maxX, real maxY, real maxZ,
                              real temperature) {
  if (name.length() == 0) name = DEFAULT_GEOMETRY_NAME;

  glm::vec3 min(minX, minY, minZ);
  glm::vec3 max(maxX, maxY, maxZ);
  glm::ivec3 voxMin = m_uc->m_to_LUA_vec(min);
  glm::ivec3 voxMax = m_uc->m_to_LUA_vec(max);
  VoxelBox box(name, voxMin, voxMax, min, max, temperature);
  for (int i = 0; i < box.m_quads.size(); i++)
    addQuadBCNodeUnits(&(box.m_quads.at(i)));

  makeHollow(box.m_min.x, box.m_min.y, box.m_min.z, box.m_max.x, box.m_max.y,
             box.m_max.z, min.x <= 1, min.y <= 1, min.z <= 1, max.x >= m_nx,
             max.y >= m_ny, max.z >= m_nz);
}
