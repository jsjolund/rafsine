#include "VoxelGeometry.hpp"

bool operator==(VoxelQuad const &a, VoxelQuad const &b) {
  return (a.m_origin.x == b.m_origin.x && a.m_origin.y == b.m_origin.y &&
          a.m_origin.z == b.m_origin.z && a.m_dir1.x == b.m_dir1.x &&
          a.m_dir1.y == b.m_dir1.y && a.m_dir1.z == b.m_dir1.z &&
          a.m_dir2.x == b.m_dir2.x && a.m_dir2.y == b.m_dir2.y &&
          a.m_dir2.z == b.m_dir2.z && a.m_mode == b.m_mode &&
          a.m_name.compare(b.m_name) == 0);
}

std::ostream &operator<<(std::ostream &os, NodeMode::Enum v) {
  switch (v) {
    case NodeMode::Enum::OVERWRITE:
      return os << "OVERWRITE";
    case NodeMode::Enum::INTERSECT:
      return os << "INTERSECT";
    case NodeMode::Enum::FILL:
      return os << "FILL";
  }
  return os << static_cast<std::uint16_t>(v);
}

VoxelGeometry::VoxelGeometry()
    : m_nx(0), m_ny(0), m_nz(0), m_voxelArray(NULL), m_newtype(1) {
  BoundaryCondition empty;
  m_bcsArray.push_back(empty);
  m_voxelArray = new VoxelArray(0, 0, 0);
}

VoxelGeometry::VoxelGeometry(const int nx, const int ny, const int nz,
                             std::shared_ptr<UnitConverter> uc)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_uc(uc), m_newtype(1) {
  BoundaryCondition empty;
  m_bcsArray.push_back(empty);
  m_voxelArray = new VoxelArray(nx, ny, nz);
}

voxel VoxelGeometry::storeType(BoundaryCondition &bc, std::string &quadName) {
  if (bc.m_type == VoxelType::Enum::FLUID) {
    bc.m_id = VoxelType::Enum::FLUID;
  } else if (bc.m_type == VoxelType::Enum::EMPTY) {
    bc.m_id = VoxelType::Enum::EMPTY;
  } else {
    std::size_t hashKey = std::hash<BoundaryCondition>{}(bc, quadName);
    if (m_types.find(hashKey) == m_types.end()) {
      // Not found, combination of boundary condition and geometry name
      bc.m_id = m_newtype++;
      m_bcsArray.push_back(bc);
      m_types[hashKey] = bc;
    } else {
      // Found combination
      bc.m_id = m_types[hashKey].m_id;
    }
  }
  return bc.m_id;
}

void VoxelGeometry::addQuadBCNodeUnits(VoxelQuad *quad) {
  storeType(quad->m_bc, quad->m_name);

  m_voxNameMap[quad->m_bc.m_id].insert(quad->m_name);

  vec3<int> origin = quad->m_voxOrigin;
  vec3<int> dir1 = quad->m_voxDir1;
  vec3<int> dir2 = quad->m_voxDir2;
  int l1 = static_cast<int>(sqrt(dir1.x * dir1.x) + sqrt(dir1.y * dir1.y) +
                            sqrt(dir1.z * dir1.z));
  int l2 = static_cast<int>(sqrt(dir2.x * dir2.x) + sqrt(dir2.y * dir2.y) +
                            sqrt(dir2.z * dir2.z));
  vec3<int> dir1n = vec3<int>(sgn(dir1.x), sgn(dir1.y), sgn(dir1.z));
  vec3<int> dir2n = vec3<int>(sgn(dir2.x), sgn(dir2.y), sgn(dir2.z));

  for (int i = 0; i <= l1; i++) {
    for (int j = 0; j <= l2; j++) {
      vec3<int> p = origin + i * dir1n + j * dir2n;
      if (get(p) == VoxelType::Enum::EMPTY ||
          get(p) == VoxelType::Enum::FLUID) {
        // Replacing empty voxel
        set(p, quad->m_bc.m_id);
      } else if (quad->m_mode == NodeMode::Enum::OVERWRITE) {
        // Overwrite whatever type was there
        set(p, quad->m_bc.m_id);
      } else if (quad->m_mode == NodeMode::Enum::INTERSECT) {
        // There is a boundary already
        voxel vox1 = get(p);
        BoundaryCondition oldBc = m_bcsArray.at(vox1);
        // normal of the exiting voxel
        vec3<int> n1 = oldBc.m_normal;
        // normal of the new boundary
        vec3<int> n2 = quad->m_bc.m_normal;
        // build a new vector, sum of the two vectors
        vec3<int> n = n1 + n2;
        // if the boundaries are opposite, they cannot be compatible, so
        // overwrite with the new boundary
        if (n1.x == -n2.x && n1.y == -n2.y && n1.z == -n2.z) n = n2;
        // TODO this suppose they have the same boundary type
        if (quad->m_bc.m_type != oldBc.m_type)
          std::cout << "Warning: Intersecting incompatible boundary conditions!"
                    << std::endl;
        BoundaryCondition mergeBc(&quad->m_bc);
        mergeBc.m_normal = n;
        storeType(mergeBc, quad->m_name);
        set(p, mergeBc.m_id);
        m_voxNameMap[mergeBc.m_id].insert(quad->m_name);
      } else if (quad->m_mode == NodeMode::Enum::FILL) {
        // Not empty, do nothing
      }
    }
  }
  m_nameQuadMap[quad->m_name].insert(*quad);
}

void VoxelGeometry::createAddQuadBC(
    std::string name, std::string mode, real originX, real originY,
    real originZ, real dir1X, real dir1Y, real dir1Z, real dir2X, real dir2Y,
    real dir2Z, int normalX, int normalY, int normalZ, std::string typeBC,
    std::string temperatureType, real temperature, real velocityX,
    real velocityY, real velocityZ, real rel_pos) {
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

  VoxelQuad *quad = new VoxelQuad(
      name, modeEnum, vec3<real>(originX, originY, originZ),
      vec3<real>(dir1X, dir1Y, dir1Z), vec3<real>(dir2X, dir2Y, dir2Z),
      vec3<int>(normalX, normalY, normalZ), typeBcEnum, temperature,
      vec3<real>(velocityX, velocityY, velocityZ),
      vec3<int>(relPosX, relPosY, relPosZ));

  addQuadBC(quad);
}

void VoxelGeometry::addQuadBC(VoxelQuad *quad) {
  if (quad->m_name.length() == 0) quad->m_name = DEFAULT_GEOMETRY_NAME;

  vec3<int> origin(0, 0, 0);
  m_uc->m_to_LUA(quad->m_origin, origin);

  vec3<int> dir1(0, 0, 0);
  vec3<real> tmp1(quad->m_origin + quad->m_dir1);
  m_uc->m_to_LUA(tmp1, dir1);
  dir1 = dir1 - origin;

  vec3<int> dir2(0, 0, 0);
  vec3<real> tmp2(quad->m_origin + quad->m_dir2);
  m_uc->m_to_LUA(tmp2, dir2);
  dir2 = dir2 - origin;

  quad->m_voxOrigin = origin;
  quad->m_voxDir1 = dir1;
  quad->m_voxDir2 = dir2;

  addQuadBCNodeUnits(quad);
}

VoxelQuad VoxelGeometry::addWallXmin() {
  vec3<int> n(1, 0, 0);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(0, m_ny - 1, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
  return quad;
}

VoxelQuad VoxelGeometry::addWallXmax() {
  vec3<int> n(-1, 0, 0);
  vec3<int> origin(m_nx, 1, 1);
  vec3<int> dir1(0, m_ny - 1, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
  return quad;
}

VoxelQuad VoxelGeometry::addWallYmin() {
  vec3<int> n(0, 1, 0);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
  return quad;
}

VoxelQuad VoxelGeometry::addWallYmax() {
  vec3<int> n(0, -1, 0);
  vec3<int> origin(1, m_ny, 1);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
  return quad;
}

VoxelQuad VoxelGeometry::addWallZmin() {
  vec3<int> n(0, 0, 1);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, m_ny - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
  return quad;
}

VoxelQuad VoxelGeometry::addWallZmax() {
  vec3<int> n(0, 0, -1);
  vec3<int> origin(1, 1, m_nz);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, m_ny - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  VoxelQuad quad(DEFAULT_GEOMETRY_NAME, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(&quad);
  return quad;
}

void VoxelGeometry::makeHollow(vec3<real> min, vec3<real> max, bool minXface,
                               bool minYface, bool minZface, bool maxXface,
                               bool maxYface, bool maxZface) {
  vec3<int> imin, imax;
  m_uc->m_to_LUA(min, imin);
  m_uc->m_to_LUA(max, imax);
  imin += vec3<int>(1, 1, 1);
  imax -= vec3<int>(1, 1, 1);
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

void VoxelGeometry::makeHollow(real minX, real minY, real minZ, real maxX,
                               real maxY, real maxZ, bool minXface,
                               bool minYface, bool minZface, bool maxXface,
                               bool maxYface, bool maxZface) {
  makeHollow(vec3<real>(minX, minY, minZ), vec3<real>(maxX, maxY, maxZ),
             minXface, minYface, minZface, maxXface, maxYface, maxZface);
}

void VoxelGeometry::createAddSolidBox(std::string name, real minX, real minY,
                                      real minZ, real maxX, real maxY,
                                      real maxZ, real temperature) {
  VoxelBox *box = new VoxelBox(name, vec3<real>(minX, minY, minZ),
                               vec3<real>(maxX, maxY, maxZ), temperature);
  addSolidBox(box);
}

void VoxelGeometry::addSolidBox(VoxelBox *box) {
  if (box->m_name.length() == 0) box->m_name = DEFAULT_GEOMETRY_NAME;

  vec3<real> velocity(NaN, NaN, NaN);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  if (!std::isnan(box->m_temperature)) {
    type = VoxelType::Enum::INLET_CONSTANT;
    velocity.x = 0;
    velocity.y = 0;
    velocity.z = 0;
  }
  vec3<int> min(0, 0, 0);
  vec3<int> max(0, 0, 0);
  m_uc->m_to_LUA(box->m_min, min);
  m_uc->m_to_LUA(box->m_max, max);
  real temperature = box->m_temperature;

  vec3<int> origin, dir1, dir2, normal;
  VoxelQuad *quad;
  NodeMode::Enum mode;

  origin = vec3<int>(min);
  dir1 = vec3<int>(0, max.y - min.y, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(-1, 0, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type,
                       temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(max.x, min.y, min.z);
  dir1 = vec3<int>(0, max.y - min.y, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(1, 0, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type,
                       temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, -1, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type,
                       temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min.x, max.y, min.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, 1, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type,
                       temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, -1);
  mode = NodeMode::Enum::OVERWRITE;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type,
                       temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min.x, min.y, max.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, 1);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type,
                       temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  makeHollow(box->m_min.x, box->m_min.y, box->m_min.z, box->m_max.x,
             box->m_max.y, box->m_max.z, min.x <= 1, min.y <= 1, min.z <= 1,
             max.x >= m_nx, max.y >= m_ny, max.z >= m_nz);
}
