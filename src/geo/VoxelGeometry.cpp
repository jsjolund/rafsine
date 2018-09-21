#include "VoxelGeometry.hpp"

VoxelGeometry::VoxelGeometry()
    : m_nx(0), m_ny(0), m_nz(0), m_voxelArray(NULL)
{
  BoundaryCondition empty;
  m_bcsArray.push_back(empty);
  m_voxelArray = new VoxelArray(0, 0, 0);
}

VoxelGeometry::VoxelGeometry(const int nx,
                             const int ny,
                             const int nz,
                             std::shared_ptr<UnitConverter> uc)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_uc(uc)
{
  BoundaryCondition empty;
  m_bcsArray.push_back(empty);
  m_voxelArray = new VoxelArray(nx, ny, nz);
}

voxel VoxelGeometry::getType(BoundaryCondition bc, std::string quadName)
{
  voxel id = -1;
  if (bc.m_type == VoxelType::Enum::FLUID)
  {
    id = VoxelType::Enum::FLUID;
  }
  else if (bc.m_type == VoxelType::Enum::EMPTY)
  {
    id = VoxelType::Enum::EMPTY;
  }
  else
  {
    std::size_t hashKey = std::hash<BoundaryCondition>{}(bc, quadName);
    if (m_types.find(hashKey) == m_types.end())
    {
      // Not found, combination of boundary condition and geometry name
      id = m_newtype++;
      m_bcsArray[id] = bc;
      m_types[hashKey] = bc;
    }
    else
    {
      // Found combination
      id = m_types[hashKey].m_id;
    }
  }
  return id;
}

BoundaryCondition VoxelGeometry::getIntersectBC(vec3<int> position, BoundaryCondition newBc)
{
  // type of the existing voxel
  voxel vox1 = get(position);
  BoundaryCondition oldBc = m_bcsArray.at(vox1);
  // normal of the exiting voxel
  vec3<int> n1 = oldBc.m_normal;
  // normal of the new boundary
  vec3<int> n2 = newBc.m_normal;
  // build a new vector, sum of the two vectors
  vec3<int> n = n1 + n2;
  // if the boundaries are opposite, they cannot be compatible, so overwrite with the new boundary
  if (n1.x == -n2.x && n1.y == -n2.y && n1.z == -n2.z)
    n = n2;
  // TODO this suppose they have the same boundary type
  BoundaryCondition mergeBc(&oldBc);
  mergeBc.m_normal = n;
  return mergeBc;
}

void VoxelGeometry::addQuadBCNodeUnits(VoxelQuad *geo)
{
  voxel voxtype = getType(geo->m_bc, geo->m_name);
  geo->m_bc.m_id = voxtype;

  if (m_quads.count(voxtype) == 0 || m_quads[voxtype].count(*geo) == 0)
    m_quads[voxtype].insert(*geo);

  vec3<int> origin = geo->m_voxOrigin;
  vec3<int> dir1 = geo->m_voxDir1;
  vec3<int> dir2 = geo->m_voxDir2;
  int l1 = int(sqrt(dir1.x * dir1.x) + sqrt(dir1.y * dir1.y) + sqrt(dir1.z * dir1.z));
  int l2 = int(sqrt(dir2.x * dir2.x) + sqrt(dir2.y * dir2.y) + sqrt(dir2.z * dir2.z));
  vec3<int> dir1n = vec3<int>(sgn(dir1.x), sgn(dir1.y), sgn(dir1.z));
  vec3<int> dir2n = vec3<int>(sgn(dir2.x), sgn(dir2.y), sgn(dir2.z));

  for (int i = 0; i <= l1; i++)
  {
    for (int j = 0; j <= l2; j++)
    {
      vec3<int> p = origin + i * dir1n + j * dir2n;
      if (get(p) == VoxelType::Enum::EMPTY || get(p) == VoxelType::Enum::FLUID)
      {
        // Replacing empty voxel
        set(p, voxtype);
      }
      else
      {
        // There is a boundary already
        BoundaryCondition mergeBc = getIntersectBC(p, geo->m_bc);
        switch (geo->m_mode)
        {
        case NodeMode::Enum::OVERWRITE:
          // Overwrite whatever type was there
          set(p, voxtype);
          break;

        case NodeMode::Enum::INTERSECT:
          // The boundary is intersecting another boundary
          mergeBc.m_id = getType(mergeBc, geo->m_name);
          set(p, mergeBc.m_id);
          if (m_quads.count(mergeBc.m_id) == 0 || m_quads[mergeBc.m_id].count(*geo) == 0)
            m_quads[mergeBc.m_id].insert(*geo);
          break;

        case NodeMode::Enum::FILL:
          // Not empty, do nothing
        default:
          break;
        }
      }
    }
  }
}

void VoxelGeometry::addQuadBC(VoxelQuad *geo)
{
  if (geo->m_name.length() == 0)
    geo->m_name = DEFAULT_GEOMETRY_NAME;

  vec3<int> origin(0, 0, 0);
  m_uc->m_to_LUA(geo->m_origin, origin);

  vec3<int> dir1(0, 0, 0);
  vec3<real> tmp1(geo->m_origin + geo->m_dir1);
  m_uc->m_to_LUA(tmp1, dir1);
  dir1 = dir1 - origin;

  vec3<int> dir2(0, 0, 0);
  vec3<real> tmp2(geo->m_origin + geo->m_dir2);
  m_uc->m_to_LUA(tmp2, dir2);
  dir2 = dir2 - origin;

  geo->m_voxOrigin = origin;
  geo->m_voxDir1 = dir1;
  geo->m_voxDir2 = dir2;

  addQuadBCNodeUnits(geo);
}

VoxelQuad VoxelGeometry::addWallXmin()
{
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

VoxelQuad VoxelGeometry::addWallXmax()
{
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

VoxelQuad VoxelGeometry::addWallYmin()
{
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

VoxelQuad VoxelGeometry::addWallYmax()
{
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

VoxelQuad VoxelGeometry::addWallZmin()
{
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

VoxelQuad VoxelGeometry::addWallZmax()
{
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

void VoxelGeometry::addSolidBox(VoxelBox *box)
{
  if (box->m_name.length() == 0)
    box->m_name = DEFAULT_GEOMETRY_NAME;

  vec3<real> velocity(NaN, NaN, NaN);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  if (!std::isnan(box->m_temperature))
  {
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
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type, temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(max.x, min.y, min.z);
  dir1 = vec3<int>(0, max.y - min.y, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(1, 0, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type, temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, -1, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type, temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min.x, max.y, min.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, 1, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type, temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, -1);
  mode = NodeMode::Enum::OVERWRITE;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type, temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  origin = vec3<int>(min.x, min.y, max.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, 1);
  mode = NodeMode::Enum::INTERSECT;
  quad = new VoxelQuad(box->m_name, mode, origin, dir1, dir2, normal, type, temperature, velocity);
  addQuadBCNodeUnits(quad);
  box->m_quads.push_back(quad);

  makeHollow(box->m_min.x, box->m_min.y, box->m_min.z,
             box->m_max.x, box->m_max.y, box->m_max.z,
             min.x <= 1, min.y <= 1, min.z <= 1,
             max.x >= m_nx, max.y >= m_ny, max.z >= m_nz);
}

std::ostream &operator<<(std::ostream &str, VoxelGeometry &vox)
{
  str << vox.getNx() << " " << vox.getNy() << " " << vox.getNz() << std::endl;
  for (int z = 1; z <= vox.getNz(); z++)
  {
    for (int y = 1; y <= vox.getNy(); y++)
    {
      for (int x = 1; x <= vox.getNx(); x++)
      {
        str << std::setw(2) << std::setfill('0') << vox.get(x, y, z) << " ";
      }
      str << std::endl;
    }
    str << std::endl;
  }
  return str;
}

void VoxelGeometry::saveToFile(std::string filename)
{
  std::ofstream stream;
  stream.open(filename, std::ofstream::out | std::ofstream::trunc);
  stream << m_nx << " " << m_ny << " " << m_nz << std::endl;
  for (int z = 1; z <= m_nz; z++)
  {
    for (int y = 1; y <= m_ny; y++)
    {
      for (int x = 1; x <= m_nx; x++)
      {
        stream << get(x, y, z) << " ";
      }
      stream << std::endl;
    }
    stream << std::endl;
  }
}

void VoxelGeometry::loadFromFile(std::string filename)
{
  std::ifstream input(filename);
  std::string line;
  int lineNbr = -1;

  while (std::getline(input, line))
  {
    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of("\t "));

    if (strs.size() == 1 && strs.at(0) == "")
    {
      continue;
    }
    else if (lineNbr == -1 && strs.size() == 3)
    {
      m_nx = std::stoi(strs.at(0));
      m_ny = std::stoi(strs.at(1));
      m_nz = std::stoi(strs.at(2));
      delete m_voxelArray;
      m_voxelArray = new VoxelArray(m_nx, m_ny, m_nz);
      lineNbr++;
    }
    else
    {
      for (unsigned int i = 0; i < strs.size() - 1; i++)
      {
        set(1 + i,
            1 + lineNbr % m_ny,
            1 + floor(lineNbr / m_ny),
            std::stoi(strs.at(i)));
      }
      lineNbr++;
    }
  }
}

void VoxelGeometry::makeHollow(vec3<real> min, vec3<real> max,
                               bool minXface, bool minYface, bool minZface,
                               bool maxXface, bool maxYface, bool maxZface)
{
  vec3<int> imin, imax;
  m_uc->m_to_LUA(min, imin);
  m_uc->m_to_LUA(max, imax);
  imin += vec3<int>(1, 1, 1);
  imax -= vec3<int>(1, 1, 1);
  if (minXface)
    imin.x--;
  if (minYface)
    imin.y--;
  if (minZface)
    imin.z--;
  if (maxXface)
    imax.x++;
  if (maxYface)
    imax.y++;
  if (maxZface)
    imax.z++;
  for (int x = imin.x; x <= imax.x; x++)
    for (int y = imin.y; y <= imax.y; y++)
      for (int z = imin.z; z <= imax.z; z++)
        set(x, y, z, VoxelType::Enum::EMPTY);
}

void VoxelGeometry::makeHollow(real minX, real minY, real minZ,
                               real maxX, real maxY, real maxZ,
                               bool minXface, bool minYface, bool minZface,
                               bool maxXface, bool maxYface, bool maxZface)
{
  makeHollow(vec3<real>(minX, minY, minZ),
             vec3<real>(maxX, maxY, maxZ),
             minXface, minYface, minZface,
             maxXface, maxYface, maxZface);
}

void VoxelGeometry::createAddSolidBox(
    std::string name,
    real minX, real minY, real minZ,
    real maxX, real maxY, real maxZ,
    real temperature)
{
  VoxelBox *box = new VoxelBox(name,
                               vec3<real>(minX, minY, minZ),
                               vec3<real>(maxX, maxY, maxZ),
                               temperature);
  addSolidBox(box);
}

void VoxelGeometry::createAddQuadBC(
    std::string name,
    std::string mode,
    real originX, real originY, real originZ,
    real dir1X, real dir1Y, real dir1Z,
    real dir2X, real dir2Y, real dir2Z,
    int normalX, int normalY, int normalZ,
    std::string typeBC,
    std::string temperatureType,
    real temperature,
    real velocityX, real velocityY, real velocityZ,
    real rel_pos)
{
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
  if (typeBC.compare("empty") == 0)
    typeBcEnum = VoxelType::EMPTY;
  else if (typeBC.compare("fluid") == 0)
    typeBcEnum = VoxelType::FLUID;
  else if (typeBC.compare("wall") == 0)
    typeBcEnum = VoxelType::WALL;
  else if (typeBC.compare("freeSlip") == 0)
    typeBcEnum = VoxelType::FREE_SLIP;
  else if (typeBC.compare("inlet") == 0)
    if (temperatureType.compare("constant") == 0)
      typeBcEnum = VoxelType::INLET_CONSTANT;
    else if (temperatureType.compare("zeroGradient") == 0)
      typeBcEnum = VoxelType::INLET_ZERO_GRADIENT;
    else if (temperatureType.compare("relative") == 0)
      typeBcEnum = VoxelType::INLET_RELATIVE;
    else
      throw std::runtime_error(ErrorFormat() << temperatureType << " is unknown temperature type");
  else
    throw std::runtime_error(ErrorFormat() << typeBC << " is unknown boundary condition");

  int relPosX = -(1 + m_uc->m_to_lu(rel_pos)) * normalX;
  int relPosY = -(1 + m_uc->m_to_lu(rel_pos)) * normalY;
  int relPosZ = -(1 + m_uc->m_to_lu(rel_pos)) * normalZ;

  VoxelQuad *quad = new VoxelQuad(
      name,
      modeEnum,
      vec3<real>(originX, originY, originZ),
      vec3<real>(dir1X, dir1Y, dir1Z),
      vec3<real>(dir2X, dir2Y, dir2Z),
      vec3<int>(normalX, normalY, normalZ),
      typeBcEnum,
      temperature,
      vec3<real>(velocityX, velocityY, velocityZ),
      vec3<int>(relPosX, relPosY, relPosZ));

  addQuadBC(quad);
}

bool operator==(VoxelQuad const &a, VoxelQuad const &b)
{
  return (a.m_origin.x == b.m_origin.x && a.m_origin.y == b.m_origin.y && a.m_origin.z == b.m_origin.z && a.m_dir1.x == b.m_dir1.x &&
          a.m_dir1.y == b.m_dir1.y && a.m_dir1.z == b.m_dir1.z && a.m_dir2.x == b.m_dir2.x && a.m_dir2.y == b.m_dir2.y && a.m_dir2.z == b.m_dir2.z && a.m_mode == b.m_mode && a.m_name.compare(b.m_name) == 0);
}

std::ostream &operator<<(std::ostream &os, NodeMode::Enum v)
{
  switch (v)
  {
  case NodeMode::Enum::OVERWRITE:
    return os << "OVERWRITE";
  case NodeMode::Enum::INTERSECT:
    return os << "INTERSECT";
  case NodeMode::Enum::FILL:
    return os << "FILL";
  };
  return os << static_cast<std::uint16_t>(v);
}
