#include "VoxelGeometry.hpp"

bool operator==(VoxelGeometryQuad const &a, VoxelGeometryQuad const &b)
{
  return (a.origin.x == b.origin.x && a.origin.y == b.origin.y && a.origin.z == b.origin.z && a.dir1.x == b.dir1.x && a.dir1.y == b.dir1.y && a.dir1.z == b.dir1.z && a.dir2.x == b.dir2.x && a.dir2.y == b.dir2.y && a.dir2.z == b.dir2.z && a.mode == b.mode && a.name.compare(b.name) == 0);
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

VoxelGeometry::VoxelGeometry()
    : m_nx(0), m_ny(0), m_nz(0), data(NULL)
{
  BoundaryCondition empty;
  voxdetail.push_back(empty);
  data = new VoxelArray(0, 0, 0);
}

VoxelGeometry::VoxelGeometry(const int nx,
                             const int ny,
                             const int nz,
                             std::shared_ptr<UnitConverter> uc)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_uc(uc)
{
  BoundaryCondition empty;
  voxdetail.push_back(empty);
  data = new VoxelArray(nx, ny, nz);
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
      delete data;
      data = new VoxelArray(m_nx, m_ny, m_nz);
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

bool VoxelGeometry::getType(BoundaryCondition *bc, int &id)
{
  if (bc->m_type == VoxelType::Enum::FLUID)
  {
    id = VoxelType::Enum::FLUID;
    return true;
  }
  else if (bc->m_type == VoxelType::Enum::EMPTY)
  {
    id = VoxelType::Enum::EMPTY;
    return true;
  }
  else
  {

    std::size_t hashKey = std::hash<BoundaryCondition>{}(*bc);
    if (types.find(hashKey) != types.end()) // found
    {
      id = types.at(hashKey).m_id;
      return true;
    }
  }
  return false;
}

void VoxelGeometry::setType(BoundaryCondition *bc, int value)
{

  bc->m_id = value;
  std::size_t hashKey = std::hash<BoundaryCondition>{}(*bc);
  if (types.find(hashKey) == types.end())
  {
    voxdetail.push_back(*bc);
    types[hashKey] = *bc;
  }
  else
  {
    std::cout << "hashKey " << hashKey << " id " << bc->m_id << " exists" << std::endl;
  }
}

int VoxelGeometry::createNewVoxelType(BoundaryCondition *bc)
{
  setType(bc, m_newtype);
  // increment the next available type
  m_newtype++;
  return bc->m_id;
}

int VoxelGeometry::getBCVoxelType(BoundaryCondition *bc)
{
  int id = 0;
  if (bc->m_type == VoxelType::INLET_CONSTANT || bc->m_type == VoxelType::INLET_ZERO_GRADIENT || bc->m_type == VoxelType::INLET_RELATIVE)
  {
    return createNewVoxelType(bc);
  }
  else if (getType(bc, id))
  {
    // if the parameters correspond to a type, then use it
    return id;
  }
  else
  {
    // otherwise, create a new type based on the parameters
    return createNewVoxelType(bc);
  }
}

int VoxelGeometry::getBCIntersectType(vec3<int> position, BoundaryCondition *bc)
{
  // type of the existing voxel
  int vox1 = get(position);
  // normal of the exiting voxel
  vec3<int> n1 = voxdetail.at(vox1).m_normal;
  // normal of the new boundary
  vec3<int> n2 = bc->m_normal;
  // build a new vector, sum of the two vectors
  vec3<int> n = n1 + n2;
  // if the boundaries are opposite, they cannot be compatible, so otherwrite with the new boundary
  if (n1.x == -n2.x && n1.y == -n2.y && n1.z == -n2.z)
  {
    n = n2;
  }
  BoundaryCondition *newBc = new BoundaryCondition(bc);
  newBc->m_normal = n;
  return getBCVoxelType(newBc);
}

void VoxelGeometry::addQuadBCNodeUnits(vec3<int> origin,
                                       vec3<int> dir1,
                                       vec3<int> dir2,
                                       VoxelGeometryQuad *geo)
{
  int voxtype = getBCVoxelType(&geo->bc);

  int l1 = int(sqrt(dir1.x * dir1.x) + sqrt(dir1.y * dir1.y) + sqrt(dir1.z * dir1.z));
  int l2 = int(sqrt(dir2.x * dir2.x) + sqrt(dir2.y * dir2.y) + sqrt(dir2.z * dir2.z));
  vec3<int> dir1n = vec3<int>(sgn(dir1.x), sgn(dir1.y), sgn(dir1.z));
  vec3<int> dir2n = vec3<int>(sgn(dir2.x), sgn(dir2.y), sgn(dir2.z));

  for (int i = 0; i <= l1; i++)
  {
    for (int j = 0; j <= l2; j++)
    {
      vec3<int> p = origin + i * dir1n + j * dir2n;
      if (get(p) != VoxelType::Enum::EMPTY && get(p) != VoxelType::Enum::FLUID)
      {
        // There is a boundary already
        if (geo->mode == NodeMode::Enum::OVERWRITE)
        {
          // overwrite whatever type was there
          set(p, voxtype);
          if (quads.count(voxtype) == 0 || quads[voxtype].count(*geo) == 0)
            quads[voxtype].insert(*geo);
        }
        else if (geo->mode == NodeMode::Enum::INTERSECT)
        {
          // the boundary is intersecting another boundary
          int ivoxtype = getBCIntersectType(p, &geo->bc);
          set(p, ivoxtype);
          if (quads.count(ivoxtype) == 0 || quads[ivoxtype].count(*geo) == 0)
            quads[ivoxtype].insert(*geo);
        }
        else if (geo->mode == NodeMode::Enum::FILL)
        {
          // do nothing
        }
      }
      else
      {
        // replacing empty voxel
        set(p, voxtype);
        if (quads.count(voxtype) == 0 || quads[voxtype].count(*geo) == 0)
          quads[voxtype].insert(*geo);
      }
    }
  }
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

  VoxelGeometryQuad *quad = new VoxelGeometryQuad(
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

void VoxelGeometry::addQuadBC(VoxelGeometryQuad *geo)
{
  vec3<int> origin(0, 0, 0);
  m_uc->m_to_LUA(geo->origin, origin);

  vec3<int> dir1(0, 0, 0);
  vec3<real> tmp1(geo->origin + geo->dir1);
  m_uc->m_to_LUA(tmp1, dir1);
  dir1 = dir1 - origin;

  vec3<int> dir2(0, 0, 0);
  vec3<real> tmp2(geo->origin + geo->dir2);
  m_uc->m_to_LUA(tmp2, dir2);
  dir2 = dir2 - origin;

  addQuadBCNodeUnits(origin, dir1, dir2, geo);
}

VoxelGeometryQuad VoxelGeometry::addWallXmin()
{
  vec3<int> n(1, 0, 0);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(0, m_ny - 1, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  std::string name = "xmin";
  VoxelGeometryQuad geo(name, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
  return geo;
}

VoxelGeometryQuad VoxelGeometry::addWallXmax()
{
  vec3<int> n(-1, 0, 0);
  vec3<int> origin(m_nx, 1, 1);
  vec3<int> dir1(0, m_ny - 1, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  std::string name = "xmax";
  VoxelGeometryQuad geo(name, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
  return geo;
}

VoxelGeometryQuad VoxelGeometry::addWallYmin()
{
  vec3<int> n(0, 1, 0);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  std::string name = "ymin";
  VoxelGeometryQuad geo(name, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
  return geo;
}

VoxelGeometryQuad VoxelGeometry::addWallYmax()
{
  vec3<int> n(0, -1, 0);
  vec3<int> origin(1, m_ny, 1);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, 0, m_nz - 1);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  std::string name = "ymax";
  VoxelGeometryQuad geo(name, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
  return geo;
}

VoxelGeometryQuad VoxelGeometry::addWallZmin()
{
  vec3<int> n(0, 0, 1);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, m_ny - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  std::string name = "zmin";
  VoxelGeometryQuad geo(name, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
  return geo;
}

VoxelGeometryQuad VoxelGeometry::addWallZmax()
{
  vec3<int> n(0, 0, -1);
  vec3<int> origin(1, 1, m_nz);
  vec3<int> dir1(m_nx - 1, 0, 0);
  vec3<int> dir2(0, m_ny - 1, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  NodeMode::Enum mode = NodeMode::Enum::INTERSECT;
  std::string name = "zmax";
  VoxelGeometryQuad geo(name, mode, origin, dir1, dir2, n, type);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
  return geo;
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
  VoxelGeometryBox *box =
      new VoxelGeometryBox(name,
                           vec3<real>(minX, minY, minZ),
                           vec3<real>(maxX, maxY, maxZ),
                           temperature);
  addSolidBox(box);
}
void VoxelGeometry::addSolidBox(VoxelGeometryBox *box)
{
  std::stringstream ss;

  vec3<real> velocity(0, 0, 0);
  VoxelType::Enum type = VoxelType::Enum::WALL;
  if (!std::isnan(box->temperature))
  {
    type = VoxelType::Enum::INLET_CONSTANT;
  }
  vec3<int> min(0, 0, 0);
  vec3<int> max(0, 0, 0);
  m_uc->m_to_LUA(box->min, min);
  m_uc->m_to_LUA(box->max, max);
  real t = box->temperature;

  vec3<int> origin, dir1, dir2, normal;
  VoxelGeometryQuad *quad;
  NodeMode::Enum mode;

  origin = vec3<int>(min);
  dir1 = vec3<int>(0, max.y - min.y, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(-1, 0, 0);
  mode = NodeMode::Enum::INTERSECT;
  ss.str("");
  ss << "Quad:xmin";
  quad = new VoxelGeometryQuad(ss.str(), mode, origin, dir1, dir2, normal, type, t);
  addQuadBCNodeUnits(origin, dir1, dir2, quad);
  box->quads.push_back(quad);

  origin = vec3<int>(max.x, min.y, min.z);
  dir1 = vec3<int>(0, max.y - min.y, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(1, 0, 0);
  mode = NodeMode::Enum::INTERSECT;
  ss.str("");
  ss << "Quad:xmax";
  quad = new VoxelGeometryQuad(ss.str(), mode, origin, dir1, dir2, normal, type, t);
  addQuadBCNodeUnits(origin, dir1, dir2, quad);
  box->quads.push_back(quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, -1, 0);
  mode = NodeMode::Enum::INTERSECT;
  ss.str("");
  ss << "Quad:ymin";
  quad = new VoxelGeometryQuad(ss.str(), mode, origin, dir1, dir2, normal, type, t);
  addQuadBCNodeUnits(origin, dir1, dir2, quad);
  box->quads.push_back(quad);

  origin = vec3<int>(min.x, max.y, min.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, 1, 0);
  mode = NodeMode::Enum::INTERSECT;
  ss.str("");
  ss << "Quad:ymax";
  quad = new VoxelGeometryQuad(ss.str(), mode, origin, dir1, dir2, normal, type, t);
  addQuadBCNodeUnits(origin, dir1, dir2, quad);
  box->quads.push_back(quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, -1);
  mode = NodeMode::Enum::OVERWRITE;
  ss.str("");
  ss << "Quad:zmin";
  quad = new VoxelGeometryQuad(ss.str(), mode, origin, dir1, dir2, normal, type, t);
  addQuadBCNodeUnits(origin, dir1, dir2, quad);
  box->quads.push_back(quad);

  origin = vec3<int>(min.x, min.y, max.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, 1);
  mode = NodeMode::Enum::INTERSECT;
  ss.str("");
  ss << "Quad:zmax";
  quad = new VoxelGeometryQuad(ss.str(), mode, origin, dir1, dir2, normal, type, t);
  addQuadBCNodeUnits(origin, dir1, dir2, quad);
  box->quads.push_back(quad);

  makeHollow(box->min.x, box->min.y, box->min.z,
             box->max.x, box->max.y, box->max.z,
             min.x <= 1, min.y <= 1, min.z <= 1,
             max.x >= m_nx, max.y >= m_ny, max.z >= m_nz);
}