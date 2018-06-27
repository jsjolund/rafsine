#include "VoxelGeometry.hpp"

VoxelGeometry::VoxelGeometry()
    : nx(0), ny(0), nz(0), uc(NULL), data(NULL)
{
  BoundaryCondition empty = new BoundaryCondition();
  voxdetail.push_back(empty);
}

VoxelGeometry::VoxelGeometry(const int nx, const int ny, const int nz, UnitConverter *uc)
    : nx(nx), ny(ny), nz(nz), uc(uc)
{
  BoundaryCondition empty = new BoundaryCondition();
  voxdetail.push_back(empty);
  data = new VoxelArray(nx, ny, nz);
  // initVoxData(nx, ny, nz);
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

void VoxelGeometry::saveToFile(string filename)
{
  std::ofstream stream;
  stream.open(filename, std::ofstream::out | std::ofstream::trunc);
  stream << nx << " " << ny << " " << nz << std::endl;
  for (int z = 1; z <= nz; z++)
  {
    for (int y = 1; y <= ny; y++)
    {
      for (int x = 1; x <= nx; x++)
      {
        stream << get(x, y, z) << " ";
      }
      stream << std::endl;
    }
    stream << std::endl;
  }
}

void VoxelGeometry::loadFromFile(string filename)
{
  std::ifstream input(filename);
  string line;
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
      nx = std::stoi(strs.at(0));
      ny = std::stoi(strs.at(1));
      nz = std::stoi(strs.at(2));
      delete data;
      data = new VoxelArray(nx, ny, nz);
      lineNbr++;
    }
    else
    {
      for (int i = 0; i < strs.size() - 1; i++)
      {
        set(1 + i,
            1 + lineNbr % ny,
            1 + floor(lineNbr / ny),
            std::stoi(strs.at(i)));
      }
      lineNbr++;
    }
  }
}

bool VoxelGeometry::getType(BoundaryCondition *bc, int &id)
{
  if (bc->type == FLUID)
  {
    id = FLUID;
    return true;
  }
  else if (bc->type == EMPTY)
  {
    id = EMPTY;
    return true;
  }
  else
  {
    std::size_t seed = std::hash<BoundaryCondition>{}(*bc);
    if (types.find(seed) != types.end())
    {
      id = types.at(seed).id;
      return true;
    }
  }
  return false;
}

void VoxelGeometry::setType(BoundaryCondition *bc, int value)
{
  std::size_t seed = std::hash<BoundaryCondition>{}(*bc);
  if (types.find(seed) == types.end())
  {
    bc->id = value;
    voxdetail.push_back(*bc);
    types[seed] = *bc;
  }
  else
  {
    std::cout << "seed " << seed << " exists" << std::endl;
  }
}

int VoxelGeometry::createNewVoxelType(BoundaryCondition *bc)
{
  int id = 0;
  // if this type of BC hasn't appeared yet, create a new one
  if (getType(bc, id))
  {
    std::cout << "type " << id << " exists" << std::endl;
  }
  else
  {
    id = newtype;
    // attach type to description
    setType(bc, id);
    // increment the next available type
    newtype++;
  }
  return id;
}

int VoxelGeometry::getBCVoxelType(BoundaryCondition *bc)
{
  int id = 0;
  if (getType(bc, id))
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
  vec3<int> n1 = voxdetail.at(vox1).normal;
  // normal of the new boundary
  vec3<int> n2 = bc->normal;
  // build a new vector, sum of the two vectors
  vec3<int> n = n1 + n2;
  // if the boundaries are opposite, they cannot be compatible, so otherwrite with the new boundary
  if (n1.x == -n2.x && n1.y == -n2.y && n1.z == -n2.z)
  {
    n = n2;
  }
  BoundaryCondition *newBc = new BoundaryCondition(bc);
  newBc->normal = n;
  return getBCVoxelType(newBc);
}

int VoxelGeometry::addQuadBCNodeUnits(vec3<int> origin, vec3<int> dir1, vec3<int> dir2, DomainGeometryQuad *geo)
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
      if (get(p) != EMPTY && get(p) != FLUID)
      {
        // There is a boundary already
        if (geo->mode == OVERWRITE)
        {
          // overwrite whatever type was there
          set(p, voxtype);
        }
        else if (geo->mode == INTERSECT)
        {
          // the boundary is intersecting another boundary
          int t = getBCIntersectType(p, &geo->bc);
          set(p, t);
        }
        else if (geo->mode == FILL)
        {
          // do nothing
        }
      }
      else
      {
        // replacing empty voxel
        set(p, voxtype);
      }
    }
  }
  return voxtype;
}

int VoxelGeometry::addQuadBC(DomainGeometryQuad *geo)
{
  vec3<int> origin(0, 0, 0);
  uc->m_to_LUA(geo->origin, origin);

  vec3<int> dir1(0, 0, 0);
  vec3<real> tmp1(geo->origin + geo->dir1);
  uc->m_to_LUA(tmp1, dir1);
  dir1 = dir1 - origin;

  vec3<int> dir2(0, 0, 0);
  vec3<real> tmp2(geo->origin + geo->dir2);
  uc->m_to_LUA(tmp2, dir2);
  dir2 = dir2 - origin;

  return addQuadBCNodeUnits(origin, dir1, dir2, geo);
}

void VoxelGeometry::addWallXmin()
{
  vec3<int> n(1, 0, 0);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(0, ny - 1, 0);
  vec3<int> dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "xmin";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallXmax()
{
  vec3<int> n(-1, 0, 0);
  vec3<int> origin(nx, 1, 1);
  vec3<int> dir1(0, ny - 1, 0);
  vec3<int> dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "xmax";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallYmin()
{
  vec3<int> n(0, 1, 0);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(nx - 1, 0, 0);
  vec3<int> dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "ymin";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallYmax()
{
  vec3<int> n(0, -1, 0);
  vec3<int> origin(1, ny, 1);
  vec3<int> dir1(nx - 1, 0, 0);
  vec3<int> dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "ymax";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallZmin()
{
  vec3<int> n(0, 0, 1);
  vec3<int> origin(1, 1, 1);
  vec3<int> dir1(nx - 1, 0, 0);
  vec3<int> dir2(0, ny - 1, 0);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "zmin";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallZmax()
{
  vec3<int> n(0, 0, -1);
  vec3<int> origin(1, 1, nz);
  vec3<int> dir1(nx - 1, 0, 0);
  vec3<int> dir2(0, ny - 1, 0);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "zmax";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::makeHollow(vec3<real> min, vec3<real> max,
                               bool xmin, bool ymin, bool zmin,
                               bool xmax, bool ymax, bool zmax)
{
  vec3<int> imin, imax;
  uc->m_to_LUA(min, imin);
  uc->m_to_LUA(max, imax);
  imin += vec3<int>(1, 1, 1);
  imax -= vec3<int>(1, 1, 1);
  if (xmin)
    imin.x--;
  if (ymin)
    imin.y--;
  if (zmin)
    imin.z--;
  if (xmax)
    imax.x++;
  if (ymax)
    imax.y++;
  if (zmax)
    imax.z++;
  for (int x = imin.x; x <= imax.x; x++)
    for (int y = imin.y; y <= imax.y; y++)
      for (int z = imin.z; z <= imax.z; z++)
        set(x, y, z, EMPTY);
}

void VoxelGeometry::addSolidBox(DomainGeometryBox *box)
{
  std::stringstream ss;

  vec3<real> velocity(0, 0, 0);
  VoxelType type = WALL;
  if (!std::isnan(box->temperature))
  {
    type = INLET_CONSTANT;
  }
  vec3<int> min(0, 0, 0);
  vec3<int> max(0, 0, 0);
  uc->m_to_LUA(box->min, min);
  uc->m_to_LUA(box->max, max);
  real t = box->temperature;

  vec3<int> origin, dir1, dir2, normal;
  DomainGeometryQuad quad;

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, -1);
  NodeMode mode = OVERWRITE;
  ss << box->name << " (bottom)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  origin = vec3<int>(min.x, min.y, max.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, max.y - min.y, 0);
  normal = vec3<int>(0, 0, 1);
  mode = INTERSECT;
  ss.str("");
  ss << box->name << " (top)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(0, max.y - min.y, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(-1, 0, 0);
  mode = INTERSECT;
  ss.str("");
  ss << box->name << " (side x minus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  origin = vec3<int>(max.x, min.y, min.z);
  dir1 = vec3<int>(0, max.y - min.y, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(1, 0, 0);
  mode = INTERSECT;
  ss.str("");
  ss << box->name << " (side x plus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  origin = vec3<int>(min);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, -1, 0);
  mode = INTERSECT;
  ss.str("");
  ss << box->name << " (side y minus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  origin = vec3<int>(min.x, max.y, min.z);
  dir1 = vec3<int>(max.x - min.x, 0, 0);
  dir2 = vec3<int>(0, 0, max.z - min.z);
  normal = vec3<int>(0, 1, 0);
  mode = INTERSECT;
  ss.str("");
  ss << box->name << " (side y plus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  makeHollow(box->min, box->max,
             min.x <= 1, min.y <= 1, min.z <= 1,
             max.x >= nx, max.y >= ny, max.z >= nz);
}