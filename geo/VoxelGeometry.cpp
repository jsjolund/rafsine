#include "VoxelGeometry.hpp"

void VoxelGeometry::saveToFile(string filename)
{
  std::cout << nx << " " << ny << " " << nz << std::endl;
  for (int z = 1; z <= nz; z++)
  {
    for (int y = 1; y <= ny; y++)
    {
      for (int x = 1; x <= nx; x++)
      {
        std::cout << std::setw(2) << std::setfill('0') << get(x, y, z) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
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

int VoxelGeometry::getBCIntersectType(ivec3 position, BoundaryCondition *bc)
{
  // type of the existing voxel
  int vox1 = get(position);
  // normal of the exiting voxel
  ivec3 n1 = voxdetail.at(vox1).normal;
  // normal of the new boundary
  ivec3 n2 = bc->normal;
  // build a new vector, sum of the two vectors
  ivec3 n = n1 + n2;
  // if the boundaries are opposite, they cannot be compatible, so otherwrite with the new boundary
  if (n1 == -n2)
  {
    n = n2;
  }
  BoundaryCondition *newBc = new BoundaryCondition(bc);
  newBc->normal = n;
  return getBCVoxelType(newBc);
}

int VoxelGeometry::addQuadBCNodeUnits(ivec3 origin, ivec3 dir1, ivec3 dir2, DomainGeometryQuad *geo)
{
  int voxtype = getBCVoxelType(&geo->bc);
  int l1 = int(sqrt(dir1.x * dir1.x) + sqrt(dir1.y * dir1.y) + sqrt(dir1.z * dir1.z));
  int l2 = int(sqrt(dir2.x * dir2.x) + sqrt(dir2.y * dir2.y) + sqrt(dir2.z * dir2.z));
  ivec3 dir1n = ivec3(sgn(dir1.x), sgn(dir1.y), sgn(dir1.z));
  ivec3 dir2n = ivec3(sgn(dir2.x), sgn(dir2.y), sgn(dir2.z));

  for (int i = 0; i <= l1; i++)
  {
    for (int j = 0; j <= l2; j++)
    {
      ivec3 p = origin + i * dir1n + j * dir2n;
      if (get(p) != EMPTY && get(p) != FLUID)
      {
        // There is a boundary already
        if (geo->mode == OVERWRITE)
        {
          // overwrite whatever type was there
          set(p.x, p.y, p.z, voxtype);
        }
        else if (geo->mode == INTERSECT)
        {
          // the boundary is intersecting another boundary
          int t = getBCIntersectType(p, &geo->bc);
          set(p.x, p.y, p.z, t);
        }
        else if (geo->mode == FILL)
        {
          // do nothing
        }
      }
      else
      {
        // replacing empty voxel
        set(p.x, p.y, p.z, voxtype);
      }
    }
  }
  return voxtype;
}

int VoxelGeometry::addQuadBC(DomainGeometryQuad *geo)
{
  ivec3 origin(0, 0, 0);
  uc->m_to_LUA(geo->origin, origin);

  ivec3 dir1(0, 0, 0);
  vec3 tmp1(geo->origin + geo->dir1);
  uc->m_to_LUA(tmp1, dir1);
  dir1 = dir1 - origin;

  ivec3 dir2(0, 0, 0);
  vec3 tmp2(geo->origin + geo->dir2);
  uc->m_to_LUA(tmp2, dir2);
  dir2 = dir2 - origin;

  return addQuadBCNodeUnits(origin, dir1, dir2, geo);
}

void VoxelGeometry::addWallXmin()
{
  ivec3 n(1, 0, 0);
  vec3 origin(1, 1, 1);
  vec3 dir1(0, ny - 1, 0);
  vec3 dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "xmin";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallXmax()
{
  ivec3 n(-1, 0, 0);
  vec3 origin(nx, 1, 1);
  vec3 dir1(0, ny - 1, 0);
  vec3 dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "xmax";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallYmin()
{
  ivec3 n(0, 1, 0);
  vec3 origin(1, 1, 1);
  vec3 dir1(nx - 1, 0, 0);
  vec3 dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "ymin";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallYmax()
{
  ivec3 n(0, -1, 0);
  vec3 origin(1, ny, 1);
  vec3 dir1(nx - 1, 0, 0);
  vec3 dir2(0, 0, nz - 1);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "ymax";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallZmin()
{
  ivec3 n(0, 0, 1);
  vec3 origin(1, 1, 1);
  vec3 dir1(nx - 1, 0, 0);
  vec3 dir2(0, ny - 1, 0);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "zmin";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

void VoxelGeometry::addWallZmax()
{
  ivec3 n(0, 0, -1);
  vec3 origin(1, 1, nz);
  vec3 dir1(nx - 1, 0, 0);
  vec3 dir2(0, ny - 1, 0);
  VoxelType type = WALL;
  NodeMode mode = INTERSECT;
  string name = "zmax";
  DomainGeometryQuad geo(origin, dir1, dir2, type, n, mode, name);
  addQuadBCNodeUnits(origin, dir1, dir2, &geo);
}

VoxelGeometry::VoxelGeometry(const int nx, const int ny, const int nz, UnitConverter *uc) : nx(nx), ny(ny), nz(nz), uc(uc)
{
  BoundaryCondition empty = new BoundaryCondition();
  voxdetail.push_back(empty);
  data = new int **[nx * ny * nz];
  for (int i = 0; i < nx; i++)
  {
    data[i] = new int *[ny];
    for (int j = 0; j < ny; j++)
    {
      data[i][j] = new int[nz];
      for (int k = 0; k < nz; k++)
      {
        data[i][j][k] = 0;
      }
    }
  }
}

void VoxelGeometry::addSolidBox(DomainGeometryBox *box)
{
  std::stringstream ss;

  vec3 velocity(0, 0, 0);
  VoxelType type = WALL;
  if (!std::isnan(box->temperature))
  {
    type = INLET_CONSTANT;
  }
  ivec3 min(0, 0, 0);
  ivec3 max(0, 0, 0);
  uc->m_to_LUA(box->min, min);
  uc->m_to_LUA(box->max, max);
  real t = box->temperature;

  ivec3 origin, dir1, dir2, normal;
  DomainGeometryQuad quad;

  origin = ivec3(min);
  dir1 = ivec3(max.x - min.x, 0, 0);
  dir2 = ivec3(0, max.y - min.y, 0);
  normal = ivec3(0, 0, -1);
  NodeMode mode = OVERWRITE;
  ss << box->name << " (bottom)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  ss.str("");

  origin = ivec3(min.x, min.y, max.z);
  dir1 = ivec3(max.x - min.x, 0, 0);
  dir2 = ivec3(0, max.y - min.y, 0);
  normal = ivec3(0, 0, 1);
  mode = INTERSECT;
  ss << box->name << " (top)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  ss.str("");

  origin = ivec3(min);
  dir1 = ivec3(0, max.y - min.y, 0);
  dir2 = ivec3(0, 0, max.z - min.z);
  normal = ivec3(-1, 0, 0);
  mode = INTERSECT;
  ss << box->name << " (side x minus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  ss.str("");

  origin = ivec3(max.x, min.y, min.z);
  dir1 = ivec3(0, max.y - min.y, 0);
  dir2 = ivec3(0, 0, max.z - min.z);
  normal = ivec3(1, 0, 0);
  mode = INTERSECT;
  ss << box->name << " (side x plus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  ss.str("");

  origin = ivec3(min);
  dir1 = ivec3(max.x - min.x, 0, 0);
  dir2 = ivec3(0, 0, max.z - min.z);
  normal = ivec3(0, -1, 0);
  mode = INTERSECT;
  ss << box->name << " (side y minus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);

  ss.str("");

  origin = ivec3(min.x, max.y, min.z);
  dir1 = ivec3(max.x - min.x, 0, 0);
  dir2 = ivec3(0, 0, max.z - min.z);
  normal = ivec3(0, 1, 0);
  mode = INTERSECT;
  ss << box->name << " (side y plus)";
  quad = DomainGeometryQuad(origin, dir1, dir2, type, normal, mode, ss.str(), t);
  addQuadBCNodeUnits(origin, dir1, dir2, &quad);
}