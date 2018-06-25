#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "../sim/BoundaryCondition.hpp"
#include "UnitConverter.hpp"
#include <iostream>

using glm::ivec3;
using glm::vec3;
using std::string;

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

class VoxelGeometry
{
private:
  int nx, ny, nz;
  int ***data;
  int newtype = 1;
  UnitConverter *uc;
  std::unordered_map<size_t, BoundaryCondition> types;
  std::vector<BoundaryCondition> voxdetail;

public:
  void inline set(int x, int y, int z, int value)
  {
    // std::cout << "set " << x << " " << y << " " << z << " = " << value << std::endl;
    (data)[x][y][z] = value;
    // std::cout << "wrote " << x << " " << y << " " << z << " = " << (data)[x][y][z] << std::endl;
  }
  int inline get(int x, int y, int z)
  {
    // std::cout << "get " << x << " " << y << " " << z << " = " << (data)[x][y][z] << std::endl;
    return (data)[x][y][z];
  }
  int inline get(ivec3 v)
  {
    return get(v.x, v.y, v.z);
  }

  // function to get the type from the description
  bool inline getType(BoundaryCondition *bc, int &id)
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

  // function to set the type from a description
  void inline setType(BoundaryCondition *bc, int value)
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

  // generate a new type of voxel
  // double link voxel type and description
  int createNewVoxelType(BoundaryCondition *bc)
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

  // return the correct voxel type for the boundary
  // create a new one if the boundary does not exist already
  int inline getBCVoxelType(BoundaryCondition *bc)
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

  // function to compute a new type for intersection of two types
  // or use one already existing
  int inline getBCIntersectType(ivec3 position, BoundaryCondition *bc)
  {
    // type of the existing voxel
    int vox1 = get(position);
    std::cout << "vox " << vox1 << std::endl;
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
    BoundaryCondition *newBc = new BoundaryCondition(bc->type, bc->normal);
    return getBCVoxelType(newBc);
  }

  int inline addQuadBCNodeUnits(ivec3 origin, ivec3 dir1, ivec3 dir2, DomainGeometry *geo)
  {
    int voxtype = getBCVoxelType(&geo->bc);
    int l1 = int(sqrt(dir1.x * dir1.x) + sqrt(dir1.y * dir1.y) + sqrt(dir1.z * dir1.z));
    int l2 = int(sqrt(dir2.x * dir2.x) + sqrt(dir2.y * dir2.y) + sqrt(dir2.z * dir2.z));
    ivec3 dir1n = ivec3(sgn(dir1.x), sgn(dir1.y), sgn(dir1.z));
    ivec3 dir2n = ivec3(sgn(dir2.x), sgn(dir2.y), sgn(dir2.z));

    for (int i = 0; i < l1; i++)
    {
      for (int j = 0; j < l2; j++)
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

  int inline addQuadBC(DomainGeometry *geo)
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

  // Add walls on the domain boundaries
  void addWallXmin()
  {
    vec3 origin(1, 1, 1);
    vec3 dir1(0, ny - 1, 0);
    vec3 dir2(0, 0, nz - 1);
    VoxelType type = WALL;
    ivec3 n(1, 0, 0);
    NodeMode mode = INTERSECT;
    string name = "xmin";
    DomainGeometry geo(origin, dir1, dir2, type, n, mode, name);
    addQuadBCNodeUnits(origin, dir1, dir2, &geo);
  }

  ~VoxelGeometry() { delete data; }

  VoxelGeometry(const int nx, const int ny, const int nz, UnitConverter *uc) : nx(nx), ny(ny), nz(nz), uc(uc)
  {
    std::cout << nx << " " << ny << " " << nz << std::endl;
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
};