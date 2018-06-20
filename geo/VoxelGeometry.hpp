#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include "../sim/BoundaryCondition.hpp"
#include "UnitConverter.hpp"
#include <iostream>

using glm::ivec3;
using glm::vec3;
using std::string;

template <class T>
inline void hash_combine(std::size_t &seed, const T &v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

enum NodeMode
{
  OVERWRITE,
  INTERSECT,
  FILL
};

class NodeUnit
{
public:
  string name;
  NodeMode mode;
  VoxelType typeBC;
  vec3 origin, dir1, dir2, normal, velocity;
  real temperature;

  bool operator==(const NodeUnit &other) const
  {
    return (
        temperature == other.temperature &&
        velocity.x == other.velocity.x &&
        velocity.y == other.velocity.y &&
        velocity.z == other.velocity.z &&
        // origin.x == other.origin.x &&
        // origin.y == other.origin.y &&
        // origin.z == other.origin.z &&
        dir1.x == other.dir1.x &&
        dir1.y == other.dir1.y &&
        dir1.z == other.dir1.z &&
        dir2.x == other.dir2.x &&
        dir2.y == other.dir2.y &&
        dir2.z == other.dir2.z &&
        normal.x == other.normal.x &&
        normal.y == other.normal.y &&
        normal.z == other.normal.z);
  }
};

// namespace std
// {
// template <>
// struct hash<NodeUnit>
// {
//   std::size_t operator()(const NodeUnit &k) const
//   {
//     using std::hash;
//     using std::size_t;

//     // Compute individual hash values for first,
//     // second and third and combine them using XOR
//     // and bit shifting:
//     size_t seed = 0;
//     ::hash_combine(seed, k.typeBC);
//     ::hash_combine(seed, k.temperature);
//     ::hash_combine(seed, k.velocity);
//     ::hash_combine(seed, k.origin.x);
//     ::hash_combine(seed, k.origin.y);
//     ::hash_combine(seed, k.origin.z);
//     ::hash_combine(seed, k.dir1.x);
//     ::hash_combine(seed, k.dir1.y);
//     ::hash_combine(seed, k.dir1.z);
//     ::hash_combine(seed, k.dir2.x);
//     ::hash_combine(seed, k.dir2.y);
//     ::hash_combine(seed, k.dir2.z);
//     ::hash_combine(seed, k.normal.x);
//     ::hash_combine(seed, k.normal.y);
//     ::hash_combine(seed, k.normal.z);
//     return seed;
//   }
// };
// } // namespace std

class VoxelGeometry
{
private:
  int nx, ny, nz;
  int ***data;
  int newtype = 1;
  UnitConverter *uc;
  std::unordered_map<size_t, int> types;

  size_t inline getHash(VoxelType typeBC, ivec3 normal, vec3 velocity,
                        real temperature, real rel_pos)
  {
    using std::hash;
    using std::size_t;
    size_t seed = 0;
    ::hash_combine(seed, typeBC);
    ::hash_combine(seed, normal.x);
    ::hash_combine(seed, normal.y);
    ::hash_combine(seed, normal.z);
    ::hash_combine(seed, velocity.x);
    ::hash_combine(seed, velocity.y);
    ::hash_combine(seed, velocity.z);
    ::hash_combine(seed, temperature);
    ::hash_combine(seed, rel_pos);
    return seed;
  }

public:
  void inline set(int x, int y, int z, int value) { *(&data)[x][y][z] = value; }
  int inline get(int x, int y, int z) { return *(&data)[x][y][z]; }
  int inline get(ivec3 v) { return *(&data)[v.x][v.y][v.z]; }

  // function to get the type from the description
  bool inline getType(VoxelType typeBC, ivec3 normal, vec3 velocity,
                      real temperature, real rel_pos, int &id)
  {
    if (typeBC == FLUID)
    {
      id = FLUID;
      return true;
    }
    else if (typeBC == EMPTY)
    {
      id = EMPTY;
      return true;
    }
    else
    {
      size_t seed = getHash(typeBC, normal, velocity, temperature, rel_pos);
      if (types.find(seed) != types.end())
      {
        id = types.at(seed);
        return true;
      }
    }
    return false;
  }

  // function to set the type from a description
  void inline setType(VoxelType typeBC, ivec3 normal, vec3 velocity,
                      real temperature, real rel_pos, int value)
  {
    size_t seed = getHash(typeBC, normal, velocity, temperature, rel_pos);
    if (types.find(seed) == types.end())
    {
      types[seed] = value;
    }
    else
    {
      std::cout << "seed " << seed << " exists" << std::endl;
    }
  }

  // generate a new type of voxel
  // double link voxel type and description
  int createNewVoxelType(VoxelType typeBC, ivec3 normal, vec3 velocity,
                         real temperature, real rel_pos)
  {
    int id = 0;
    // if this type of BC hasn't appeared yet, create a new one
    if (getType(typeBC, normal, velocity, temperature, rel_pos, id))
    {
      std::cout << "type " << id << " exists" << std::endl;
    }
    else
    {
      id = newtype;
      // attach type to description
      setType(typeBC, normal, velocity, temperature, rel_pos, id);
      // increment the next available type
      newtype++;
    }
    return id;
  }

  // return the correct voxel type for the boundary
  // create a new one if the boundary does not exist already
  int inline getBCVoxelType(VoxelType typeBC, ivec3 normal, vec3 velocity,
                            real temperature, real rel_pos)
  {
    int id = 0;
    if (getType(typeBC, normal, velocity, temperature, rel_pos, id))
    {
      // if the parameters correspond to a type, then use it
      return id;
    }
    else
    {
      // otherwise, create a new type based on the parameters
      return createNewVoxelType(typeBC, normal, velocity, temperature, rel_pos);
    }
  }

  // function to compute a new type for intersection of two types
  // or use one already existing
  int inline getBCIntersectType(ivec3 position, VoxelType typeBC, ivec3 normal, vec3 velocity,
                                real temperature, real rel_pos)
  {
    // type of the existing voxel
    int vox1 = get(position);
    // normal of the exiting voxel
    
    return 0;
  }

  int inline addQuadBCNodeUnits(
      string name,
      ivec3 origin,
      ivec3 dir1,
      ivec3 dir2,
      ivec3 normal,
      VoxelType typeBC,
      NodeMode mode,
      vec3 velocity,
      real temperature,
      real rel_pos)
  {
    int voxtype = getBCVoxelType(typeBC, normal, velocity, temperature, rel_pos);
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
          if (mode == OVERWRITE)
          {
            // overwrite whatever type was there
            set(p.x, p.y, p.z, voxtype);
          }
          else if (mode == INTERSECT)
          {
            // the boundary is intersecting another boundary
            int t = getBCIntersectType();
            set(p.x, p.y, p.z, t);
          }
          else if (mode == FILL)
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

  int inline addQuadBC(NodeUnit node)
  {
    ivec3 origin(0, 0, 0);
    uc->m_to_LUA(node.origin, origin);
    return 0;
  }

  ~VoxelGeometry() { delete data; }
  VoxelGeometry(int nx, int ny, int nz, UnitConverter *uc) : nx(nx), ny(ny), nz(nz), uc(uc)
  {
    data = new int **[nx * ny * nz];
    for (int x = 0; x < nx; x++)
      for (int y = 0; y < ny; y++)
        for (int z = 0; z < nz; z++)
          set(x, y, z, 0);
  }
};