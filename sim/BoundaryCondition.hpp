#pragma once

#include <glm/glm.hpp>
#include <string>

using glm::ivec3;
using glm::vec3;
using std::string;

typedef float real;

enum VoxelType
{
  EMPTY = -1,
  FLUID = 0,
  WALL = 1,
  FREE_SLIP = 2,
  INLET_CONSTANT = 3,
  INLET_ZERO_GRADIENT = 4,
  INLET_RELATIVE = 5
};

enum NodeMode
{
  OVERWRITE,
  INTERSECT,
  FILL
};

class BoundaryCondition
{
public:
  // Voxel id
  int id;
  // Type
  VoxelType type;
  // Temperature
  real temperature;
  // Velocity
  vec3 velocity;
  // Plane normal of this boundary condition
  ivec3 normal;
  // Relative position of temperature condition (in voxel units)
  ivec3 rel_pos;

  BoundaryCondition()
      : type(EMPTY),
        id(0),
        temperature(0),
        velocity(vec3(0, 0, 0)),
        normal(vec3(0, 0, 0)),
        rel_pos(vec3(0, 0, 0))
  {
  }
  BoundaryCondition(VoxelType type, ivec3 normal)
      : type(type),
        id(0),
        temperature(0),
        velocity(vec3(0, 0, 0)),
        normal(normal),
        rel_pos(vec3(0, 0, 0))
  {
  }
};

template <class T>
inline void hash_combine(std::size_t &seed, const T &v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
namespace std
{
template <>
struct hash<BoundaryCondition>
{
  std::size_t operator()(const BoundaryCondition &k) const
  {
    using std::hash;
    using std::size_t;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:
    size_t seed = 0;
    ::hash_combine(seed, k.type);
    ::hash_combine(seed, k.normal.x);
    ::hash_combine(seed, k.normal.y);
    ::hash_combine(seed, k.normal.z);
    ::hash_combine(seed, k.velocity.x);
    ::hash_combine(seed, k.velocity.y);
    ::hash_combine(seed, k.velocity.z);
    ::hash_combine(seed, k.temperature);
    ::hash_combine(seed, k.rel_pos.x);
    ::hash_combine(seed, k.rel_pos.y);
    ::hash_combine(seed, k.rel_pos.z);
    return seed;
  }
};
} // namespace std

class DomainGeometry
{
public:
  // Name of boundary condition
  string name;
  // Origin (in m)
  vec3 origin;
  // Extents (in m)
  vec3 dir1;
  vec3 dir2;
  // Mode
  NodeMode mode;
  // BC
  BoundaryCondition bc;

  DomainGeometry(vec3 origin, vec3 dir1, vec3 dir2,
                 VoxelType type, ivec3 normal,
                 NodeMode mode, string name)
      : name(name), origin(origin), dir1(dir1),
        dir2(dir2), mode(mode), bc(type, normal)
  {
  }
};
