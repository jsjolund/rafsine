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
        temperature(std::numeric_limits<real>::quiet_NaN()),
        velocity(vec3(0, 0, 0)),
        normal(vec3(0, 0, 0)),
        rel_pos(vec3(0, 0, 0))
  {
  }

  BoundaryCondition(BoundaryCondition *other)
      : type(other->type),
        id(other->id),
        temperature(other->temperature),
        velocity(other->velocity),
        normal(other->normal),
        rel_pos(other->rel_pos)
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

class DomainGeometryQuad
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

  DomainGeometryQuad()
      : name(""), origin(0, 0, 0), dir1(0, 0, 0),
        dir2(0, 0, 0), mode(FILL), bc(new BoundaryCondition())
  {
  }

  DomainGeometryQuad(vec3 origin, vec3 dir1, vec3 dir2,
                     VoxelType type, ivec3 normal,
                     NodeMode mode, string name)
      : name(name), origin(origin), dir1(dir1),
        dir2(dir2), mode(mode)
  {
    bc.type = type;
    bc.normal = normal;
  }

  DomainGeometryQuad(vec3 origin, vec3 dir1, vec3 dir2,
                     VoxelType type, ivec3 normal,
                     NodeMode mode, string name, real temperature)
      : name(name), origin(origin), dir1(dir1),
        dir2(dir2), mode(mode)
  {
    bc.type = type;
    bc.normal = normal;
    bc.temperature = temperature;
  }
};

class DomainGeometryBox
{
public:
  // Name of boundary condition
  string name;
  // Minmax (in m)
  vec3 min;
  vec3 max;
  // NaN for no temperature
  real temperature;
  DomainGeometryBox(string name, vec3 min, vec3 max, real temperature)
      : name(name), min(min), max(max), temperature(temperature)
  {
  }
};
