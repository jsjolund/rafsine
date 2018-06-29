#pragma once

#include <glm/glm.hpp>
#include <string>

#include "../ext/ordered-map/tsl/ordered_map.h"

#include "../geo/Primitives.hpp"

using std::ostream;
using std::string;

typedef float real;
typedef tsl::ordered_map<string, string> UserConstants;

#define NaN std::numeric_limits<real>::quiet_NaN()

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
  vec3<real> velocity;
  // Plane normal of this boundary condition
  vec3<int> normal;
  // Relative position of temperature condition (in voxel units)
  vec3<int> rel_pos;

  BoundaryCondition()
      : type(EMPTY),
        id(0),
        temperature(NaN),
        velocity(vec3<real>(0, 0, 0)),
        normal(vec3<int>(0, 0, 0)),
        rel_pos(vec3<int>(0, 0, 0))
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

class VoxelGeometryObject
{
public:
  string name;
  VoxelGeometryObject(string name_) : name(name_){};
};

class VoxelGeometryGroup
{
  string name;
  std::vector<VoxelGeometryObject> objs;
};

class VoxelGeometryQuad : public VoxelGeometryObject
{
public:
  // Origin (in m)
  vec3<real> origin;
  // Extents (in m)
  vec3<real> dir1;
  vec3<real> dir2;
  // Mode
  NodeMode mode;
  // BC
  BoundaryCondition bc;

  VoxelGeometryQuad()
      : VoxelGeometryObject(std::string()), origin(0, 0, 0), dir1(0, 0, 0),
        dir2(0, 0, 0), mode(FILL), bc(new BoundaryCondition()) {}

  VoxelGeometryQuad(vec3<real> origin, vec3<real> dir1, vec3<real> dir2,
                    VoxelType type, vec3<int> normal,
                    NodeMode mode, string name)
      : VoxelGeometryObject(name), origin(origin), dir1(dir1),
        dir2(dir2), mode(mode), bc(new BoundaryCondition())
  {
    bc.type = type;
    bc.normal = normal;
  }

  VoxelGeometryQuad(vec3<real> origin, vec3<real> dir1, vec3<real> dir2,
                    VoxelType type, vec3<int> normal,
                    NodeMode mode, string name,
                    real temperature)
      : VoxelGeometryObject(name), origin(origin), dir1(dir1),
        dir2(dir2), mode(mode), bc(new BoundaryCondition())
  {
    bc.type = type;
    bc.normal = normal;
    bc.temperature = temperature;
  }
};

class VoxelGeometryBox : public VoxelGeometryObject
{
public:
  // Minmax (in m)
  vec3<real> min;
  vec3<real> max;
  // NaN for no temperature
  real temperature;
  VoxelGeometryBox(string name, vec3<real> min, vec3<real> max, real temperature)
      : VoxelGeometryObject(name), min(min), max(max), temperature(temperature)
  {
  }
  VoxelGeometryBox(string name, vec3<real> min, vec3<real> max)
      : VoxelGeometryObject(name), min(min), max(max), temperature(NaN)
  {
  }
};
