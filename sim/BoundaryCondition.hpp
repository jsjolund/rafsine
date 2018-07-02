#pragma once

#include <glm/glm.hpp>
#include <string>

#include "../ext/ordered-map/tsl/ordered_map.h"

#include "../geo/Primitives.hpp"

using std::ostream;
using std::string;

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
