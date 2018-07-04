#pragma once

#include <glm/glm.hpp>
#include <string>

#include "../ext/ordered-map/tsl/ordered_map.h"
#include "../geo/Primitives.hpp"

namespace VoxelType
{
enum Enum
{
  EMPTY = -1,
  FLUID = 0,
  WALL = 1,
  FREE_SLIP = 2,
  INLET_CONSTANT = 3,
  INLET_ZERO_GRADIENT = 4,
  INLET_RELATIVE = 5
};
}

std::ostream &operator<<(std::ostream &os, VoxelType::Enum v);

class BoundaryCondition
{
public:
  // Voxel id
  int id;
  // Type
  VoxelType::Enum type;
  // Temperature
  real temperature;
  // Velocity
  vec3<real> velocity;
  // Plane normal of this boundary condition
  vec3<int> normal;
  // Relative position of temperature condition (in voxel units)
  vec3<int> rel_pos;

  BoundaryCondition()
      : id(0),
        type(VoxelType::Enum::EMPTY),
        temperature(NaN),
        velocity(vec3<real>(0, 0, 0)),
        normal(vec3<int>(0, 0, 0)),
        rel_pos(vec3<int>(0, 0, 0))
  {
  }

  explicit BoundaryCondition(BoundaryCondition *other)
      : id(other->id),
        type(other->type),
        temperature(other->temperature),
        velocity(other->velocity),
        normal(other->normal),
        rel_pos(other->rel_pos)
  {
  }

  BoundaryCondition(int id,
                    VoxelType::Enum type,
                    real temperature,
                    vec3<real> velocity,
                    vec3<int> normal,
                    vec3<int> rel_pos)
      : id(id),
        type(type),
        temperature(temperature),
        velocity(velocity),
        normal(normal),
        rel_pos(rel_pos)
  {
  }
};

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
