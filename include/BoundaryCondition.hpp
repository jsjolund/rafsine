#pragma once

#include <string>
#include <vector>

#include "Primitives.hpp"

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

class BoundaryCondition
{
public:
  // Voxel id
  int m_id;
  // Type
  VoxelType::Enum m_type;
  // Temperature
  real m_temperature;
  // Velocity
  vec3<real> m_velocity;
  // Plane normal of this boundary condition
  vec3<int> m_normal;
  // Relative position of temperature condition (in voxel units)
  vec3<int> m_rel_pos;

  BoundaryCondition()
      : m_id(0),
        m_type(VoxelType::Enum::FLUID),
        m_temperature(NaN),
        m_velocity(vec3<real>(NaN, NaN, NaN)),
        m_normal(vec3<int>(0, 0, 0)),
        m_rel_pos(vec3<int>(0, 0, 0))
  {
  }

  explicit BoundaryCondition(BoundaryCondition *other)
      : m_id(other->m_id),
        m_type(other->m_type),
        m_temperature(other->m_temperature),
        m_velocity(other->m_velocity),
        m_normal(other->m_normal),
        m_rel_pos(other->m_rel_pos)
  {
  }

  BoundaryCondition(int id,
                    VoxelType::Enum type,
                    real temperature,
                    vec3<real> velocity,
                    vec3<int> normal,
                    vec3<int> rel_pos)
      : m_id(id),
        m_type(type),
        m_temperature(temperature),
        m_velocity(velocity),
        m_normal(normal),
        m_rel_pos(rel_pos)
  {
  }
};

std::ostream &operator<<(std::ostream &os, VoxelType::Enum v);
std::ostream &operator<<(std::ostream &os, BoundaryCondition bc);
bool operator==(BoundaryCondition const &a, BoundaryCondition const &b);
typedef std::vector<BoundaryCondition> BoundaryConditionsArray;

namespace std
{
template <>
struct hash<BoundaryCondition>
{
  std::size_t operator()(const BoundaryCondition &bc, bool unique = false) const
  {
    using std::hash;
    std::size_t seed = 0;

    ::hash_combine(seed, bc.m_type);
    ::hash_combine(seed, bc.m_normal.x);
    ::hash_combine(seed, bc.m_normal.y);
    ::hash_combine(seed, bc.m_normal.z);

    if (!std::isnan(bc.m_velocity.x) && !std::isnan(bc.m_velocity.y) && !std::isnan(bc.m_velocity.z))
    {
      ::hash_combine(seed, bc.m_velocity.x);
      ::hash_combine(seed, bc.m_velocity.y);
      ::hash_combine(seed, bc.m_velocity.z);

      if (!std::isnan(bc.m_temperature) || bc.m_type == VoxelType::INLET_CONSTANT || bc.m_type == VoxelType::INLET_ZERO_GRADIENT || bc.m_type == VoxelType::INLET_RELATIVE)
      {
        ::hash_combine(seed, bc.m_temperature);
        ::hash_combine(seed, bc.m_rel_pos.x);
        ::hash_combine(seed, bc.m_rel_pos.y);
        ::hash_combine(seed, bc.m_rel_pos.z);
      }
    }
    if (unique)
      ::hash_combine(seed, bc.m_id);
    return seed;
  }
};
} // namespace std
