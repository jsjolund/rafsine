#pragma once

#include <glm/glm.hpp>

using glm::ivec3;
using glm::vec3;

enum BCtype
{
  NONE = 0,
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
  // Type of boundary condition
  BCtype type;
  // Temperature
  float temperature;
  // Velocity
  vec3 velocity;
  // Plane normal of this boundary condition
  ivec3 normal;
  // Relative position of temperature condition (in voxel units)
  ivec3 rel_pos;
  // Origin (in m)
  vec3 origin;
  // Extents (in m)
  vec3 extents;
};