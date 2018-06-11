#pragma once

#include <glm/glm.hpp>
#include <string>

using glm::ivec3;
using glm::vec3;
using std::string;

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
  // Name of boundary condition
  string name;
  // Type
  BCtype type;
  // Voxel id
  int id;
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