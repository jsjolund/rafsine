#include "VoxelObject.hpp"

bool operator==(VoxelQuad const &a, VoxelQuad const &b) {
  return (a.m_origin.x == b.m_origin.x && a.m_origin.y == b.m_origin.y &&
          a.m_origin.z == b.m_origin.z && a.m_dir1.x == b.m_dir1.x &&
          a.m_dir1.y == b.m_dir1.y && a.m_dir1.z == b.m_dir1.z &&
          a.m_dir2.x == b.m_dir2.x && a.m_dir2.y == b.m_dir2.y &&
          a.m_dir2.z == b.m_dir2.z && a.m_mode == b.m_mode &&
          a.m_name.compare(b.m_name) == 0);
}

std::ostream &operator<<(std::ostream &os, NodeMode::Enum v) {
  switch (v) {
    case NodeMode::Enum::OVERWRITE:
      return os << "OVERWRITE";
    case NodeMode::Enum::INTERSECT:
      return os << "INTERSECT";
    case NodeMode::Enum::FILL:
      return os << "FILL";
  }
  return os << static_cast<std::uint16_t>(v);
}

VoxelBox::VoxelBox(std::string name, vec3<int> voxMin, vec3<int> voxMax,
                   vec3<real> min, vec3<real> max, real temperature)
    : VoxelObject(name),
      m_min(min),
      m_max(max),
      m_voxMin(voxMin),
      m_voxMax(voxMax),
      m_temperature(temperature) {
  VoxelType::Enum type = VoxelType::Enum::WALL;
  vec3<real> velocity(NaN, NaN, NaN);
  if (!std::isnan(temperature)) {
    type = VoxelType::Enum::INLET_CONSTANT;
    velocity.x = 0;
    velocity.y = 0;
    velocity.z = 0;
  }
  vec3<int> origin, dir1, dir2, normal;
  VoxelQuad quad;
  NodeMode::Enum mode;

  origin = vec3<int>(voxMin);
  dir1 = vec3<int>(0, voxMax.y - voxMin.y, 0);
  dir2 = vec3<int>(0, 0, voxMax.z - voxMin.z);
  normal = vec3<int>(-1, 0, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = VoxelQuad(m_name, mode, origin, dir1, dir2, normal, type, temperature,
                   velocity);
  m_quads.push_back(quad);

  origin = vec3<int>(voxMax.x, voxMin.y, voxMin.z);
  dir1 = vec3<int>(0, voxMax.y - voxMin.y, 0);
  dir2 = vec3<int>(0, 0, voxMax.z - voxMin.z);
  normal = vec3<int>(1, 0, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = VoxelQuad(m_name, mode, origin, dir1, dir2, normal, type, temperature,
                   velocity);
  m_quads.push_back(quad);

  origin = vec3<int>(voxMin);
  dir1 = vec3<int>(voxMax.x - voxMin.x, 0, 0);
  dir2 = vec3<int>(0, 0, voxMax.z - voxMin.z);
  normal = vec3<int>(0, -1, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = VoxelQuad(m_name, mode, origin, dir1, dir2, normal, type, temperature,
                   velocity);
  m_quads.push_back(quad);

  origin = vec3<int>(voxMin.x, voxMax.y, voxMin.z);
  dir1 = vec3<int>(voxMax.x - voxMin.x, 0, 0);
  dir2 = vec3<int>(0, 0, voxMax.z - voxMin.z);
  normal = vec3<int>(0, 1, 0);
  mode = NodeMode::Enum::INTERSECT;
  quad = VoxelQuad(m_name, mode, origin, dir1, dir2, normal, type, temperature,
                   velocity);
  m_quads.push_back(quad);

  origin = vec3<int>(voxMin);
  dir1 = vec3<int>(voxMax.x - voxMin.x, 0, 0);
  dir2 = vec3<int>(0, voxMax.y - voxMin.y, 0);
  normal = vec3<int>(0, 0, -1);
  mode = NodeMode::Enum::OVERWRITE;
  quad = VoxelQuad(m_name, mode, origin, dir1, dir2, normal, type, temperature,
                   velocity);
  m_quads.push_back(quad);

  origin = vec3<int>(voxMin.x, voxMin.y, voxMax.z);
  dir1 = vec3<int>(voxMax.x - voxMin.x, 0, 0);
  dir2 = vec3<int>(0, voxMax.y - voxMin.y, 0);
  normal = vec3<int>(0, 0, 1);
  mode = NodeMode::Enum::INTERSECT;
  quad = VoxelQuad(m_name, mode, origin, dir1, dir2, normal, type, temperature,
                   velocity);
  m_quads.push_back(quad);
}