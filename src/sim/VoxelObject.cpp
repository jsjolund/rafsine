#include "VoxelObject.hpp"

bool operator==(VoxelQuad const &a, VoxelQuad const &b) {
  return (a.m_origin.x == b.m_origin.x && a.m_origin.y == b.m_origin.y &&
          a.m_origin.z == b.m_origin.z && a.m_dir1.x == b.m_dir1.x &&
          a.m_dir1.y == b.m_dir1.y && a.m_dir1.z == b.m_dir1.z &&
          a.m_dir2.x == b.m_dir2.x && a.m_dir2.y == b.m_dir2.y &&
          a.m_dir2.z == b.m_dir2.z && a.m_mode == b.m_mode &&
          a.m_name.compare(b.m_name) == 0);
}

bool operator==(VoxelVolume const &a, VoxelVolume const &b) {
  return (a.m_min.x == b.m_min.x && a.m_min.y == b.m_min.y &&
          a.m_min.z == b.m_min.z && a.m_max.x == b.m_max.x &&
          a.m_max.y == b.m_max.y && a.m_max.z == b.m_max.z &&
          a.m_voxMin.x == b.m_voxMin.x && a.m_voxMin.y == b.m_voxMin.y &&
          a.m_voxMin.z == b.m_voxMin.z && a.m_voxMax.x == b.m_voxMax.x &&
          a.m_voxMax.y == b.m_voxMax.y && a.m_voxMax.z == b.m_voxMax.z &&
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
    : VoxelVolume(name, voxMin, voxMax, min, max),
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