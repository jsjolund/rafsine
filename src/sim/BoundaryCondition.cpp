#include "BoundaryCondition.hpp"

std::ostream &operator<<(std::ostream &os, VoxelType::Enum v)
{
  switch (v)
  {
  case VoxelType::Enum::EMPTY:
    return os << "EMPTY";
  case VoxelType::Enum::FLUID:
    return os << "FLUID";
  case VoxelType::Enum::WALL:
    return os << "WALL";
  case VoxelType::Enum::FREE_SLIP:
    return os << "FREE_SLIP";
  case VoxelType::Enum::INLET_CONSTANT:
    return os << "INLET_CONSTANT";
  case VoxelType::Enum::INLET_ZERO_GRADIENT:
    return os << "INLET_ZERO_GRADIENT";
  case VoxelType::Enum::INLET_RELATIVE:
    return os << "INLET_RELATIVE";
  };
  return os << static_cast<std::uint16_t>(v);
}

std::ostream &operator<<(std::ostream &os, BoundaryCondition bc)
{
  os << "id: " << bc.m_id << std::endl;
  os << "type: " << bc.m_type << std::endl;
  os << "temperature: " << bc.m_temperature << std::endl;
  os << "velocity: " << bc.m_velocity.x << ", " << bc.m_velocity.y << ", " << bc.m_velocity.z << std::endl;
  os << "normal: " << bc.m_normal.x << ", " << bc.m_normal.y << ", " << bc.m_normal.z << std::endl;
  os << "rel_pos: " << bc.m_rel_pos.x << ", " << bc.m_rel_pos.y << ", " << bc.m_rel_pos.z << std::endl;
  return os;
}