#include "BoundaryCondition.hpp"

std::ostream& operator<<(std::ostream& os, VoxelType::Enum v) {
  switch (v) {
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
  }
  return os << static_cast<std::uint16_t>(v);
}

std::ostream& operator<<(std::ostream& os, BoundaryCondition bc) {
  os << "ID: " << bc.m_id << std::endl;
  os << "Type: " << bc.m_type << std::endl;
  os << "Temp: " << bc.m_temperature << std::endl;
  os << "Velocity: " << bc.m_velocity.x() << ", " << bc.m_velocity.y() << ", "
     << bc.m_velocity.z() << std::endl;
  os << "Normal: " << bc.m_normal.x() << ", " << bc.m_normal.y() << ", "
     << bc.m_normal.z() << std::endl;
  os << "Rel.pos: " << bc.m_rel_pos.x() << ", " << bc.m_rel_pos.y() << ", "
     << bc.m_rel_pos.z();
  return os;
}

bool operator==(BoundaryCondition const& a, BoundaryCondition const& b) {
  return (
      a.m_id == b.m_id && a.m_type == b.m_type &&
      a.m_temperature == b.m_temperature &&
      a.m_velocity.x() == b.m_velocity.x() &&
      a.m_velocity.y() == b.m_velocity.y() &&
      a.m_velocity.z() == b.m_velocity.z() &&
      a.m_normal.x() == b.m_normal.x() && a.m_normal.y() == b.m_normal.y() &&
      a.m_normal.z() == b.m_normal.z() && a.m_rel_pos.x() == b.m_rel_pos.x() &&
      a.m_rel_pos.y() == b.m_rel_pos.y() && a.m_rel_pos.z() == b.m_rel_pos.z());
}
