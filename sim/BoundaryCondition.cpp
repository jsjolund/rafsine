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