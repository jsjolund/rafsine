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