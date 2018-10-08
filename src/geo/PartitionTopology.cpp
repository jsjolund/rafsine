#include "PartitionTopology.hpp"

bool operator==(Partition const &a, Partition const &b) {
  return (a.getLatticeMin() == b.getLatticeMin() &&
          a.getLatticeMax() == b.getLatticeMax());
}

static void recursiveSubpartition(int divisions, glm::ivec3 *partitionCount,
                                  std::vector<Partition *> *partitions) {
  if (divisions > 0) {
    std::vector<Partition *> oldPartitions;
    oldPartitions.insert(oldPartitions.end(), partitions->begin(),
                         partitions->end());
    partitions->clear();
    const Partition::Enum axis = oldPartitions.at(0)->getDivisionAxis();
    if (axis == Partition::X_AXIS) partitionCount->x *= 2;
    if (axis == Partition::Y_AXIS) partitionCount->y *= 2;
    if (axis == Partition::Z_AXIS) partitionCount->z *= 2;

    for (Partition *partition : oldPartitions) {
      glm::ivec3 a_min = partition->getLatticeMin(),
                 a_max = partition->getLatticeMax(),
                 b_min = partition->getLatticeMin(),
                 b_max = partition->getLatticeMax();
      switch (axis) {
        case Partition::X_AXIS:
          a_max.x = partition->getLatticeMin().x +
                    std::ceil(1.0 * partition->getLatticeSize().x / 2);
          b_min.x = a_max.x;
          break;
        case Partition::Y_AXIS:
          a_max.y = partition->getLatticeMin().y +
                    std::ceil(1.0 * partition->getLatticeSize().y / 2);
          b_min.y = a_max.y;
          break;
        case Partition::Z_AXIS:
          a_max.z = partition->getLatticeMin().z +
                    std::ceil(1.0 * partition->getLatticeSize().z / 2);
          b_min.z = a_max.z;
          break;
        default:
          break;
      }
      partitions->push_back(new Partition(a_min, a_max));
      partitions->push_back(new Partition(b_min, b_max));
    }
    for (Partition *partition : oldPartitions) delete partition;
    recursiveSubpartition(divisions - 1, partitionCount, partitions);
  }
}

Partition::Enum Partition::getDivisionAxis() {
  int nx = getLatticeSize().x, ny = getLatticeSize().y, nz = getLatticeSize().z;
  int xz = nx * nz, yz = ny * nz, xy = nx * ny;
  if (xy <= xz && xy <= yz) return Partition::Z_AXIS;
  if (xz <= yz && xz <= xy)
    return Partition::Y_AXIS;
  else
    return Partition::X_AXIS;
}

Topology::Topology(unsigned int latticeSizeX, unsigned int latticeSizeY,
                   unsigned int latticeSizeZ, unsigned int subdivisions)
    : m_partitionCount(glm::ivec3(1, 1, 1)),
      m_latticeSize(glm::ivec3(latticeSizeX, latticeSizeY, latticeSizeZ)) {
  m_partitions.push_back(new Partition(glm::ivec3(0, 0, 0), m_latticeSize));
  if (subdivisions > 0)
    recursiveSubpartition(subdivisions, &m_partitionCount, &m_partitions);

  std::sort(m_partitions.begin(), m_partitions.end(),
            [](Partition *a, Partition *b) {
              if (a->getLatticeMin().z != b->getLatticeMin().z)
                return a->getLatticeMin().z < b->getLatticeMin().z;
              if (a->getLatticeMin().y != b->getLatticeMin().y)
                return a->getLatticeMin().y < b->getLatticeMin().y;
              return a->getLatticeMin().x < b->getLatticeMin().x;
            });

  for (int x = 0; x < getNumPartitions().x; x++)
    for (int y = 0; y < getNumPartitions().y; y++)
      for (int z = 0; z < getNumPartitions().z; z++) {
        glm::ivec3 position(x, y, z);
        Partition *partition = getPartition(position);
        for (glm::ivec3 haloDirection : HALO_DIRECTIONS) {
          glm::ivec3 neighbourPos = position + haloDirection;
          // Periodic
          neighbourPos.x =
              (neighbourPos.x == getNumPartitions().x) ? 0 : neighbourPos.x;
          neighbourPos.x = (neighbourPos.x == -1) ? getNumPartitions().x - 1
                                                  : neighbourPos.x;
          neighbourPos.y =
              (neighbourPos.y == getNumPartitions().y) ? 0 : neighbourPos.y;
          neighbourPos.y = (neighbourPos.y == -1) ? getNumPartitions().y - 1
                                                  : neighbourPos.y;
          neighbourPos.z =
              (neighbourPos.z == getNumPartitions().z) ? 0 : neighbourPos.z;
          neighbourPos.z = (neighbourPos.z == -1) ? getNumPartitions().z - 1
                                                  : neighbourPos.z;
          Partition *neighbour = getPartition(neighbourPos);
          partition->m_neighbours[haloDirection] = neighbour;
        }
      }
}

void Partition::getHalo(glm::ivec3 direction,
                        std::vector<glm::ivec3> *srcPoints,
                        std::vector<glm::ivec3> *haloPoints) {
  glm::ivec3 haloOrigin, dir1, dir2;

  // 6 faces
  if (direction == glm::ivec3(1, 0, 0)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(0, m_max.y - m_min.y, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(-1, 0, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(0, m_max.y - m_min.y, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(0, 1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(0, -1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(0, 0, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(0, 0, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);

    // 12 edges
  } else if (direction == glm::ivec3(1, 1, 0)) {
    haloOrigin = glm::ivec3(m_max.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(-1, -1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(1, -1, 0)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_min.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(-1, 1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(1, 0, 1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(-1, 0, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(1, 0, -1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(-1, 0, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(0, 1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_max.z);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  } else if (direction == glm::ivec3(0, -1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(0, -1, -1);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  } else if (direction == glm::ivec3(0, 1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  } else if (direction == glm::ivec3(0, -1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);

    // 8 corners
  } else if (direction == glm::ivec3(1, 1, 1)) {
    haloOrigin = glm::ivec3(m_max.x, m_max.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, -1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, -1, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, 1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_max.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(1, -1, -1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_min.z) + glm::ivec3(0, -1, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(1, -1, 1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_max.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, 1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z) + glm::ivec3(-1, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(1, 1, -1)) {
    haloOrigin = glm::ivec3(m_max.x, m_max.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, -1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z) + glm::ivec3(-1, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else {
    throw std::out_of_range("Unknown halo direction vector");
  }
  int n1 = abs(dir1.x) + abs(dir1.y) + abs(dir1.z);
  int n2 = abs(dir2.x) + abs(dir2.y) + abs(dir2.z);
  glm::ivec3 e1 = dir1 / n1;
  glm::ivec3 e2 = dir2 / n2;
  for (int i1 = 0; i1 < n1; i1++) {
    for (int i2 = 0; i2 < n2; i2++) {
      glm::ivec3 halo = haloOrigin + e1 * i1 + e2 * i2;
      haloPoints->push_back(halo);
      glm::ivec3 src = haloOrigin - direction + e1 * i1 + e2 * i2;
      srcPoints->push_back(src);
    }
  }
}
