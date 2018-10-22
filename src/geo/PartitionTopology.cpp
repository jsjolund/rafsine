#include "PartitionTopology.hpp"

bool operator==(Partition const &a, Partition const &b) {
  return (a.getLatticeMin() == b.getLatticeMin() &&
          a.getLatticeMax() == b.getLatticeMax());
}

std::ostream &operator<<(std::ostream &os, const Partition p) {
  os << "min=" << p.getLatticeMin() << ", max=" << p.getLatticeMax();
  return os;
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
                    std::ceil(1.0 * partition->getLatticeDims().x / 2);
          b_min.x = a_max.x;
          break;
        case Partition::Y_AXIS:
          a_max.y = partition->getLatticeMin().y +
                    std::ceil(1.0 * partition->getLatticeDims().y / 2);
          b_min.y = a_max.y;
          break;
        case Partition::Z_AXIS:
          a_max.z = partition->getLatticeMin().z +
                    std::ceil(1.0 * partition->getLatticeDims().z / 2);
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
  int nx = getLatticeDims().x, ny = getLatticeDims().y, nz = getLatticeDims().z;
  int xz = nx * nz, yz = ny * nz, xy = nx * ny;
  if (xy <= xz && xy <= yz)
    return Partition::Z_AXIS;
  else if (xz <= yz && xz <= xy)
    return Partition::Y_AXIS;
  else
    return Partition::X_AXIS;
}

int Partition::toLocalIndex(unsigned int df_idx, int x, int y, int z) {
  glm::ivec3 p(x, y, z);
  glm::ivec3 min = getLatticeMin() - glm::ivec3(1, 1, 1);
  glm::ivec3 max = getLatticeMax() + glm::ivec3(1, 1, 1);
  if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
      p.y < max.y && p.z < max.z) {
    p = p - getLatticeMin() + glm::ivec3(1, 1, 1);
    glm::ivec3 n = getLatticeDims() + glm::ivec3(2, 2, 2);
    int idx = I4D(df_idx, p.x, p.y, p.z, n.x, n.y, n.z);
    return idx;
  }
  return -1;
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

        for (glm::ivec3 haloDirection : D3Q19directionVectors) {
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
          HaloExchangeData data;
          data.neighbour = getPartition(neighbourPos);
          data.srcIndex = std::vector<int>();
          data.dstIndex = std::vector<int>();

          std::vector<glm::ivec3> pSrc, pDst;
          partition->getHalo(haloDirection, &pSrc, nullptr);
          data.neighbour->getHalo(-haloDirection, nullptr, &pDst);

          for (int j = 0; j < pSrc.size(); j++) {
            glm::ivec3 src = pSrc.at(j);
            glm::ivec3 dst = pDst.at(j);
            data.srcIndex.push_back(
                partition->toLocalIndex(0, src.x, src.y, src.z));
            data.dstIndex.push_back(
                data.neighbour->toLocalIndex(0, dst.x, dst.y, dst.z));
          }
          m_haloData[*partition].push_back(data);
        }
      }
}

Partition *Topology::getPartitionContaining(unsigned int x, unsigned int y,
                                            unsigned int z) {
  if (x >= m_latticeSize.x || y >= m_latticeSize.y || z >= m_latticeSize.z)
    throw std::out_of_range("Invalid range");
  // Interval tree or similar would scale better...
  int px = 0, py = 0, pz = 0;
  for (int ix = 0; ix < m_partitionCount.x; ix++)
    if (x < getPartition(ix, 0, 0)->getLatticeMax().x) {
      px = ix;
      break;
    }
  for (int iy = 0; iy < m_partitionCount.y; iy++)
    if (y < getPartition(0, iy, 0)->getLatticeMax().y) {
      py = iy;
      break;
    }
  for (int iz = 0; iz < m_partitionCount.z; iz++)
    if (z < getPartition(0, 0, iz)->getLatticeMax().z) {
      pz = iz;
      break;
    }
  return (m_partitions.data())[I3D(px, py, pz, m_partitionCount.x,
                                   m_partitionCount.y, m_partitionCount.z)];
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
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(0, m_max.y - m_min.y, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(0, 1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(0, -1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(0, 0, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(0, 0, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);

    // 12 edges
  } else if (direction == glm::ivec3(1, 1, 0)) {
    haloOrigin = glm::ivec3(m_max.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(-1, -1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(1, -1, 0)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(-1, 1, 0)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  } else if (direction == glm::ivec3(1, 0, 1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(-1, 0, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(1, 0, -1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(-1, 0, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  } else if (direction == glm::ivec3(0, 1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_max.z);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  } else if (direction == glm::ivec3(0, -1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  } else if (direction == glm::ivec3(0, 1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  } else if (direction == glm::ivec3(0, -1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);

    // 8 corners
  } else if (direction == glm::ivec3(1, 1, 1)) {
    haloOrigin = glm::ivec3(m_max.x, m_max.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, -1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, 1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(1, -1, -1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(1, -1, 1)) {
    haloOrigin = glm::ivec3(m_max.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, 1, -1)) {
    haloOrigin = glm::ivec3(m_min.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(1, 1, -1)) {
    haloOrigin = glm::ivec3(m_max.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else if (direction == glm::ivec3(-1, -1, 1)) {
    haloOrigin = glm::ivec3(m_min.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  } else {
    throw std::out_of_range("Unknown halo direction vector");
  }
  haloOrigin = haloOrigin + glm::ivec3(min(direction.x, 0), min(direction.y, 0),
                                       min(direction.z, 0));
  int n1 = abs(dir1.x) + abs(dir1.y) + abs(dir1.z);
  int n2 = abs(dir2.x) + abs(dir2.y) + abs(dir2.z);
  glm::ivec3 e1 = dir1 / n1;
  glm::ivec3 e2 = dir2 / n2;
  for (int i1 = 0; i1 < n1; i1++) {
    for (int i2 = 0; i2 < n2; i2++) {
      glm::ivec3 halo = haloOrigin + e1 * i1 + e2 * i2;
      if (haloPoints) haloPoints->push_back(halo);
      glm::ivec3 src = haloOrigin - direction + e1 * i1 + e2 * i2;
      if (srcPoints) srcPoints->push_back(src);
    }
  }
}
