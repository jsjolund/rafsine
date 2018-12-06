#include "PartitionTopology.hpp"

bool operator==(Partition const &a, Partition const &b) {
  return (a.getLatticeMin() == b.getLatticeMin() &&
          a.getLatticeMax() == b.getLatticeMax());
}

std::ostream &operator<<(std::ostream &os, const Partition p) {
  os << "min=" << p.getLatticeMin() << ", max=" << p.getLatticeMax();
  return os;
}

static void primeFactors(int n, std::vector<int> *factors) {
  while (n % 2 == 0) {
    factors->push_back(2);
    n = n / 2;
  }
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    while (n % i == 0) {
      factors->push_back(i);
      n = n / i;
    }
  }
  if (n > 2) factors->push_back(n);
}

static void subdivide(int factor, glm::ivec3 *partitionCount,
                      std::vector<Partition *> *partitions) {
  std::vector<Partition *> oldPartitions;
  oldPartitions.insert(oldPartitions.end(), partitions->begin(),
                       partitions->end());
  partitions->clear();
  const Partition::Enum axis = oldPartitions.at(0)->getDivisionAxis();
  if (axis == Partition::X_AXIS) partitionCount->x *= factor;
  if (axis == Partition::Y_AXIS) partitionCount->y *= factor;
  if (axis == Partition::Z_AXIS) partitionCount->z *= factor;

  for (Partition *partition : oldPartitions) {
    glm::ivec3 min = partition->getLatticeMin(),
               max = partition->getLatticeMax();
    for (int i = 0; i < factor; i++) {
      float d = static_cast<float>(i + 1) / factor;
      switch (axis) {
        case Partition::X_AXIS:
          max.x = partition->getLatticeMin().x +
                  std::ceil(1.0 * partition->getLatticeDims().x * d);
          break;
        case Partition::Y_AXIS:
          max.y = partition->getLatticeMin().y +
                  std::ceil(1.0 * partition->getLatticeDims().y * d);
          break;
        case Partition::Z_AXIS:
          max.z = partition->getLatticeMin().z +
                  std::ceil(1.0 * partition->getLatticeDims().z * d);
          break;
        default:
          break;
      }
      if (i == factor - 1) {
        max.x = partition->getLatticeMax().x;
        max.y = partition->getLatticeMax().y;
        max.z = partition->getLatticeMax().z;
      }
      partitions->push_back(new Partition(min, max));
      switch (axis) {
        case Partition::X_AXIS:
          min.x = max.x;
          break;
        case Partition::Y_AXIS:
          min.y = max.y;
          break;
        case Partition::Z_AXIS:
          min.z = max.z;
          break;
        default:
          break;
      }
    }
  }
  for (Partition *partition : oldPartitions) delete partition;
}

static void createPartitions(unsigned int divisions, glm::ivec3 *partitionCount,
                             std::vector<Partition *> *partitions) {
  if (divisions <= 1) return;
  std::vector<int> factors;
  primeFactors(divisions, &factors);
  std::reverse(factors.begin(), factors.end());
  for (int factor : factors) subdivide(factor, partitionCount, partitions);

  std::sort(partitions->begin(), partitions->end(),
            [](Partition *a, Partition *b) {
              if (a->getLatticeMin().z != b->getLatticeMin().z)
                return a->getLatticeMin().z < b->getLatticeMin().z;
              if (a->getLatticeMin().y != b->getLatticeMin().y)
                return a->getLatticeMin().y < b->getLatticeMin().y;
              return a->getLatticeMin().x < b->getLatticeMin().x;
            });
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
  throw std::out_of_range("Invalid range");
}

Partition Topology::getNeighbour(Partition partition, int dfIdx) {
  glm::ivec3 haloDirection = D3Q27[dfIdx];
  glm::ivec3 partPos = m_partitionPositions[partition];
  return *getPartition(partPos + haloDirection);
}

Topology::Topology(unsigned int Q, unsigned int latticeSizeX,
                   unsigned int latticeSizeY, unsigned int latticeSizeZ,
                   unsigned int divisions)
    : m_partitionCount(glm::ivec3(1, 1, 1)),
      m_latticeSize(glm::ivec3(latticeSizeX, latticeSizeY, latticeSizeZ)),
      m_Q(Q) {
  m_partitions.push_back(new Partition(glm::ivec3(0, 0, 0), m_latticeSize));

  if (divisions > 1)
    createPartitions(divisions, &m_partitionCount, &m_partitions);

  for (int x = 0; x < getNumPartitions().x; x++)
    for (int y = 0; y < getNumPartitions().y; y++)
      for (int z = 0; z < getNumPartitions().z; z++) {
        glm::ivec3 position(x, y, z);
        Partition *partition = getPartition(position);
        m_partitionPositions[*partition] = position;

        for (int i = 0; i < 27; i++) {
          glm::ivec3 haloDirection = D3Q27[i];
          glm::ivec3 neighbourPos = position + haloDirection;
          Partition *neighbour = getPartition(neighbourPos);

          HaloParamsLocal *haloData;

          if (m_haloData.find(*partition) != m_haloData.end() &&
              m_haloData[*partition].find(*neighbour) !=
                  m_haloData[*partition].end()) {
            haloData = m_haloData[*partition][*neighbour];

          } else {
            haloData = new HaloParamsLocal();
          }

          std::vector<glm::ivec3> pSrc, pDst;
          partition->getHalo(haloDirection, &pSrc, nullptr);
          neighbour->getHalo(-haloDirection, nullptr, &pDst);

          for (int j = 0; j < pSrc.size(); j++) {
            glm::ivec3 src = pSrc.at(j);
            glm::ivec3 dst = pDst.at(j);
            haloData->srcIndexH.push_back(
                partition->toLocalIndex(0, src.x, src.y, src.z));
            haloData->dstIndexH.push_back(
                neighbour->toLocalIndex(0, dst.x, dst.y, dst.z));
          }
          // Upload to GPU
          haloData->srcIndexD = haloData->srcIndexH;
          haloData->dstIndexD = haloData->dstIndexH;

          m_haloData[*partition][*neighbour] = haloData;
        }
      }
  // assert(m_haloData[*getPartition(glm::ivec3(0, 0, 0))].size() == m_Q);
  // if (divisions > 1) assert(m_haloData.size() == getNumPartitionsTotal());
}

void Partition::getHalo(glm::ivec3 direction,
                        std::vector<glm::ivec3> *srcPoints,
                        std::vector<glm::ivec3> *haloPoints) {
  glm::ivec3 haloOrigin, dir1, dir2;

  // Origin
  if (direction == glm::ivec3(0, 0, 0)) {
    return;
    // 6 faces
  } else if (direction == glm::ivec3(1, 0, 0)) {
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
