#include "Lattice.hpp"

Partition Lattice::getPartitionContaining(unsigned int x, unsigned int y,
                                          unsigned int z) const {
  if (x >= m_latticeSize.x() || y >= m_latticeSize.y() ||
      z >= m_latticeSize.z())
    throw std::out_of_range("Invalid range");
  int px = 0, py = 0, pz = 0;
  for (int ix = 0; ix < m_partitionCount.x(); ix++)
    if (x < getPartition(ix, 0, 0).getMax().x()) {
      px = ix;
      break;
    }
  for (int iy = 0; iy < m_partitionCount.y(); iy++)
    if (y < getPartition(0, iy, 0).getMax().y()) {
      py = iy;
      break;
    }
  for (int iz = 0; iz < m_partitionCount.z(); iz++)
    if (z < getPartition(0, 0, iz).getMax().z()) {
      pz = iz;
      break;
    }
  return getPartition(px, py, pz);
}

Lattice::Lattice(const unsigned int nx, const unsigned int ny,
                 const unsigned int nz, const unsigned int nd,
                 const unsigned int ghostLayerSize,
                 const D3Q4::Enum partitioning)
    : m_partitioning(partitioning),
      m_latticeSize(nx, ny, nz),
      m_partitions(),
      m_partitionCount(1, 1, 1),
      m_partitionPositions(),
      m_segments() {
  Partition fullLattice(vector3<int>(0, 0, 0), m_latticeSize,
                        vector3<int>(0, 0, 0));
  fullLattice.split(&m_partitions, &m_partitionCount, nd, ghostLayerSize,
                    partitioning);

  for (int x = 0; x < getNumPartitions().x(); x++)
    for (int y = 0; y < getNumPartitions().y(); y++)
      for (int z = 0; z < getNumPartitions().z(); z++) {
        vector3<int> position(x, y, z);
        Partition partition = getPartition(position);
        m_partitionPositions[partition] = position;

        if (ghostLayerSize > 0) {
          for (int i = 0; i < 27; i++) {
            vector3<int> direction = D3Q27vectors[i];
            vector3<int> neighbourPos = position + direction;
            Partition neighbour = getPartition(neighbourPos);
            m_segments[partition][neighbour] =
                std::vector<GhostLayerParameters>(27);
          }

          for (int i = 0; i < 27; i++) {
            vector3<int> direction = D3Q27vectors[i];
            vector3<int> neighbourPos = position + direction;
            Partition neighbour = getPartition(neighbourPos);
            m_segments[partition][neighbour].at(i) =
                partition.getGhostLayer(direction, neighbour);
          }
        }
      }
}
