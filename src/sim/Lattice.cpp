#include "Lattice.hpp"

Partition Lattice::getPartitionContaining(unsigned int x, unsigned int y,
                                          unsigned int z) const {
  if (x >= m_latticeSize.x() || y >= m_latticeSize.y() ||
      z >= m_latticeSize.z())
    throw std::out_of_range("Invalid range");
  // Interval tree or similar would scale better...
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
  return (m_partitions.data())[I3D(px, py, pz, m_partitionCount.x(),
                                   m_partitionCount.y(), m_partitionCount.z())];
}

Lattice::Lattice(unsigned int latticeSizeX, unsigned int latticeSizeY,
                 unsigned int latticeSizeZ, unsigned int divisions,
                 unsigned int ghostLayerSize)
    : m_partitionCount(1, 1, 1),
      m_latticeSize(latticeSizeX, latticeSizeY, latticeSizeZ) {
  Partition fullLattice(Eigen::Vector3i(0, 0, 0), m_latticeSize,
                        Eigen::Vector3i(0, 0, 0));
  fullLattice.split(divisions, &m_partitionCount, &m_partitions,
                    ghostLayerSize);

  for (int x = 0; x < getNumPartitions().x(); x++)
    for (int y = 0; y < getNumPartitions().y(); y++)
      for (int z = 0; z < getNumPartitions().z(); z++) {
        Eigen::Vector3i position(x, y, z);
        Partition partition = getPartition(position);
        m_partitionPositions[partition] = position;

        if (ghostLayerSize > 0) {
          for (int i = 0; i < 27; i++) {
            Eigen::Vector3i direction = D3Q27[i];
            Eigen::Vector3i neighbourPos = position + direction;
            Partition neighbour = getPartition(neighbourPos);
            m_segments[partition][neighbour] =
                std::vector<GhostLayerParameters>(27);
          }

          for (int i = 0; i < 27; i++) {
            Eigen::Vector3i direction = D3Q27[i];
            Eigen::Vector3i neighbourPos = position + direction;
            Partition neighbour = getPartition(neighbourPos);
            m_segments[partition][neighbour].at(i) =
                partition.getGhostLayer(direction, neighbour);
          }
        }
      }
}
