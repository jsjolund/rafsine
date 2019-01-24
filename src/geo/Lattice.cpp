#include "Lattice.hpp"

SubLattice Lattice::getSubLatticeContaining(unsigned int x, unsigned int y,
                                            unsigned int z) {
  if (x >= m_latticeSize.x || y >= m_latticeSize.y || z >= m_latticeSize.z)
    throw std::out_of_range("Invalid range");
  // Interval tree or similar would scale better...
  int px = 0, py = 0, pz = 0;
  for (int ix = 0; ix < m_subLatticeCount.x; ix++)
    if (x < getSubLattice(ix, 0, 0).getLatticeMax().x) {
      px = ix;
      break;
    }
  for (int iy = 0; iy < m_subLatticeCount.y; iy++)
    if (y < getSubLattice(0, iy, 0).getLatticeMax().y) {
      py = iy;
      break;
    }
  for (int iz = 0; iz < m_subLatticeCount.z; iz++)
    if (z < getSubLattice(0, 0, iz).getLatticeMax().z) {
      pz = iz;
      break;
    }
  return (m_subLattices.data())[I3D(px, py, pz, m_subLatticeCount.x,
                                    m_subLatticeCount.y, m_subLatticeCount.z)];
}

SubLattice Lattice::getNeighbour(SubLattice subLattice, glm::ivec3 direction) {
  glm::ivec3 partPos = m_subLatticePositions[subLattice];
  return getSubLattice(partPos + direction);
}

Lattice::Lattice(unsigned int Q, unsigned int latticeSizeX,
                 unsigned int latticeSizeY, unsigned int latticeSizeZ,
                 unsigned int divisions)
    : m_subLatticeCount(glm::ivec3(1, 1, 1)),
      m_latticeSize(glm::ivec3(latticeSizeX, latticeSizeY, latticeSizeZ)),
      m_Q(Q) {
  SubLattice fullLattice(glm::ivec3(0, 0, 0), m_latticeSize);
  fullLattice.split(divisions, &m_subLatticeCount, &m_subLattices);

  for (int x = 0; x < getNumSubLattices().x; x++)
    for (int y = 0; y < getNumSubLattices().y; y++)
      for (int z = 0; z < getNumSubLattices().z; z++) {
        glm::ivec3 position(x, y, z);
        SubLattice subLattice = getSubLattice(position);
        m_subLatticePositions[subLattice] = position;

        for (int i = 0; i < 27; i++) {
          glm::ivec3 direction = D3Q27[i];
          glm::ivec3 neighbourPos = position + direction;
          SubLattice neighbour = getSubLattice(neighbourPos);
          m_segments[subLattice][neighbour] =
              std::vector<SubLatticeSegment>(27);
        }

        for (int i = 0; i < 27; i++) {
          glm::ivec3 direction = D3Q27[i];
          glm::ivec3 neighbourPos = position + direction;
          SubLattice neighbour = getSubLattice(neighbourPos);
          m_segments[subLattice][neighbour].at(i) =
              subLattice.getSubLatticeSegment(direction, neighbour);
        }
      }
}
