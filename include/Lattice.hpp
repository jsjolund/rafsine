#pragma once

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "Primitives.hpp"
#include "SubLattice.hpp"

class Lattice {
 protected:
  //! The size of the entire lattice
  glm::ivec3 m_latticeSize;
  //! A list of sublattices representing domain decomposition
  std::vector<SubLattice> m_subLattices;
  //! The number of sublattices in three dimensions
  glm::ivec3 m_subLatticeCount;
  //! Maps sublattices to their positions in domain decomposition
  std::unordered_map<SubLattice, glm::ivec3> m_subLatticePositions;
  //! Maps the halo exchange parameters between two adjacent sublattices
  std::unordered_map<
      SubLattice,
      std::unordered_map<SubLattice, std::vector<SubLatticeSegment>>>
      m_segments;

 public:
  /**
   * @brief Get the neighbouring sublattice in a certain direction.
   *
   * @param subLattice
   * @param direction
   * @return SubLattice
   */
  inline SubLattice getNeighbour(SubLattice subLattice, glm::ivec3 direction) {
    glm::ivec3 partPos = m_subLatticePositions[subLattice];
    return getSubLattice(partPos + direction);
  }
  inline SubLattice getNeighbour(SubLattice subLattice, D3Q7::Enum direction) {
    return getNeighbour(subLattice, D3Q27[direction]);
  }

  SubLatticeSegment getSubLatticeSegment(SubLattice subLattice,
                                         SubLattice neighbour,
                                         D3Q7::Enum direction) {
    return m_segments[subLattice][neighbour].at(direction);
  }

  inline std::vector<SubLattice> getSubLattices() const {
    return m_subLattices;
  }
  inline glm::ivec3 getDims() const { return m_latticeSize; }
  inline size_t getSize() const {
    return m_latticeSize.x * m_latticeSize.y * m_latticeSize.z;
  }
  inline glm::ivec3 getNumSubLattices() const { return m_subLatticeCount; }
  inline int getNumSubLatticesTotal() const { return m_subLattices.size(); }

  Lattice(unsigned int latticeSizeX, unsigned int latticeSizeY,
          unsigned int latticeSizeZ, unsigned int subdivisions = 1,
          unsigned int haloSize = 0);

  SubLattice getSubLatticeContaining(unsigned int x, unsigned int y,
                                     unsigned int z) const;

  inline SubLattice getSubLattice(int x = 0, int y = 0, int z = 0) const {
    // Periodic
    x = x % m_subLatticeCount.x;
    y = y % m_subLatticeCount.y;
    z = z % m_subLatticeCount.z;
    x = (x < 0) ? m_subLatticeCount.x + x : x;
    y = (y < 0) ? m_subLatticeCount.y + y : y;
    z = (z < 0) ? m_subLatticeCount.z + z : z;
    return (
        m_subLattices.data())[I3D(x, y, z, m_subLatticeCount.x,
                                  m_subLatticeCount.y, m_subLatticeCount.z)];
  }

  inline SubLattice getSubLattice(glm::ivec3 pos) const {
    return getSubLattice(pos.x, pos.y, pos.z);
  }
};
