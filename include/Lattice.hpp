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
  std::vector<SubLattice> m_subLattices;

  glm::ivec3 m_latticeSize;
  glm::ivec3 m_subLatticeCount;
  // Number of arrays (or directions for distribution functions)
  const unsigned int m_Q;

  std::unordered_map<SubLattice, glm::ivec3> m_subLatticePositions;

 public:
  std::unordered_map<
      SubLattice,
      std::unordered_map<SubLattice, std::vector<SubLatticeSegment>>>
      m_segments;

  SubLattice getNeighbour(SubLattice subLattice, glm::ivec3 direction);

  inline SubLattice getNeighbour(SubLattice subLattice, D3Q7::Enum direction) {
    return getNeighbour(subLattice, D3Q27[direction]);
  }

  inline std::vector<SubLattice> getSubLattices() { return m_subLattices; }
  inline glm::ivec3 getLatticeDims() const { return glm::ivec3(m_latticeSize); }
  inline size_t getLatticeSize() const {
    return m_latticeSize.x * m_latticeSize.y * m_latticeSize.z;
  }
  inline glm::ivec3 getNumSubLattices() {
    return glm::ivec3(m_subLatticeCount);
  }
  inline int getNumSubLatticesTotal() { return m_subLattices.size(); }

  /**
   * @brief Return the number of arrays in the group i.e. the number of
   * distribution functions
   *
   * @return unsigned int
   */
  unsigned int getQ() const { return m_Q; }

  Lattice(unsigned int Q, unsigned int latticeSizeX, unsigned int latticeSizeY,
          unsigned int latticeSizeZ, unsigned int subdivisions = 0);

  inline ~Lattice() {
    // for (SubLattice p : m_subLattices) delete p;
  }

  SubLattice getSubLatticeContaining(unsigned int x, unsigned int y,
                                     unsigned int z);

  inline SubLattice getSubLattice(int x, int y, int z) const {
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

  inline SubLattice getSubLattice(glm::ivec3 pos) {
    return getSubLattice(pos.x, pos.y, pos.z);
  }
};
