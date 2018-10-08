#pragma once

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <glm/vec3.hpp>

#include "ColorSet.hpp"
#include "CudaUtils.hpp"
#include "Primitives.hpp"

const glm::ivec3 HALO_DIRECTIONS[26] = {
    // 6 faces
    glm::ivec3(1, 0, 0),
    glm::ivec3(-1, 0, 0),
    glm::ivec3(0, 1, 0),
    glm::ivec3(0, -1, 0),
    glm::ivec3(0, 0, 1),
    glm::ivec3(0, 0, -1),
    // 12 edges
    glm::ivec3(1, 1, 0),
    glm::ivec3(-1, -1, 0),
    glm::ivec3(1, -1, 0),
    glm::ivec3(-1, 1, 0),
    glm::ivec3(1, 0, 1),
    glm::ivec3(-1, 0, -1),
    glm::ivec3(1, 0, -1),
    glm::ivec3(-1, 0, 1),
    glm::ivec3(0, 1, 1),
    glm::ivec3(0, -1, -1),
    glm::ivec3(0, 1, -1),
    glm::ivec3(0, -1, 1),
    // 8 corners
    glm::ivec3(1, 1, 1),
    glm::ivec3(-1, -1, -1),
    glm::ivec3(-1, 1, 1),
    glm::ivec3(1, -1, -1),
    glm::ivec3(1, -1, 1),
    glm::ivec3(-1, 1, -1),
    glm::ivec3(1, 1, -1),
    glm::ivec3(-1, -1, 1),
};

namespace std {
template <>
struct hash<glm::ivec3> {
  std::size_t operator()(const glm::ivec3 &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, p.x);
    ::hash_combine(seed, p.y);
    ::hash_combine(seed, p.z);
    return seed;
  }
};
}  // namespace std

class Partition {
 private:
  glm::ivec3 m_min, m_max;

 public:
  std::unordered_map<glm::ivec3, Partition *> m_neighbours;

  enum Enum { X_AXIS, Y_AXIS, Z_AXIS };

  inline Partition(glm::ivec3 min, glm::ivec3 max) : m_min(min), m_max(max) {}
  inline glm::ivec3 getLatticeMin() const { return glm::ivec3(m_min); }
  inline glm::ivec3 getLatticeMax() const { return glm::ivec3(m_max); }
  inline glm::ivec3 getLatticeSize() const {
    return glm::ivec3(m_max.x - m_min.x, m_max.y - m_min.y, m_max.z - m_min.z);
  }
  inline int getVolume() {
    return getLatticeSize().x * getLatticeSize().y * getLatticeSize().z;
  }

  Partition::Enum getDivisionAxis();

  void subpartition(int divisions, std::vector<Partition> *partitions);

  void getHalo(glm::ivec3 direction, std::vector<glm::ivec3> *srcPoints,
               std::vector<glm::ivec3> *haloPoints);
};
bool operator==(Partition const &a, Partition const &b);

namespace std {
template <>
struct hash<Partition> {
  std::size_t operator()(const Partition &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, p.getLatticeMin().x);
    ::hash_combine(seed, p.getLatticeMin().y);
    ::hash_combine(seed, p.getLatticeMin().z);
    ::hash_combine(seed, p.getLatticeMax().x);
    ::hash_combine(seed, p.getLatticeMax().y);
    ::hash_combine(seed, p.getLatticeMax().z);
    return seed;
  }
};
}  // namespace std

class Topology {
 protected:
  std::vector<Partition *> m_partitions;
  glm::ivec3 m_latticeSize;
  glm::ivec3 m_partitionCount;

 public:
  inline glm::ivec3 getLatticeSize() { return glm::ivec3(m_latticeSize); }
  inline glm::ivec3 getNumPartitions() {
    // return m_partitionCount.x * m_partitionCount.y * m_partitionCount.z;
    return glm::ivec3(m_partitionCount);
  }

  Topology(unsigned int latticeSizeX, unsigned int latticeSizeY,
           unsigned int latticeSizeZ, unsigned int subdivisions);

  inline ~Topology() {
    for (Partition *p : m_partitions) delete p;
  }
  inline Partition *getPartition(unsigned int x, unsigned int y,
                                 unsigned int z) {
    return (m_partitions.data())[I3D(x, y, z, m_partitionCount.x,
                                     m_partitionCount.y, m_partitionCount.z)];
  }
  inline Partition *getPartition(glm::ivec3 pos) {
    return getPartition(pos.x, pos.y, pos.z);
  }
};
