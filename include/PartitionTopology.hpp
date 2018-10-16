#pragma once

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

#include "ColorSet.hpp"
#include "CudaUtils.hpp"
#include "Kernel.hpp"
#include "Primitives.hpp"

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
  enum Enum { X_AXIS, Y_AXIS, Z_AXIS };

  inline Partition(glm::ivec3 min, glm::ivec3 max) : m_min(min), m_max(max) {}
  inline ~Partition() {}

  inline glm::ivec3 getLatticeMin() const { return glm::ivec3(m_min); }
  inline glm::ivec3 getLatticeMax() const { return glm::ivec3(m_max); }
  inline glm::ivec3 getLatticeSize() const { return m_max - m_min; }
  inline int getVolume() {
    glm::ivec3 size = getLatticeSize();
    return size.x * size.y * size.z;
  }

  Partition::Enum getDivisionAxis();

  void subpartition(int divisions, std::vector<Partition> *partitions);

  void getHalo(glm::ivec3 direction, std::vector<glm::ivec3> *srcPoints,
               std::vector<glm::ivec3> *haloPoints);
};
bool operator==(Partition const &a, Partition const &b);
std::ostream &operator<<(std::ostream &os, Partition p);

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
  std::unordered_map<Partition, std::vector<Partition *>> m_neighbours;

  inline std::vector<Partition *> getPartitions() { return m_partitions; }
  inline glm::ivec3 getLatticeSize() const { return glm::ivec3(m_latticeSize); }
  inline glm::ivec3 getNumPartitions() { return glm::ivec3(m_partitionCount); }
  inline int getNumPartitionsTotal() { return m_partitions.size(); }

  Topology(unsigned int latticeSizeX, unsigned int latticeSizeY,
           unsigned int latticeSizeZ, unsigned int subdivisions);

  inline ~Topology() {
    for (Partition *p : m_partitions) delete p;
  }

  Partition *getPartitionContaining(unsigned int x, unsigned int y,
                                    unsigned int z);

  inline Partition *getPartition(unsigned int x, unsigned int y,
                                 unsigned int z) const {
    return (m_partitions.data())[I3D(x, y, z, m_partitionCount.x,
                                     m_partitionCount.y, m_partitionCount.z)];
  }

  inline Partition *getPartition(glm::ivec3 pos) {
    return getPartition(pos.x, pos.y, pos.z);
  }
};
