#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <assert.h>

#include <glm/vec3.hpp>

#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/Vec3>

#include "ColorSet.hpp"

class Partition
{
private:
  glm::ivec3 m_min, m_max;

public:
  enum Enum
  {
    X_AXIS,
    Y_AXIS,
    Z_AXIS
  };

  inline Partition(glm::ivec3 min, glm::ivec3 max) : m_min(min), m_max(max){};
  inline glm::ivec3 getMin() { return glm::ivec3(m_min); }
  inline glm::ivec3 getMax() { return glm::ivec3(m_max); }
  inline int getNx() { return m_max.x - m_min.x; }
  inline int getNy() { return m_max.y - m_min.y; }
  inline int getNz() { return m_max.z - m_min.z; }
  inline int getVolume() { return getNx() * getNy() * getNz(); }

  Partition::Enum getDivisionAxis();

  void subpartition(int divisions, std::vector<Partition> *partitions);
};

class Topology
{
private:
  std::vector<Partition *> m_partitions;

  glm::ivec3 m_latticeSize;
  glm::ivec3 m_partitionCount;
  ColorSet *m_colorSet;

  void buildMesh();

public:
  osg::ref_ptr<osg::Group> m_root;

  inline int getLatticeX() { return m_partitionCount.x; }
  inline int getNx() { return m_partitionCount.x; }
  inline int getNy() { return m_partitionCount.y; }
  inline int getNz() { return m_partitionCount.z; }
  inline int size() { return m_partitionCount.x * m_partitionCount.y * m_partitionCount.z; }

  Topology(int latticeSizeX, int latticeSizeY, int latticeSizeZ, int subdivisions);
  inline ~Topology() { delete m_colorSet; };

  inline Partition *&operator()(unsigned int x, unsigned int y, unsigned int z)
  {
    return (m_partitions.data())[x + y * m_partitionCount.x + z * m_partitionCount.x * m_partitionCount.y];
  }
};