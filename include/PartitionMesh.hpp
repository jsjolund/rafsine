#pragma once

#include <osg/BlendFunc>
#include <osg/Geode>
#include <osg/Material>
#include <osg/Point>
#include <osg/PolygonMode>
#include <osg/PolygonOffset>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osg/Vec3>

#include "PartitionTopology.hpp"

/**
 * @brief A 3D graphics model of domain decomposition
 *
 */
class PartitionMesh : public Topology, public osg::Geode {
 private:
  ColorSet *m_colorSet;

 protected:
  ~PartitionMesh();

 public:
  /**
   * @brief Construct a new Partition Mesh object
   *
   * @param latticeSizeX Size of the lattice on X-axis
   * @param latticeSizeY Size of the lattice on Y-axis
   * @param latticeSizeZ Size of the lattice on Z-axis
   * @param subdivisions Number of lattice divisions
   */
  PartitionMesh(unsigned int latticeSizeX, unsigned int latticeSizeY,
                unsigned int latticeSizeZ, unsigned int subdivisions);
};
