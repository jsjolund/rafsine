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
 * @brief A 3D graphics visualization of domain decomposition
 * 
 */
class PartitionMesh : public Topology, public osg::Geode {
 protected:
  ~PartitionMesh();

 public:
  ColorSet *m_colorSet;
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  PartitionMesh(unsigned int latticeSizeX, unsigned int latticeSizeY,
                unsigned int latticeSizeZ, unsigned int subdivisions);
};
