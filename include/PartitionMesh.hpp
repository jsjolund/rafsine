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

#include <sstream>

#include "BillboardText.hpp"
#include "ColorSet.hpp"
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
  void addLabel(osg::Vec3d center, std::string content);
  void setTransparent(osg::ref_ptr<osg::ShapeDrawable> drawable);

 public:
  /**
   * @brief Construct a new Partition Mesh object
   *
   * @param latticeSizeX Size of the lattice on X-axis
   * @param latticeSizeY Size of the lattice on Y-axis
   * @param latticeSizeZ Size of the lattice on Z-axis
   * @param partitions Number of lattice partitions
   */
  PartitionMesh(unsigned int Q, unsigned int latticeSizeX,
                unsigned int latticeSizeY, unsigned int latticeSizeZ,
                unsigned int partitions);
};
