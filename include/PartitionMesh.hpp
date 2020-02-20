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
#include <string>

#include "BillboardText.hpp"
#include "ColorSet.hpp"
#include "DistributedLattice.hpp"
#include "VoxelMesh.hpp"

/**
 * @brief A 3D graphics model of domain decomposition
 *
 */
class PartitionMesh : public osg::Geode {
 private:
  ColorSet m_colorSet;
  osg::ref_ptr<VoxelMesh> m_voxMesh;

 protected:
  ~PartitionMesh() {}
  void addLabel(osg::Vec3d center, std::string content);
  void setProperties(osg::ref_ptr<osg::ShapeDrawable> drawable);

 public:
  /**
   * @brief Construct a new Partition Mesh object
   *
   * @param latticeSizeX Size of the lattice on X-axis
   * @param latticeSizeY Size of the lattice on Y-axis
   * @param latticeSizeZ Size of the lattice on Z-axis
   * @param partitions Number of lattice partitions
   * @param alpha Opacity 0.0 - 1.0
   */
  PartitionMesh(const VoxelMesh& voxMesh,
                int numDevices,
                D3Q4::Enum partitioning,
                float alpha);
};
