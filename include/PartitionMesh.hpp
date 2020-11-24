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
  /**
   * @brief Adds a label showing CUDA device number for partition
   *
   * @param center
   * @param content
   */
  void addLabel(osg::Vec3d center, std::string content);

  /**
   * @brief Sets graphical settings such as opacity
   *
   * @param drawable
   */
  void setProperties(osg::ref_ptr<osg::ShapeDrawable> drawable);

 public:
  /**
   * @brief Construct a new Partition Mesh object
   *
   * @param voxMesh The full voxel mesh
   * @param nd Number of CUDA devices
   * @param partitioning Lattice partitioning axis
   * @param alpha Opacity 0.0 - 1.0
   */
  PartitionMesh(const VoxelMesh& voxMesh,
                int nd,
                D3Q4::Enum partitioning,
                float alpha);
};
