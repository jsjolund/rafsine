#pragma once

#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgDB/ReadFile>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3i>
#include <osg/Vec3d>

#include <osg/Material>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/LightModel>

#include <thrust/device_vector.h>

#include "VoxelMesh.hpp"
#include "VoxelContourMesh.hpp"
#include "VoxelFloorMesh.hpp"
#include "VoxelGeometry.hpp"
#include "CFDScene.hpp"
#include "SliceRender.hpp"
#include "BoundaryCondition.hpp"

// Which quantity to display
namespace DisplayQuantity
{
enum Enum
{
  VELOCITY_NORM,
  DENSITY,
  TEMPERATURE
};
}

namespace DisplayMode
{
enum Enum
{
  SLICE,
  VOX_GEOMETRY
};
}

class CFDScene
{
private:
  osg::ref_ptr<osg::Group> m_root;

  std::shared_ptr<VoxelGeometry> m_voxels;
  osg::ref_ptr<VoxelMesh> m_voxMesh;
  osg::ref_ptr<VoxelContourMesh> m_voxContour;
  osg::ref_ptr<VoxelFloorMesh> m_voxFloor;
  osg::Vec3i *m_voxSize, *m_voxMax, *m_voxMin;

  osg::ref_ptr<SliceRender> m_sliceX, m_sliceY, m_sliceZ;
  osg::Vec3i *m_slicePositions;

  DisplayMode::Enum m_displayMode;
  DisplayQuantity::Enum m_displayQuantity;

  // GPU memory to store the display informations
  thrust::device_vector<real> m_plot3d;
  // GPU memory to store color set gradient image
  thrust::device_vector<real> m_plotGradient;
  // Minimum and maximum value in the plot (used for color scaling)
  real m_plotMin, m_plotMax;

public:
  void adjustDisplayColors();
  inline DisplayQuantity::Enum getDisplayQuantity() { return m_displayQuantity; }
  void setDisplayQuantity(DisplayQuantity::Enum quantity);
  void setDisplayMode(DisplayMode::Enum mode);

  inline real *getPlot3d() { return thrust::raw_pointer_cast(&(m_plot3d)[0]); }
  inline osg::ref_ptr<osg::Group> getRoot() { return m_root; }
  osg::Vec3 getCenter();

  inline VoxelMesh *getVoxelMesh() { return m_voxMesh; }
  void setVoxelGeometry(std::shared_ptr<VoxelGeometry> voxels);

  void frame(osg::Camera &camera);
  void redrawVoxelMesh();

  void moveSlice(SliceRenderAxis::Enum axis, int inc);

  bool pickVoxel(osg::Vec3d worldCoords);

  CFDScene();
};
