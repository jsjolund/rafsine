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
#include "CFDScene.hpp"
#include "SliceRender.hpp"

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

  osg::ref_ptr<VoxelMesh> m_voxMesh;
  osg::Vec3i *m_voxSize, *m_voxMax, *m_voxMin;

  osg::ref_ptr<SliceRender> m_sliceX, m_sliceY, m_sliceZ;
  osg::Vec3i *m_slicePositions;

  DisplayMode::Enum m_displayMode;
  DisplayQuantity::Enum m_displayQuantity;

  cudaStream_t m_renderStream;
  // GPU memory to store the display informations
  thrust::device_vector<real> m_plot3d;
  // GPU memory to store color set gradient image
  thrust::device_vector<real> m_plotGradient;
  // Minimum and maximum value in the plot (used for color scaling)
  real m_plotMin, m_plotMax;

  // osg::ref_ptr<osg::Node> m_axes;
  osg::ref_ptr<osg::PositionAttitudeTransform> m_axesTransform;

public:
  inline osg::ref_ptr<osg::Group> getRoot() { return m_root; }
  inline VoxelMesh *getVoxelMesh() { return m_voxMesh; }

  void setCudaRenderStream(cudaStream_t stream) { m_renderStream = stream; };
  void redrawVoxelMesh();
  void setVoxelMesh(VoxelMesh *mesh);
  void setDisplayMode(DisplayMode::Enum mode);
  void adjustDisplayColors();
  // // TODO: destructor to release GPU memory and OpenGL memory
  // // Return a pointer to the plot data on the GPU memory
  // inline real *gpu_ptr() { return thrust::raw_pointer_cast(&(m_plot3d)[0]); }
  // inline real *gpu_ptr_c() { return thrust::raw_pointer_cast(&(m_plotGradient)[0]); }

  void moveSlice(SliceRenderAxis::Enum axis, int inc);
  void frame(osg::Camera &camera);
  osg::Vec3 getCenter();

  CFDScene();

};