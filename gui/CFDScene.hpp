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

#include "../geo/VoxelMesh.hpp"
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
  VoxelMesh *m_voxMesh;
  osg::PositionAttitudeTransform *m_voxGeoTransform;
  osg::Vec3i *m_voxSize, *m_voxMax, *m_voxMin;

  DisplayMode::Enum m_displayMode;
  DisplayQuantity::Enum m_displayQuantity;

  osg::ref_ptr<SliceRender> m_sliceX, m_sliceY, m_sliceZ;
  osg::Vec3i *m_slicePositions;

  cudaStream_t m_renderStream;

  osg::ref_ptr<osg::Group> m_root;

  // GPU memory to store the display informations
  thrust::device_vector<real> m_plot_d;
  // GPU memory to store color set gradient image
  thrust::device_vector<real> m_plot_c;
  // Minimum and maximum value in the plot (used for color scaling)
  real m_plotMin, m_plotMax;
  // Size of the color map gradient
  unsigned int sizeC_;

public:
  inline osg::ref_ptr<osg::Group> getRoot() { return m_root; }
  inline VoxelMesh *getVoxelMesh() { return m_voxMesh; }

  void setCudaRenderStream(cudaStream_t stream) { m_renderStream = stream; };
  void redrawVoxelMesh();
  void setVoxelMesh(VoxelMesh *mesh);
  inline void setDisplayMode(DisplayMode::Enum mode) { m_displayMode = mode; }
  // TODO: destructor to release GPU memory and OpenGL memory
  // Return a pointer to the plot data on the GPU memory
  inline real *gpu_ptr() { return thrust::raw_pointer_cast(&(m_plot_d)[0]); }
  inline real *gpu_ptr_c() { return thrust::raw_pointer_cast(&(m_plot_c)[0]); }
  
  void moveSlice(SliceRenderAxis::Enum axis, int inc);

  osg::Vec3 getCenter();

  CFDScene();
};
