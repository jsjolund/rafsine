#pragma once

#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgDB/ReadFile>
#include <osg/PositionAttitudeTransform>

#include "../geo/VoxelMesh.hpp"

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
  VoxelMesh *voxmesh_;
  osg::Geometry *voxGeo;
  osg::PositionAttitudeTransform *voxGeoTransform;
  DisplayMode::Enum displayMode_;
  DisplayQuantity::Enum displayQuantity_;
  vec3<int> vox_size_, vox_max_, vox_min_, slice_pos_;

  cudaStream_t renderStream_;
  // SliceRender *sliceX_, *sliceY_, *sliceZ_, *sliceC_;
  osg::ref_ptr<osg::Group> root_;

  // GPU memory to store the display informations
  thrust::device_vector<real> plot_d_;
  // GPU memory to store color set gradient image
  thrust::device_vector<real> plot_c_;
  // Minimum and maximum value in the plot (used for color scaling)
  real min_;
  real max_;
  // Size of the color map gradient
  unsigned int sizeC_;

  real tmp_;

public:
  inline osg::ref_ptr<osg::Group> getRoot() { return root_; }
  inline VoxelMesh *getVoxelMesh() { return voxmesh_; }

  void setCudaRenderStream(cudaStream_t stream) { renderStream_ = stream; };
  void redrawVoxelMesh();
  void setVoxelMesh(VoxelMesh *mesh);
  void sliceXup();
  void sliceXdown();
  void sliceYup();
  void sliceYdown();
  void sliceZup();
  void sliceZdown();
  void setSliceXpos(int pos);
  void setSliceYpos(int pos);
  void setSliceZpos(int pos);
  inline void setDisplayMode(DisplayMode::Enum mode) { displayMode_ = mode; }
  // TODO: destructor to release GPU memory and OpenGL memory
  // Return a pointer to the plot data on the GPU memory
  inline real *gpu_ptr() { return thrust::raw_pointer_cast(&(plot_d_)[0]); }
  inline real *gpu_ptr_c() { return thrust::raw_pointer_cast(&(plot_c_)[0]); }
  void drawSliceX();
  void drawSliceY();
  void drawSliceZ();

  osg::Vec3 getCenter();

  CFDScene();
};