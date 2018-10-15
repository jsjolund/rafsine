#pragma once

#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <osg/Vec3i>
#include <osgGA/TrackballManipulator>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <thrust/device_vector.h>

#include "AxesMesh.hpp"
#include "BoundaryCondition.hpp"
#include "CFDHud.hpp"
#include "CFDScene.hpp"
#include "SliceRender.hpp"
#include "SliceRenderGradient.hpp"
#include "VoxelContourMesh.hpp"
#include "VoxelFloorMesh.hpp"
#include "VoxelGeometry.hpp"
#include "VoxelMarker.hpp"
#include "VoxelMesh.hpp"

/**
 * @brief Enumerates quantities to display on slices
 *
 */
namespace DisplayQuantity {
enum Enum { VELOCITY_NORM, DENSITY, TEMPERATURE };
}

/**
 * @brief Enumerates display modes
 *
 */
namespace DisplayMode {
enum Enum { SLICE, VOX_GEOMETRY };
}

/**
 * @brief This class holds the various 3d objects used for visualization of the
 * CFD situation
 *
 */
class CFDScene {
 private:
  // Root node of non-HUD stuff
  osg::ref_ptr<osg::Group> m_root;
  // Root node for HUD
  osg::ref_ptr<CFDHud> m_hud;

  // The current voxel geometry
  std::shared_ptr<VoxelGeometry> m_voxels;
  // Visualization stuff
  osg::ref_ptr<VoxelMesh> m_voxMesh;
  osg::ref_ptr<VoxelContourMesh> m_voxContour;
  osg::ref_ptr<VoxelFloorMesh> m_voxFloor;
  osg::ref_ptr<VoxelMarker> m_marker;
  // For slicing the voxel geometry
  osg::Vec3i *m_voxSize, *m_voxMax, *m_voxMin;

  // Display slices
  osg::ref_ptr<SliceRender> m_sliceX, m_sliceY, m_sliceZ;
  osg::Vec3i *m_slicePositions;
  // Gradient of display slices with labels
  osg::ref_ptr<SliceRenderGradient> m_sliceGradient;

  // Current display mode (voxels or slices)
  DisplayMode::Enum m_displayMode;
  // Which quantity (temperature, etc)
  DisplayQuantity::Enum m_displayQuantity;
  // Axes arrows on HUD
  osg::ref_ptr<AxesMesh> m_axes;

  // GPU memory to store the display informations
  thrust::device_vector<real> m_plot3d;
  // GPU memory to store color set gradient image
  thrust::device_vector<real> m_plotGradient;
  // Minimum and maximum value in the plot (used for color scaling)
  real m_plotMin, m_plotMax;

 public:
  void resize(int width, int height);

  inline osg::ref_ptr<osg::Projection> getHUDmatrix() {
    return m_hud->m_projectionMatrix;
  }
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getAxes() {
    return m_axes;
  }
  inline void setAxesVisible(bool visible) {
    if (m_hud->getChildIndex(m_axes) == m_hud->getNumChildren()) {
      // Axes not in scene
      if (visible) m_hud->addChild(m_axes);
    } else {
      if (!visible) m_hud->removeChild(m_axes);
    }
  }

  inline DisplayQuantity::Enum getDisplayQuantity() {
    return m_displayQuantity;
  }
  void setDisplayQuantity(DisplayQuantity::Enum quantity);
  void setDisplayMode(DisplayMode::Enum mode);
  void adjustDisplayColors();

  inline real *getPlot3d() { return thrust::raw_pointer_cast(&(m_plot3d)[0]); }
  inline osg::ref_ptr<osg::Group> getRoot() { return m_root; }
  osg::Vec3 getCenter();

  inline VoxelMesh *getVoxelMesh() { return m_voxMesh; }
  void setVoxelGeometry(std::shared_ptr<VoxelGeometry> voxels);

  void moveSlice(SliceRenderAxis::Enum axis, int inc);

  bool selectVoxel(osg::Vec3d worldCoords);
  void deselectVoxel();

  inline osg::ref_ptr<osg::Geometry> getSliceRenderGradient() {
    return m_sliceGradient;
  }

  void setColorScheme(ColorScheme::Enum colorScheme);

  CFDScene();
};
