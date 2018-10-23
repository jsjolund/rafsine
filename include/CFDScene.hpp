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
enum Enum {
  VELOCITY_NORM, /**< Fluid velocity norm */
  DENSITY,       /**< Fluid density */
  TEMPERATURE    /**< Fluid temperature */
};
}

/**
 * @brief Enumerates display modes
 *
 */
namespace DisplayMode {
enum Enum {
  SLICE,       /**< Display slice of current display quantity*/
  VOX_GEOMETRY /**< Display 3D model of voxel geometry*/
};
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
  /**
   * @brief Resize the various HUD objects to fit the screen
   *
   * @param width
   * @param height
   */
  void resize(int width, int height);
  /**
   * @brief Get the HUD projection matrix
   *
   * @return osg::ref_ptr<osg::Projection>
   */
  inline osg::ref_ptr<osg::Projection> getHUDmatrix() {
    return m_hud->getProjectionMatrix();
  }
  /**
   * @brief Get the transform for the axis direction 3d model
   *
   * @return osg::ref_ptr<osg::PositionAttitudeTransform>
   */
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getAxes() {
    return m_axes;
  }
  /**
   * @brief Display/ignore the axis direction 3d model
   *
   * @param visible
   */
  inline void setAxesVisible(bool visible) {
    if (m_hud->getChildIndex(m_axes) == m_hud->getNumChildren()) {
      // Axes not in scene
      if (visible) m_hud->addChild(m_axes);
    } else {
      if (!visible) m_hud->removeChild(m_axes);
    }
  }
  /**
   * @brief Get the currently active display quantity (i.e. temperature,
   * velocity...)
   *
   * @return DisplayQuantity::Enum
   */
  inline DisplayQuantity::Enum getDisplayQuantity() {
    return m_displayQuantity;
  }
  /**
   * @brief Set the currently active display quantity (i.e. temperature,
   * velocity...)
   *
   * @return DisplayQuantity::Enum
   */
  void setDisplayQuantity(DisplayQuantity::Enum quantity);
  /**
   * @brief Set the currently active display mode (i.e. slices, voxels)
   *
   * @param mode
   */
  void setDisplayMode(DisplayMode::Enum mode);
  /**
   * @brief Adjust the colors of slices to range between min and max measured
   *
   */
  void adjustDisplayColors();
  /**
   * @brief Get a GPU pointer to the CFD visualization plot
   *
   * @return real*
   */
  inline real *gpu_ptr() { return thrust::raw_pointer_cast(&(m_plot3d)[0]); }
  /**
   * @brief Get the scene graph root
   *
   * @return osg::ref_ptr<osg::Group>
   */
  inline osg::ref_ptr<osg::Group> getRoot() { return m_root; }
  /**
   * @brief Get the center of the voxel geometry in world coordinates
   *
   * @return osg::Vec3
   */
  osg::Vec3 getCenter();
  /**
   * @brief Set the voxel geometry 3D model to be displayed
   *
   * @param voxels
   */
  void setVoxelGeometry(std::shared_ptr<VoxelGeometry> voxels);
  /**
   * @brief Move a display slice
   *
   * @param axis X-,Y-, or Z-axis
   * @param inc Number of voxels to move along negative or positive axis
   */
  void moveSlice(SliceRenderAxis::Enum axis, int inc);
  /**
   * @brief Select a voxel to display extra information about
   *
   * @param worldCoords The coordinates of the voxel
   * @return true True if there is a voxel present at coordinates
   * @return false
   */
  bool selectVoxel(osg::Vec3d worldCoords);
  /**
   * @brief Remove any extra information about a selected voxel
   *
   */
  void deselectVoxel();
  /**
   * @brief Set the color scheme for the display slices
   *
   * @param colorScheme
   */
  void setColorScheme(ColorScheme::Enum colorScheme);

  CFDScene();
};
