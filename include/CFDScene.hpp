#pragma once

#include <memory>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <osg/Vec3i>
#include <osgDB/FileUtils>
#include <osgGA/TrackballManipulator>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <string>

#include "AxesMesh.hpp"
#include "BoundaryCondition.hpp"
#include "CFDHud.hpp"
#include "DistributionArray.hpp"
#include "HistogramMesh.hpp"
#include "PartitionMesh.hpp"
#include "SliceRender.hpp"
#include "SliceRenderGradient.hpp"
#include "Vector3.hpp"
#include "VoxelAreaMesh.hpp"
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
/**
 * @brief Enumerated quantities to display on slices
 *
 */
enum Enum {
  VELOCITY_NORM,  //< Fluid velocity norm
  DENSITY,        //< Fluid density */
  TEMPERATURE     //< Fluid temperature
};
}  // namespace DisplayQuantity

/**
 * @brief Enumerates display modes
 *
 */
namespace DisplayMode {
/**
 * @brief Enumeration of display modes
 *
 */
enum Enum {
  SLICE,         //< Display slice of current display quantity
  VOX_GEOMETRY,  //< Display 3D model of voxel geometry
  DEVICES        //< Display CUDA device domain decomposition
};
}  // namespace DisplayMode

/**
 * @brief This class holds the various 3d objects used for visualization of the
 * CFD situation
 *
 */
class CFDScene : public osg::Geode {
 private:
  // For slicing the voxel geometry
  osg::Vec3i *m_voxMin, *m_voxMax, *m_voxSize;

  bool m_showBCLabels;
  bool m_showAvgLabels;

  // Minimum and maximum value in the plot (used for color scaling)
  real_t m_plotMin, m_plotMax;

  // Gradient of display slices with labels
  osg::ref_ptr<SliceRenderGradient> m_sliceGradient;

  // Current position of slices
  osg::Vec3i* m_slicePositions;

  // Root node for HUD
  osg::ref_ptr<CFDHud> m_hud;

  // Visualization stuff
  osg::ref_ptr<HistogramMesh> m_histogram;
  osg::ref_ptr<VoxelMarker> m_marker;

  // Color scheme of slices
  ColorScheme::Enum m_colorScheme;

  // Axes arrows on HUD
  osg::ref_ptr<AxesMesh> m_axes;

  osg::ref_ptr<osg::Geode> m_voxLabels;
  osg::ref_ptr<osg::Geode> m_avgs;
  osg::ref_ptr<osg::Geode> m_avgLabels;
  osg::ref_ptr<PartitionMesh> m_partitionMesh;
  osg::ref_ptr<VoxelMesh> m_voxMesh;
  osg::ref_ptr<VoxelContourMesh> m_voxContour;
  osg::ref_ptr<VoxelFloorMesh> m_voxFloor;

  // Display slices
  osg::ref_ptr<SliceRender> m_sliceX, m_sliceY, m_sliceZ;

  // Current display mode (voxels or slices)
  DisplayMode::Enum m_displayMode;
  // Which quantity (temperature, etc)
  DisplayQuantity::Enum m_displayQuantity;

 public:
  vector3<int> getSlicePosition() {
    return vector3<int>(m_slicePositions->x(), m_slicePositions->y(),
                        m_slicePositions->z());
  }
  inline real_t* getSliceX() { return m_sliceX->gpu_ptr(); }
  inline real_t* getSliceY() { return m_sliceY->gpu_ptr(); }
  inline real_t* getSliceZ() { return m_sliceZ->gpu_ptr(); }

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
  void setAxesVisible(bool visible);
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
   * @brief Get the currently active display mode (i.e. slices, voxels)
   *
   * @return DisplayMode::Enum
   */
  inline DisplayMode::Enum getDisplayMode() { return m_displayMode; }
  /**
   * @brief Set true to show geometry labels
   *
   * @param state
   */
  void setLabelsVisible(bool state) {
    m_showBCLabels = state;
    setDisplayMode(m_displayMode);
  }
  void setSensorsVisible(bool state) {
    m_showAvgLabels = state;
    setDisplayMode(m_displayMode);
  }
  /**
   * @brief Adjust the colors of slices to range between min and max measured
   *
   */
  void adjustDisplayColors(real_t min, real_t max,
                           const thrust::host_vector<real_t>& histogram);
  /**
   * @brief Get the center of the voxel geometry in world coordinates
   *
   * @return osg::Vec3
   */
  osg::Vec3 getCenter();
  /**
   * @brief Set the voxel geometry 3D model to be displayed
   *
   * @param voxels The voxels generated by the LUA script
   * @param nd The number of CUDA devices
   */
  void setVoxelGeometry(std::shared_ptr<VoxelGeometry> voxels,
                        std::string voxMeshFilePath, int nd,
                        D3Q4::Enum partitioning);
  void deleteVoxelGeometry();
  /**
   * @brief Move a display slice
   *
   * @param axis X-,Y-, or Z-axis
   * @param inc Number of voxels to move along negative or positive axis
   */
  void moveSlice(D3Q4::Enum axis, int inc);
  // /**
  //  * @brief Select a voxel to display extra information about
  //  *
  //  * @param worldCoords The coordinates of the voxel
  //  * @return true True if there is a voxel present at coordinates
  //  * @return false
  //  */
  // bool selectVoxel(osg::Vec3d worldCoords);
  // /**
  //  * @brief Remove any extra information about a selected voxel
  //  *
  //  */
  // void deselectVoxel();
  /**
   * @brief Set the color scheme for the display slices
   *
   * @param colorScheme
   */
  void setColorScheme(ColorScheme::Enum colorScheme);

  CFDScene();
};
