#pragma once

#include <osg/Geode>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3i>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "CudaTexturedQuadGeometry.hpp"
#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "SliceRenderKernel.hpp"

namespace ColorScheme {
enum Enum {
  BLACK_AND_WHITE,
  RAINBOW,
  DIVERGING,
  OBLIVION,
  BLUES,
  SAND,
  FIRE,
  PARAVIEW
};
}  // namespace ColorScheme

/**
 * @brief A 3D quad with CUDA generated texture which shows a graphical
 * representation of fluid properties such as temperature, velocity and density.
 *
 */
class SliceRender : public CudaTexturedQuadGeometry {
 public:
  /**
   * @brief Construct a new Slice Render object
   *
   * @param axis The direction in which the slice can move
   * @param width Width of the texture
   * @param height Height of the texture
   * @param plot3d  Pointer to the 3D plot from which the slice texture is
   * generated
   * @param plot3dSize Size of the 3D plot
   */
  SliceRender(D3Q4::Enum axis,
              unsigned int width,
              unsigned int height,
              osg::Vec3i plot3dSize);

  /**
   * @brief Set the min/max values of the color range
   *
   * @param min
   * @param max
   */
  inline virtual void setMinMax(real_t min, real_t max) {
    m_min = min;
    m_max = max;
  }

  /**
   * @brief Get the OSG world transform. Manipulating this moves/rotates the
   * slice.
   *
   * @return osg::ref_ptr<osg::PositionAttitudeTransform>
   */
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() {
    return m_transform;
  }
  /**
   * @brief Set the color scheme of the slice
   *
   * @param colorScheme
   */
  inline void setColorScheme(ColorScheme::Enum colorScheme) {
    m_colorScheme = colorScheme;
  }

 protected:
  //! World transform matrix of the quad
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  //! Min threshold for determining plot color from 2D slice df values
  real_t m_min;
  //! Max threshold for determining plot color from 2D slice df values
  real_t m_max;
  //! Axis of slice
  D3Q4::Enum m_axis;
  //! Number of voxels in each direction
  osg::Vec3i m_plot3dSize;
  //! Color scheme
  ColorScheme::Enum m_colorScheme;

  ~SliceRender() {}

  virtual void runCudaKernel(real_t* srcPtr,
                             uchar3* texDevPtr,
                             unsigned int texWidth,
                             unsigned int texHeight) const;
};
