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
  SliceRender(D3Q4::Enum axis, unsigned int width, unsigned int height,
              real *plot3d, osg::Vec3i plot3dSize);

  /**
   * @brief Set the min/max values of the color range
   *
   * @param min
   * @param max
   */
  inline virtual void setMinMax(real min, real max) {
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
  ~SliceRender();
  //! World transform matrix of the quad
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  //! Min threshold for determining plot color from 2D slice df values
  real m_min;
  //! Max threshold for determining plot color from 2D slice df values
  real m_max;
  //! Axis of slice
  D3Q4::Enum m_axis;
  //! Pointer to the plot on GPU
  real *m_plot3d;
  //! Number of voxels in each direction
  osg::Vec3i m_plot3dSize;
  //! Color scheme
  ColorScheme::Enum m_colorScheme;

  virtual void runCudaKernel(uchar3 *texDevPtr, unsigned int texWidth,
                             unsigned int texHeight) const;
};

// Render the volume as a slice cut at z=slice_pos
__global__ void SliceGradientRenderKernel(real *plot3D, int nx, int ny, int nz,
                                          real *plot2D, int slice_pos);

// Render the volume as a slice cut at z=slice_pos
__global__ void SliceZRenderKernel(real *plot3D, int nx, int ny, int nz,
                                   real *plot2D, int slice_pos);

// Render the volume as a slice cut at y=slice_pos
__global__ void SliceYRenderKernel(real *plot3D, int nx, int ny, int nz,
                                   real *plot2D, int slice_pos);

// Render the volume as a slice cut at x=slice_pos
__global__ void SliceXRenderKernel(real *plot3D, int nx, int ny, int nz,
                                   real *plot2D, int slice_pos);

// kernel for black and white colors
__global__ void compute_color_kernel_black_and_white(uchar3 *d_color_array,
                                                     real *d_plot,
                                                     unsigned int width,
                                                     unsigned int height,
                                                     real min, real max);

// kernel to replicate default Paraview colors
__global__ void compute_color_kernel_paraview(uchar3 *d_color_array,
                                              real *d_plot, unsigned int width,
                                              unsigned int height, real min,
                                              real max);

// rainbow colors
__global__ void compute_color_kernel_rainbow(uchar3 *d_color_array,
                                             real *d_plot, unsigned int width,
                                             unsigned int height, real min,
                                             real max);

__global__ void compute_color_kernel_diverging(uchar3 *d_color_array,
                                               real *d_plot, unsigned int width,
                                               unsigned int height, real min,
                                               real max);

// Oblivion colors
__global__ void compute_color_kernel_Oblivion(uchar3 *d_color_array,
                                              real *d_plot, unsigned int width,
                                              unsigned int height, real min,
                                              real max);

__global__ void compute_color_kernel_blues(uchar3 *d_color_array, real *d_plot,
                                           unsigned int width,
                                           unsigned int height, real min,
                                           real max);

__global__ void compute_color_kernel_sand(uchar3 *d_color_array, real *d_plot,
                                          unsigned int width,
                                          unsigned int height, real min,
                                          real max);

__global__ void compute_color_kernel_fire(uchar3 *d_color_array, real *d_plot,
                                          unsigned int width,
                                          unsigned int height, real min,
                                          real max);
