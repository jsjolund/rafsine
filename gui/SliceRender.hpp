#pragma once

#include <osg/PositionAttitudeTransform>
#include <osg/Geode>

#include <cuda.h>
#include <thrust/device_vector.h>

#include "../cuda/CudaTexturedQuadGeometry.hpp"
#include "../cuda/CudaUtils.hpp"

// Render the volume as a slice cut at z=slice_pos
__global__ void SliceZRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos);

// Render the volume as a slice cut at y=slice_pos
__global__ void SliceYRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos);

// Render the volume as a slice cut at x=slice_pos
__global__ void SliceXRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos);

//kernel for black and white colors
__global__ void compute_color_kernel_black_and_white(uchar3 *d_color_array,
                                                     real *d_plot,
                                                     unsigned int width,
                                                     unsigned int height,
                                                     real min,
                                                     real max);

//kernel to replicate default Paraview colors
__global__ void compute_color_kernel_paraview(uchar3 *d_color_array,
                                              real *d_plot,
                                              unsigned int width,
                                              unsigned int height,
                                              real min,
                                              real max);

//rainbow colors
__global__ void compute_color_kernel_rainbow(uchar3 *d_color_array,
                                             real *d_plot,
                                             unsigned int width,
                                             unsigned int height,
                                             real min,
                                             real max);

__global__ void compute_color_kernel_diverging(uchar3 *d_color_array,
                                               real *d_plot,
                                               unsigned int width,
                                               unsigned int height,
                                               real min,
                                               real max);

//Oblivion colors
__global__ void compute_color_kernel_Oblivion(uchar3 *d_color_array,
                                              real *d_plot,
                                              unsigned int width,
                                              unsigned int height,
                                              real min,
                                              real max);

__global__ void compute_color_kernel_blues(uchar3 *d_color_array,
                                           real *d_plot,
                                           unsigned int width,
                                           unsigned int height,
                                           real min,
                                           real max);

__global__ void compute_color_kernel_sand(uchar3 *d_color_array,
                                          real *d_plot,
                                          unsigned int width,
                                          unsigned int height,
                                          real min,
                                          real max);

__global__ void compute_color_kernel_fire(uchar3 *d_color_array,
                                          real *d_plot,
                                          unsigned int width,
                                          unsigned int height,
                                          real min,
                                          real max);

namespace ColorScheme
{
enum Enum
{
  BLACK_AND_WHITE,
  RAINBOW,
  DIVERGING,
  OBLIVION,
  BLUES,
  SAND,
  FIRE,
  PARAVIEW
};
}
namespace SliceRenderAxis
{
enum Enum
{
  X_AXIS,
  Y_AXIS,
  Z_AXIS
};
}

class SliceRender : public CudaTexturedQuadGeometry
{
private:
  // Cuda rendering stream for texture compute
  cudaStream_t m_renderStream;

  // World transform matrix of the quad
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;

public:
  // Min and max thresholds for determining plot color from 2D slice df values
  real m_min, m_max;

  // Axis of slice
  SliceRenderAxis::Enum m_axis;

  // Color scheme
  ColorScheme::Enum m_colorScheme;

  SliceRender(SliceRenderAxis::Enum axis, unsigned int width, unsigned int height, cudaStream_t renderStream);

  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() { return m_transform; }

protected:
  virtual void runCudaKernel(uchar3 *texDevPtr,
                             unsigned int texWidth,
                             unsigned int texHeight) const;
};