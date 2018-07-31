#pragma once

#include <osgViewer/Viewer>

#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>

#include "../geo/CudaUtils.hpp"

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

class SliceRender
{
private:
  //resolution of the slice
  unsigned int width_, height_;
  //Data on the slice
  thrust::device_vector<real> plot_d_;
  //Index of the pixel buffer object
  GLuint pboID_;
  // //Index of the corresponding texture
  // GLuint textureID_;
  cudaStream_t renderStream_;

  osg::ref_ptr<osg::GraphicsContext> pbuffer_;

public:
  //Constructor
  SliceRender(cudaStream_t renderStream, unsigned int width, unsigned int height);
  //TODO: destructor to release GPU memory and OpenGL memory
  //return a device pointer to the GPU data
  real *gpu_ptr();
  //Compute the slice
  void compute(real min, real max);
  //Bind the slice texture for rendering
  void bind();
  //Color scheme
  ColorScheme::Enum color_scheme_;
};
