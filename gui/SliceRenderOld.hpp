// #pragma once

// #include <osgDB/ReadFile>
// #include <osg/PositionAttitudeTransform>
// #include <osg/Texture2D>

// #include <cuda_gl_interop.h>
// #include <thrust/device_vector.h>

// #include "../geo/CudaUtils.hpp"

// // Render the volume as a slice cut at z=slice_pos
// __global__ void SliceZRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos);

// // Render the volume as a slice cut at y=slice_pos
// __global__ void SliceYRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos);

// // Render the volume as a slice cut at x=slice_pos
// __global__ void SliceXRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos);

// //kernel for black and white colors
// __global__ void compute_color_kernel_black_and_white(uchar3 *d_color_array,
//                                                      real *d_plot,
//                                                      unsigned int width,
//                                                      unsigned int height,
//                                                      real min,
//                                                      real max);

// //kernel to replicate default Paraview colors
// __global__ void compute_color_kernel_paraview(uchar3 *d_color_array,
//                                               real *d_plot,
//                                               unsigned int width,
//                                               unsigned int height,
//                                               real min,
//                                               real max);

// //rainbow colors
// __global__ void compute_color_kernel_rainbow(uchar3 *d_color_array,
//                                              real *d_plot,
//                                              unsigned int width,
//                                              unsigned int height,
//                                              real min,
//                                              real max);

// __global__ void compute_color_kernel_diverging(uchar3 *d_color_array,
//                                                real *d_plot,
//                                                unsigned int width,
//                                                unsigned int height,
//                                                real min,
//                                                real max);

// //Oblivion colors
// __global__ void compute_color_kernel_Oblivion(uchar3 *d_color_array,
//                                               real *d_plot,
//                                               unsigned int width,
//                                               unsigned int height,
//                                               real min,
//                                               real max);

// __global__ void compute_color_kernel_blues(uchar3 *d_color_array,
//                                            real *d_plot,
//                                            unsigned int width,
//                                            unsigned int height,
//                                            real min,
//                                            real max);

// __global__ void compute_color_kernel_sand(uchar3 *d_color_array,
//                                           real *d_plot,
//                                           unsigned int width,
//                                           unsigned int height,
//                                           real min,
//                                           real max);

// __global__ void compute_color_kernel_fire(uchar3 *d_color_array,
//                                           real *d_plot,
//                                           unsigned int width,
//                                           unsigned int height,
//                                           real min,
//                                           real max);

// namespace ColorScheme
// {
// enum Enum
// {
//   BLACK_AND_WHITE,
//   RAINBOW,
//   DIVERGING,
//   OBLIVION,
//   BLUES,
//   SAND,
//   FIRE,
//   PARAVIEW
// };
// }

// class SliceRender
// {
// private:
//   //resolution of the slice
//   unsigned int width_, height_;
//   //Data on the slice
//   thrust::device_vector<real> plot_d_;
//   //Index of the pixel buffer object
//   GLuint pboID_;
//   // Cuda rendering stream for texture compute
//   cudaStream_t renderStream_;
//   // Pixel buffer
//   osg::ref_ptr<osg::GraphicsContext> pbuffer_;
//   // Texture
//   osg::ref_ptr<osg::Texture2D> texture_;
//   // Image
//   osg::ref_ptr<osg::Image> image_;
//   // 3D Quad
//   osg::ref_ptr<osg::Geometry> quad_;
//   // Texture holder
//   osg::ref_ptr<osg::Geode> holder_;

// public:
//   osg::ref_ptr<osg::PositionAttitudeTransform> transform;

//   //Constructor
//   SliceRender(cudaStream_t renderStream, unsigned int width, unsigned int height);
//   //TODO: destructor to release GPU memory and OpenGL memory
//   //return a device pointer to the GPU data
//   real *gpu_ptr();
//   //Compute the slice
//   void compute(real min, real max);
//   //Bind the slice texture for rendering
//   void bind();
//   //Color scheme
//   ColorScheme::Enum color_scheme_;
// };
