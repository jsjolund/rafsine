#pragma once

#include <cuda.h>
#include "CudaUtils.hpp"

// Render the volume as a slice cut at z=slice_pos
__global__ void SliceGradientRenderKernel(real_t* plot3D,
                                          int nx,
                                          int ny,
                                          int nz,
                                          real_t* plot2D,
                                          int slice_pos);

// Render the volume as a slice cut at z=slice_pos
__global__ void SliceZRenderKernel(real_t* plot3D,
                                   int nx,
                                   int ny,
                                   int nz,
                                   real_t* plot2D,
                                   int slice_pos);

// Render the volume as a slice cut at y=slice_pos
__global__ void SliceYRenderKernel(real_t* plot3D,
                                   int nx,
                                   int ny,
                                   int nz,
                                   real_t* plot2D,
                                   int slice_pos);

// Render the volume as a slice cut at x=slice_pos
__global__ void SliceXRenderKernel(real_t* plot3D,
                                   int nx,
                                   int ny,
                                   int nz,
                                   real_t* plot2D,
                                   int slice_pos);

// kernel for black and white colors
__global__ void compute_color_kernel_black_and_white(uchar3* d_color_array,
                                                     real_t* d_plot,
                                                     unsigned int width,
                                                     unsigned int height,
                                                     real_t min,
                                                     real_t max);

// kernel to replicate default Paraview colors
__global__ void compute_color_kernel_paraview(uchar3* d_color_array,
                                              real_t* d_plot,
                                              unsigned int width,
                                              unsigned int height,
                                              real_t min,
                                              real_t max);

// rainbow colors
__global__ void compute_color_kernel_rainbow(uchar3* d_color_array,
                                             real_t* d_plot,
                                             unsigned int width,
                                             unsigned int height,
                                             real_t min,
                                             real_t max);

__global__ void compute_color_kernel_diverging(uchar3* d_color_array,
                                               real_t* d_plot,
                                               unsigned int width,
                                               unsigned int height,
                                               real_t min,
                                               real_t max);

// Oblivion colors
__global__ void compute_color_kernel_Oblivion(uchar3* d_color_array,
                                              real_t* d_plot,
                                              unsigned int width,
                                              unsigned int height,
                                              real_t min,
                                              real_t max);

__global__ void compute_color_kernel_blues(uchar3* d_color_array,
                                           real_t* d_plot,
                                           unsigned int width,
                                           unsigned int height,
                                           real_t min,
                                           real_t max);

__global__ void compute_color_kernel_sand(uchar3* d_color_array,
                                          real_t* d_plot,
                                          unsigned int width,
                                          unsigned int height,
                                          real_t min,
                                          real_t max);

__global__ void compute_color_kernel_fire(uchar3* d_color_array,
                                          real_t* d_plot,
                                          unsigned int width,
                                          unsigned int height,
                                          real_t min,
                                          real_t max);
