#include "SliceRender.hpp"

SliceRender::~SliceRender() {}

SliceRender::SliceRender(SliceRenderAxis::Enum axis, unsigned int width,
                         unsigned int height, real *plot3d, osg::Vec3i voxSize)
    : CudaTexturedQuadGeometry(width, height),
      m_voxSize(voxSize),
      m_plot3d(plot3d),
      m_colorScheme(ColorScheme::PARAVIEW),
      m_axis(axis),
      m_min(0),
      m_max(0) {
  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);
  osg::StateSet *stateSet = geode->getOrCreateStateSet();
  stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);

  m_transform = new osg::PositionAttitudeTransform();
  m_transform->addChild(geode);
}

void SliceRender::runCudaKernel(uchar3 *texDevPtr, unsigned int texWidth,
                                unsigned int texHeight) const {
  dim3 blockSize, gridSize;

  thrust::device_vector<real> slice(texWidth * texHeight);
  real *slicePtr = thrust::raw_pointer_cast(&(slice)[0]);

  osg::Vec3d position = m_transform->getPosition();

  switch (m_axis) {
    case SliceRenderAxis::X_AXIS:
      setDims(m_voxSize.y() * m_voxSize.z(), BLOCK_SIZE_DEFAULT, blockSize,
              gridSize);
      SliceXRenderKernel<<<gridSize, blockSize>>>(m_plot3d, m_voxSize.x(),
                                                  m_voxSize.y(), m_voxSize.z(),
                                                  slicePtr, position.x());
      CUDA_CHECK_ERRORS("SliceXRenderKernel");
      break;
    case SliceRenderAxis::Y_AXIS:
      setDims(m_voxSize.x() * m_voxSize.z(), BLOCK_SIZE_DEFAULT, blockSize,
              gridSize);
      SliceYRenderKernel<<<gridSize, blockSize>>>(m_plot3d, m_voxSize.x(),
                                                  m_voxSize.y(), m_voxSize.z(),
                                                  slicePtr, position.y());
      CUDA_CHECK_ERRORS("SliceYRenderKernel");
      break;
    case SliceRenderAxis::Z_AXIS:
      setDims(m_voxSize.x() * m_voxSize.y(), BLOCK_SIZE_DEFAULT, blockSize,
              gridSize);
      SliceZRenderKernel<<<gridSize, blockSize>>>(m_plot3d, m_voxSize.x(),
                                                  m_voxSize.y(), m_voxSize.z(),
                                                  slicePtr, position.z());
      CUDA_CHECK_ERRORS("SliceZRenderKernel");
      break;
    case SliceRenderAxis::GRADIENT:
      setDims(texWidth * texHeight, BLOCK_SIZE_DEFAULT, blockSize, gridSize);
      SliceGradientRenderKernel<<<gridSize, blockSize>>>(
          m_plot3d, texWidth, texHeight, 1, slicePtr, 0);
      CUDA_CHECK_ERRORS("SliceGradientRenderKernel");
      break;
  }

  // Configure block size and grid size
  setDims(texWidth * texHeight, BLOCK_SIZE_DEFAULT, blockSize, gridSize);

  switch (m_colorScheme) {
    case ColorScheme::BLACK_AND_WHITE:
      compute_color_kernel_black_and_white<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_black_and_white");
      break;
    case ColorScheme::RAINBOW:
      compute_color_kernel_rainbow<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_rainbow");
      break;
    case ColorScheme::DIVERGING:
      compute_color_kernel_diverging<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_diverging");
      break;
    case ColorScheme::OBLIVION:
      compute_color_kernel_Oblivion<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_Oblivion");
      break;
    case ColorScheme::BLUES:
      compute_color_kernel_blues<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_blues");
      break;
    case ColorScheme::SAND:
      compute_color_kernel_sand<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_sand");
      break;
    case ColorScheme::FIRE:
      compute_color_kernel_fire<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_fire");
      break;
    case ColorScheme::PARAVIEW:
      compute_color_kernel_paraview<<<gridSize, blockSize>>>(
          texDevPtr, slicePtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_paraview");
      break;
  }
}
__global__ void SliceGradientRenderKernel(real *plot3D, int nx, int ny, int nz,
                                          real *plot2D, int slice_pos) {
  int x, y;
  idx2d(x, y, nx);
  if ((x >= nx) || (y >= ny)) return;
  // plot2D[x+nx*y] = plot3D[I3D(x, y, slice_pos, nx,ny,nz)];
  // gaussian blur
  int xp = (x == nx - 1) ? (x) : (x + 1);
  int xm = (x == 0) ? (x) : (x - 1);
  int yp = (y == ny - 1) ? (y) : (y + 1);
  int ym = (y == 0) ? (y) : (y - 1);
  int max = (nx - 1) + nx * (ny - 1);
  plot2D[max - (x + nx * y)] =
      1 / 4.f * plot3D[I3D(x, y, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(xp, y, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(xm, y, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(x, yp, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(x, ym, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xm, ym, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xm, yp, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xp, ym, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xp, yp, slice_pos, nx, ny, nz)];
  // average over the height
  /*
  float average = 0;
  for(int z=0; z<nz; z++)
    average += plot3D[I3D(x, y, z, nx,ny,nz)];
  plot2D[x+nx*y] = average/nz;
  */
}

__global__ void SliceZRenderKernel(real *plot3D, int nx, int ny, int nz,
                                   real *plot2D, int slice_pos) {
  int x, y;
  idx2d(x, y, nx);
  if ((x >= nx) || (y >= ny)) return;
  // plot2D[x+nx*y] = plot3D[I3D(x, y, slice_pos, nx,ny,nz)];
  // gaussian blur
  int xp = (x == nx - 1) ? (x) : (x + 1);
  int xm = (x == 0) ? (x) : (x - 1);
  int yp = (y == ny - 1) ? (y) : (y + 1);
  int ym = (y == 0) ? (y) : (y - 1);
  plot2D[x + nx * y] = 1 / 4.f * plot3D[I3D(x, y, slice_pos, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(xp, y, slice_pos, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(xm, y, slice_pos, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(x, yp, slice_pos, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(x, ym, slice_pos, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xm, ym, slice_pos, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xm, yp, slice_pos, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xp, ym, slice_pos, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xp, yp, slice_pos, nx, ny, nz)];
  // average over the height
  /*
  float average = 0;
  for(int z=0; z<nz; z++)
    average += plot3D[I3D(x, y, z, nx,ny,nz)];
  plot2D[x+nx*y] = average/nz;
  */
}

__global__ void SliceYRenderKernel(real *plot3D, int nx, int ny, int nz,
                                   real *plot2D, int slice_pos) {
  int x, z;
  idx2d(x, z, nx);
  if ((x >= nx) || (z >= nz)) return;
  // plot2D[x+nx*z] = plot3D[I3D(x, slice_pos, z, nx,ny,nz)];
  // gaussian blur
  int xp = (x == nx - 1) ? (x) : (x + 1);
  int xm = (x == 0) ? (x) : (x - 1);
  int zp = (z == nz - 1) ? (z) : (z + 1);
  int zm = (z == 0) ? (z) : (z - 1);
  plot2D[x + nx * z] = 1 / 4.f * plot3D[I3D(x, slice_pos, z, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(xp, slice_pos, z, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(xm, slice_pos, z, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(x, slice_pos, zp, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(x, slice_pos, zm, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xm, slice_pos, zm, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xm, slice_pos, zp, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xp, slice_pos, zm, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(xp, slice_pos, zp, nx, ny, nz)];
}

__global__ void SliceXRenderKernel(real *plot3D, int nx, int ny, int nz,
                                   real *plot2D, int slice_pos) {
  int y, z;
  idx2d(y, z, ny);
  if ((y >= ny) || (z >= nz)) return;
  // plot2D[y+ny*z] = plot3D[I3D(slice_pos, y, z, nx,ny,nz)];
  // gaussian blur
  int yp = (y == ny - 1) ? (y) : (y + 1);
  int ym = (y == 0) ? (y) : (y - 1);
  int zp = (z == nz - 1) ? (z) : (z + 1);
  int zm = (z == 0) ? (z) : (z - 1);
  plot2D[y + ny * z] = 1 / 4.f * plot3D[I3D(slice_pos, y, z, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(slice_pos, yp, z, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(slice_pos, ym, z, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(slice_pos, y, zp, nx, ny, nz)] +
                       1 / 8.f * plot3D[I3D(slice_pos, y, zm, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(slice_pos, ym, zm, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(slice_pos, ym, zp, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(slice_pos, yp, zm, nx, ny, nz)] +
                       1 / 16.f * plot3D[I3D(slice_pos, yp, zp, nx, ny, nz)];
}

__global__ void compute_color_kernel_black_and_white(uchar3 *d_color_array,
                                                     real *d_plot,
                                                     unsigned int width,
                                                     unsigned int height,
                                                     real min, real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    color.x = normal_value * 255;
    color.y = normal_value * 255;
    color.z = normal_value * 255;
    // increase intensity
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
}

__global__ void compute_color_kernel_paraview(uchar3 *d_color_array,
                                              real *d_plot, unsigned int width,
                                              unsigned int height, real min,
                                              real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    normal_value = 1 - normal_value;
    real v1 = 1.0 / 10.0;
    real v2 = 2.0 / 10.0;
    real v3 = 3.0 / 10.0;
    real v4 = 4.0 / 10.0;
    real v5 = 5.0 / 10.0;
    real v6 = 6.0 / 10.0;
    real v7 = 7.0 / 10.0;
    real v8 = 8.0 / 10.0;
    real v9 = 9.0 / 10.0;
    // compute color
    if (normal_value < v1) {
      real c = normal_value / v1;
      color.x = 103 * (1 - c) + 178 * c;
      color.y = 0 * (1 - c) + 24 * c;
      color.z = 31 * (1 - c) + 43 * c;
    } else if (normal_value < v2) {
      real c = (normal_value - v1) / (v2 - v1);
      color.x = 178 * (1 - c) + 214 * c;
      color.y = 24 * (1 - c) + 96 * c;
      color.z = 43 * (1 - c) + 77 * c;
    } else if (normal_value < v3) {
      real c = (normal_value - v2) / (v3 - v2);
      color.x = 214 * (1 - c) + 244 * c;
      color.y = 96 * (1 - c) + 165 * c;
      color.z = 77 * (1 - c) + 130 * c;
    } else if (normal_value < v4) {
      real c = (normal_value - v3) / (v4 - v3);
      color.x = 244 * (1 - c) + 253 * c;
      color.y = 165 * (1 - c) + 219 * c;
      color.z = 130 * (1 - c) + 199 * c;
    } else if (normal_value < v5) {
      real c = (normal_value - v4) / (v5 - v4);
      color.x = 253 * (1 - c) + 247 * c;
      color.y = 219 * (1 - c) + 247 * c;
      color.z = 199 * (1 - c) + 247 * c;
    } else if (normal_value < v6) {
      real c = (normal_value - v5) / (v6 - v5);
      color.x = 247 * (1 - c) + 209 * c;
      color.y = 247 * (1 - c) + 229 * c;
      color.z = 247 * (1 - c) + 240 * c;
    } else if (normal_value < v7) {
      real c = (normal_value - v6) / (v7 - v6);
      color.x = 209 * (1 - c) + 146 * c;
      color.y = 229 * (1 - c) + 197 * c;
      color.z = 240 * (1 - c) + 222 * c;
    } else if (normal_value < v8) {
      real c = (normal_value - v7) / (v8 - v7);
      color.x = 146 * (1 - c) + 67 * c;
      color.y = 197 * (1 - c) + 147 * c;
      color.z = 222 * (1 - c) + 195 * c;
    } else if (normal_value < v9) {
      real c = (normal_value - v8) / (v9 - v8);
      color.x = 67 * (1 - c) + 33 * c;
      color.y = 147 * (1 - c) + 102 * c;
      color.z = 195 * (1 - c) + 172 * c;
    } else {
      real c = (normal_value - v9) / (1 - v9);
      color.x = 33 * (1 - c) + 5 * c;
      color.y = 102 * (1 - c) + 48 * c;
      color.z = 172 * (1 - c) + 97 * c;
    }
    // increase intensity
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
}

__global__ void compute_color_kernel_rainbow(uchar3 *d_color_array,
                                             real *d_plot, unsigned int width,
                                             unsigned int height, real min,
                                             real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    normal_value = log1p(normal_value) / log(2.f);
    real v1 = 0.25;
    real v2 = 0.50;
    real v3 = 0.75;
    real v4 = 1.00;
    // compute color
    if (normal_value < v1) {  // blue to cian
      // red component
      color.x = 0;
      // green coponent
      color.y = (normal_value / v1) * 255;
      // blue component
      // color.z = (1.f - normal_value/v1) * 255;
      color.z = 255;
    } else if (normal_value < v2) {  // cian to green
      // red component
      color.x = 0;
      // green coponent
      color.y = 255;
      // blue component
      color.z = (1.f - (normal_value - v1) / (v2 - v1)) * 255;
    } else if (normal_value < v3) {  // green to yellow
      // red component
      color.x = (normal_value - v2) / (v3 - v2) * 255;
      // green coponent
      color.y = 255;
      // blue component
      color.z = 0.f;
    } else {  // yellow to red
      // red component
      color.x = 255;
      // green component
      color.y = (1.f - (normal_value - v3) / (v4 - v3)) * 255;
      // blue component
      color.z = 0.f;
    }
    // increase intensity
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
}

__global__ void compute_color_kernel_diverging(uchar3 *d_color_array,
                                               real *d_plot, unsigned int width,
                                               unsigned int height, real min,
                                               real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    normal_value = log1p(normal_value) / log(2.f);
    real v1 = 0.2;
    real v2 = 0.4;
    real v3 = 0.6;
    real v4 = 0.8;
    // compute color
    if (normal_value < v1) {
      real c = normal_value / v1;
      color.x = 43 * (1 - c) + 171 * c;
      color.y = 131 * (1 - c) + 221 * c;
      color.z = 186 * (1 - c) + 164 * c;
    } else if (normal_value < v2) {
      real c = (normal_value - v1) / (v2 - v1);
      color.x = 171 * (1 - c) + 255 * c;
      color.y = 221 * (1 - c) + 255 * c;
      color.z = 164 * (1 - c) + 191 * c;
    } else if (normal_value < v3) {
      real c = (normal_value - v2) / (v3 - v2);
      color.x = 255 * (1 - c) + 253 * c;
      color.y = 255 * (1 - c) + 174 * c;
      color.z = 191 * (1 - c) + 97 * c;
    } else if (normal_value < v4) {
      real c = (normal_value - v3) / (v4 - v3);
      color.x = 253 * (1 - c) + 215 * c;
      color.y = 174 * (1 - c) + 25 * c;
      color.z = 97 * (1 - c) + 28 * c;
    } else {
      real c = (normal_value - v4) / (1 - v4);
      color.x = 215 * (1 - c) + 255 * c;
      color.y = 25 * (1 - c) + 0 * c;
      color.z = 28 * (1 - c) + 0 * c;
    }
    // increase intensity
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
}

__global__ void compute_color_kernel_Oblivion(uchar3 *d_color_array,
                                              real *d_plot, unsigned int width,
                                              unsigned int height, real min,
                                              real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    // normal_value = log(1+normal_value)/log(2.f);
    real v1 = 1.0 / 5.0;
    real v2 = 2.0 / 5.0;
    real v3 = 3.0 / 5.0;
    real v4 = 4.0 / 5.0;
    // compute color
    if (normal_value < v1) {
      real c = normal_value / v1;
      color.x = 53 * (1 - c) + 104 * c;
      color.y = 70 * (1 - c) + 221 * c;
      color.z = 78 * (1 - c) + 239 * c;
    } else if (normal_value < v2) {
      real c = (normal_value - v1) / (v2 - v1);
      color.x = 104 * (1 - c) + 149 * c;
      color.y = 221 * (1 - c) + 243 * c;
      color.z = 239 * (1 - c) + 253 * c;
    } else if (normal_value < v3) {
      real c = (normal_value - v2) / (v3 - v2);
      color.x = 149 * (1 - c) + 223 * c;
      color.y = 243 * (1 - c) + 255 * c;
      color.z = 253 * (1 - c) + 254 * c;
    } else if (normal_value < v4) {
      real c = (normal_value - v3) / (v4 - v3);
      color.x = 223 * (1 - c) + 255 * c;
      color.y = 255 * (1 - c) + 255 * c;
      color.z = 254 * (1 - c) + 227 * c;
    } else {
      real c = (normal_value - v4) / (1 - v4);
      color.x = 255 * (1 - c) + 255 * c;
      color.y = 255 * (1 - c) + 115 * c;
      color.z = 227 * (1 - c) + 72 * c;
    }
    // increase intensity
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
}

__global__ void compute_color_kernel_blues(uchar3 *d_color_array, real *d_plot,
                                           unsigned int width,
                                           unsigned int height, real min,
                                           real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    // normal_value = log(1+normal_value)/log(2.f);
    real v1 = 1.0 / 8.0;
    real v2 = 2.0 / 8.0;
    real v3 = 3.0 / 8.0;
    real v4 = 4.0 / 8.0;
    real v5 = 5.0 / 8.0;
    real v6 = 6.0 / 8.0;
    real v7 = 7.0 / 8.0;
    // compute color
    if (normal_value < v1) {
      real c = normal_value / v1;
      color.x = 8 * (1 - c) + 37 * c;
      color.y = 29 * (1 - c) + 52 * c;
      color.z = 88 * (1 - c) + 148 * c;
    } else if (normal_value < v2) {
      real c = (normal_value - v1) / (v2 - v1);
      color.x = 37 * (1 - c) + 34 * c;
      color.y = 52 * (1 - c) + 94 * c;
      color.z = 148 * (1 - c) + 168 * c;
    } else if (normal_value < v3) {
      real c = (normal_value - v2) / (v3 - v2);
      color.x = 34 * (1 - c) + 29 * c;
      color.y = 94 * (1 - c) + 145 * c;
      color.z = 168 * (1 - c) + 192 * c;
    } else if (normal_value < v4) {
      real c = (normal_value - v3) / (v4 - v3);
      color.x = 29 * (1 - c) + 65 * c;
      color.y = 145 * (1 - c) + 182 * c;
      color.z = 192 * (1 - c) + 196 * c;
    } else if (normal_value < v5) {
      real c = (normal_value - v4) / (v5 - v4);
      color.x = 65 * (1 - c) + 127 * c;
      color.y = 182 * (1 - c) + 205 * c;
      color.z = 196 * (1 - c) + 187 * c;
    } else if (normal_value < v6) {
      real c = (normal_value - v5) / (v6 - v5);
      color.x = 127 * (1 - c) + 199 * c;
      color.y = 205 * (1 - c) + 233 * c;
      color.z = 187 * (1 - c) + 180 * c;
    } else if (normal_value < v7) {
      real c = (normal_value - v6) / (v7 - v6);
      color.x = 199 * (1 - c) + 237 * c;
      color.y = 233 * (1 - c) + 248 * c;
      color.z = 180 * (1 - c) + 177 * c;
    } else {
      real c = (normal_value - v7) / (1 - v7);
      color.x = 237 * (1 - c) + 255 * c;
      color.y = 248 * (1 - c) + 255 * c;
      color.z = 177 * (1 - c) + 217 * c;
    }
    // increase intensity
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
  // rgb(255,255,217)
  // rgb(237,248,177)
  // rgb(199,233,180)
  // rgb(127,205,187)
  // rgb(65,182,196)
  // rgb(29,145,192)
  // rgb(34,94,168)
  // rgb(37,52,148)
  // rgb(8,29,88)
}

__global__ void compute_color_kernel_sand(uchar3 *d_color_array, real *d_plot,
                                          unsigned int width,
                                          unsigned int height, real min,
                                          real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    // normal_value = log(1+normal_value)/log(2.f);
    real v1 = 1.0 / 8.0;
    real v2 = 2.0 / 8.0;
    real v3 = 3.0 / 8.0;
    real v4 = 4.0 / 8.0;
    real v5 = 5.0 / 8.0;
    real v6 = 6.0 / 8.0;
    real v7 = 7.0 / 8.0;
    // compute color
    if (normal_value < v1) {
      real c = normal_value / v1;
      color.x = 102 * (1 - c) + 153 * c;
      color.y = 37 * (1 - c) + 52 * c;
      color.z = 6 * (1 - c) + 4 * c;
    } else if (normal_value < v2) {
      real c = (normal_value - v1) / (v2 - v1);
      color.x = 153 * (1 - c) + 204 * c;
      color.y = 52 * (1 - c) + 76 * c;
      color.z = 4 * (1 - c) + 2 * c;
    } else if (normal_value < v3) {
      real c = (normal_value - v2) / (v3 - v2);
      color.x = 204 * (1 - c) + 236 * c;
      color.y = 76 * (1 - c) + 112 * c;
      color.z = 2 * (1 - c) + 20 * c;
    } else if (normal_value < v4) {
      real c = (normal_value - v3) / (v4 - v3);
      color.x = 236 * (1 - c) + 254 * c;
      color.y = 112 * (1 - c) + 153 * c;
      color.z = 20 * (1 - c) + 41 * c;
    } else if (normal_value < v5) {
      real c = (normal_value - v4) / (v5 - v4);
      color.x = 254 * (1 - c) + 254 * c;
      color.y = 153 * (1 - c) + 196 * c;
      color.z = 41 * (1 - c) + 79 * c;
    } else if (normal_value < v6) {
      real c = (normal_value - v5) / (v6 - v5);
      color.x = 254 * (1 - c) + 254 * c;
      color.y = 196 * (1 - c) + 227 * c;
      color.z = 79 * (1 - c) + 145 * c;
    } else if (normal_value < v7) {
      real c = (normal_value - v6) / (v7 - v6);
      color.x = 254 * (1 - c) + 255 * c;
      color.y = 227 * (1 - c) + 247 * c;
      color.z = 145 * (1 - c) + 188 * c;
    } else {
      real c = (normal_value - v7) / (1 - v7);
      color.x = 255 * (1 - c) + 255 * c;
      color.y = 247 * (1 - c) + 255 * c;
      color.z = 188 * (1 - c) + 229 * c;
    }
    // increase intensity
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
}

__global__ void compute_color_kernel_fire(uchar3 *d_color_array, real *d_plot,
                                          unsigned int width,
                                          unsigned int height, real min,
                                          real max) {
  int index = idx1d();
  if (index < width * height) {
    uchar3 color;
    // local value of the scalar field, resized to be in [0;1]
    real normal_value = (d_plot[index] - min) / (max - min);
    if (normal_value < 0) normal_value = 0;
    if (normal_value > 1) normal_value = 1;
    // compute color
    real v1 = 0.5;
    real v2 = 0.8;
    if (normal_value < v1) {  // black to red
      // red component
      color.x = normal_value / v1 * 255;
      // green coponent
      color.y = normal_value / v1 * 50;
      // blue component
      color.z = 0;
    } else if (normal_value < v2) {  // red to yellow
      // red component
      color.x = 255;
      // green coponent
      color.y = 50 + (normal_value - v1) / (v2 - v1) * 205;
      // blue component
      color.z = 0;
    } else {  // yellow to white
      // red component
      color.x = 255;
      // green coponent
      color.y = 255;
      // blue component
      color.z = (normal_value - v2) / (1.f - v2) * 255;
    }
    // each thread writes one pixel location in the texture (texel)
    d_color_array[index] = color;
  }
}
