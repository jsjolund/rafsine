#include "SliceRender.hpp"

SliceRender::SliceRender(D3Q4::Enum axis, unsigned int width,
                         unsigned int height, osg::Vec3i plot3dSize)
    : CudaTexturedQuadGeometry(width, height),
      m_plot3dSize(plot3dSize),
      m_colorScheme(ColorScheme::PARAVIEW),
      m_axis(axis),
      m_min(0),
      m_max(0) {
  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  m_transform = new osg::PositionAttitudeTransform();
  m_transform->addChild(geode);
}

void SliceRender::runCudaKernel(real *plot2dPtr, uchar3 *texDevPtr,
                                unsigned int texWidth,
                                unsigned int texHeight) const {
  dim3 blockSize, gridSize;

  // Configure block size and grid size
  setDims(texWidth * texHeight, BLOCK_SIZE_DEFAULT, &blockSize, &gridSize);

  switch (m_colorScheme) {
    case ColorScheme::BLACK_AND_WHITE:
      compute_color_kernel_black_and_white<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_black_and_white");
      break;
    case ColorScheme::RAINBOW:
      compute_color_kernel_rainbow<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_rainbow");
      break;
    case ColorScheme::DIVERGING:
      compute_color_kernel_diverging<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_diverging");
      break;
    case ColorScheme::OBLIVION:
      compute_color_kernel_Oblivion<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_Oblivion");
      break;
    case ColorScheme::BLUES:
      compute_color_kernel_blues<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_blues");
      break;
    case ColorScheme::SAND:
      compute_color_kernel_sand<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_sand");
      break;
    case ColorScheme::FIRE:
      compute_color_kernel_fire<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_fire");
      break;
    case ColorScheme::PARAVIEW:
      compute_color_kernel_paraview<<<gridSize, blockSize>>>(
          texDevPtr, plot2dPtr, texWidth, texHeight, m_min, m_max);
      CUDA_CHECK_ERRORS("compute_color_kernel_paraview");
      break;
  }
}
