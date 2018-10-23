#pragma once

#include <osg/Geometry>
#include <osg/Image>
#include <osg/StateSet>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "CudaTexture2D.hpp"
#include "CudaTextureSubloadCallback.hpp"

/**
 * @brief A 3D quad with a CUDA generated texture on it
 *
 */
class CudaTexturedQuadGeometry : public osg::Geometry {
 private:
  osg::ref_ptr<osg::Image> m_image;

 public:
  /**
   * @brief Construct a new textured quad which can generate dynamics textures
   * with CUDA
   *
   * @param width Width of the image in pixels
   * @param height Height of the image in pixels
   */
  CudaTexturedQuadGeometry(unsigned int width, unsigned int height);
  /**
   * @brief The draw implementation calls the CUDA kernel
   * 
   * @param renderInfo OSG RenderInfo context
   */
  virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

 protected:
  osg::ref_ptr<opencover::CudaTexture2D> m_texture;
  int m_width, m_height;

  /**
   * @brief Callback for drawing the texture
   * 
   * @param texDevPtr CUDA texture pointer
   * @param texWidth Texture width
   * @param texHeight Texture height
   */
  virtual void runCudaKernel(uchar3 *texDevPtr, unsigned int texWidth,
                             unsigned int texHeight) const = 0;
};
