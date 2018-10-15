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
  CudaTexturedQuadGeometry(unsigned int width, unsigned int height);
  virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

 protected:
  osg::ref_ptr<opencover::CudaTexture2D> m_texture;
  int m_width, m_height;

  virtual void runCudaKernel(uchar3 *texDevPtr, unsigned int texWidth,
                             unsigned int texHeight) const = 0;
};
