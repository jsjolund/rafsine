#pragma once

#include <osg/Image>
#include <osg/Geometry>
#include <osg/StateSet>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "CudaTexture2D.hpp"
#include "CudaTextureSubloadCallback.hpp"

class CudaTexturedQuadGeometry : public osg::Geometry
{
private:
  osg::Image *m_image;

public:
  CudaTexturedQuadGeometry(unsigned int width, unsigned int height);
  virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

protected:
  opencover::CudaTexture2D *m_texture;
  unsigned int m_width, m_height;

  virtual void runCudaKernel() const = 0;
};