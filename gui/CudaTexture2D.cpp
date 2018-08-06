/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#include "CudaTexture2D.hpp"

#include <iostream>
#include <unistd.h>
#include <stdio.h>
namespace opencover
{

CudaTexture2D::CudaTexture2D() : pbo_(new osg::PixelDataBufferObject),
                                 resourceDataSize_(0)
{
  pbo_->setTarget(GL_PIXEL_UNPACK_BUFFER);

  resource_.map();
}

CudaTexture2D::~CudaTexture2D()
{
  resource_.unmap();
}

void CudaTexture2D::apply(osg::State &state) const
{ // std::cout << "apply" << std::endl;
  pbo_->dirty();
  osg::GLExtensions *ext = osg::GLExtensions::Get(state.getContextID(), true);
  osg::GLBufferObject *glBufferObject = pbo_->getGLBufferObject(state.getContextID());
  if (glBufferObject == nullptr)
  {
    osg::Texture2D::apply(state);

    return;
  }
  const_cast<CudaGraphicsResource *>(&resource_)->unmap();

  ext->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBufferObject->getGLObjectID());

  osg::Texture2D::apply(state);

  ext->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  const_cast<CudaGraphicsResource *>(&resource_)->map();
}

void CudaTexture2D::resize(osg::State *state, int w, int h, int dataTypeSize)
{
  resource_.unmap();

  resourceDataSize_ = w * h * dataTypeSize;

  pbo_->setDataSize(resourceDataSize_);
  pbo_->compileBuffer(*state);

  resource_.register_buffer(pbo_->getGLBufferObject(state->getContextID())->getGLObjectID(), cudaGraphicsRegisterFlagsWriteDiscard);

  resource_.map();
}

void *CudaTexture2D::resourceData()
{
  return resource_.dev_ptr();
}

void CudaTexture2D::clear()
{
  if (resourceData() == nullptr)
    return;

  cudaMemset(resourceData(), 0, resourceDataSize_);
}

} // namespace opencover

#endif