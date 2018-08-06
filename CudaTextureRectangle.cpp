/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

// #ifdef HAVE_CUDA

// #include <GL/glew.h>

#include "CudaTextureRectangle.hpp"

#include <iostream>
#include <unistd.h>
#include <stdio.h>
/// check if there is any error and display the details if there are some
inline void cuda_check_errors(const char *func_name)
{
  cudaError_t cerror = cudaGetLastError();
  if (cerror != cudaSuccess)
  {
    char host[256];
    gethostname(host, 256);
    printf("%s: CudaError: %s (on %s)\n", func_name, cudaGetErrorString(cerror), host);
    // exit(1);
  }
}

namespace opencover
{

CudaTextureRectangle::CudaTextureRectangle() : osg::Texture2D(),
                                               pbo_(new osg::PixelDataBufferObject),
                                               resourceDataSize_(0)
{
  pbo_->setTarget(GL_PIXEL_UNPACK_BUFFER);

  resource_.map();
}

CudaTextureRectangle::~CudaTextureRectangle()
{
  resource_.unmap();
}

void CudaTextureRectangle::apply(osg::State &state) const
{
  osg::GLBufferObject *glBufferObject = pbo_->getGLBufferObject(state.getContextID());
  if (glBufferObject == nullptr)
  {
    osg::Texture2D::apply(state);

    return;
  }
  unsigned int ctxtID = state.getContextID();
  osg::GLExtensions *ext = osg::GLExtensions::Get(state.getContextID(), true);

  const_cast<CudaGraphicsResource *>(&resource_)->unmap();
  GLuint bufferObj = glBufferObject->getGLObjectID();
  ext->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBufferObject->getGLObjectID());

  osg::Texture2D::apply(state);

  ext->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  const_cast<CudaGraphicsResource *>(&resource_)->map();
}

void CudaTextureRectangle::resize(osg::State *state, int w, int h, int dataTypeSize)
{
  resource_.unmap();

  resourceDataSize_ = w * h * dataTypeSize;

  pbo_->setDataSize(resourceDataSize_);
  pbo_->dirty();
  pbo_->compileBuffer(*state);

  GLuint id = pbo_->getGLBufferObject(state->getContextID())->getGLObjectID();

  resource_.register_buffer(id, cudaGraphicsRegisterFlagsWriteDiscard);
  // resource_.register_buffer(id, cudaGraphicsRegisterFlagsSurfaceLoadStore);
  cuda_check_errors("after register_buffer");
  resource_.map();
}

void *CudaTextureRectangle::resourceData()
{
  return resource_.dev_ptr();
}

void CudaTextureRectangle::clear()
{
  if (resourceData() == nullptr)
    return;

  cudaMemset(resourceData(), 0, resourceDataSize_);
}

} // namespace opencover

// #endif
