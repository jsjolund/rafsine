/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#include "CudaTexture2D.hpp"

namespace opencover
{

CudaTexture2D::CudaTexture2D() : m_pbo(new osg::PixelDataBufferObject),
                                 m_resourceDataSize(0)
{
  m_pbo->setTarget(GL_PIXEL_UNPACK_BUFFER);
}

CudaTexture2D::~CudaTexture2D()
{
  m_resource.unmap();
}

void CudaTexture2D::apply(osg::State &state) const
{
  osg::GLBufferObject *glBufferObject = m_pbo->getGLBufferObject(state.getContextID());
  if (glBufferObject == nullptr)
  {
    osg::Texture2D::apply(state);

    return;
  }
  const_cast<CudaGraphicsResource *>(&m_resource)->unmap();

  osg::GLExtensions *ext = osg::GLExtensions::Get(state.getContextID(), true);

  ext->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBufferObject->getGLObjectID());

  osg::Texture2D::apply(state);

  ext->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  const_cast<CudaGraphicsResource *>(&m_resource)->map();
}

void CudaTexture2D::resize(osg::State &state, int w, int h, int dataTypeSize)
{
  m_resource.unmap();

  m_resourceDataSize = w * h * dataTypeSize;

  m_pbo->setDataSize(m_resourceDataSize);
  m_pbo->compileBuffer(state);

  m_resource.register_buffer(m_pbo->getGLBufferObject(state.getContextID())->getGLObjectID(), cudaGraphicsRegisterFlagsWriteDiscard);

  m_resource.map();
}

void *CudaTexture2D::resourceData()
{
  return m_resource.dev_ptr();
}

void CudaTexture2D::clear()
{
  if (resourceData() == nullptr)
    return;

  cudaMemset(resourceData(), 0, m_resourceDataSize);
}

} // namespace opencover

#endif