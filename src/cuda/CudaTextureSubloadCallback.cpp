
#include "CudaTextureSubloadCallback.hpp"

void CudaTextureSubloadCallback::setImage(osg::ref_ptr<osg::Image> image)
{
  m_image = image;
  if (m_image)
    m_image->setPixelBufferObject(0);
}

CudaTextureSubloadCallback::CudaTextureSubloadCallback(unsigned int width,
                                                       unsigned int height)
    : osg::Texture2D::SubloadCallback(),
      m_width(width),
      m_height(height) {}

void CudaTextureSubloadCallback::load(const osg::Texture2D &,
                                      osg::State &) const
{
  glTexImage2D(
      GL_TEXTURE_2D,
      0,
      m_image->getInternalTextureFormat(),
      (int)m_width,
      (int)m_height,
      0,
      m_image->getPixelFormat(),
      m_image->getDataType(),
      0x0);
}

void CudaTextureSubloadCallback::subload(const osg::Texture2D &,
                                         osg::State &) const
{
  if (m_image.valid())
  {
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0,
        m_image->s(), m_image->t(),
        m_image->getPixelFormat(),
        m_image->getDataType(),
        m_image->getDataPointer());
  }
}