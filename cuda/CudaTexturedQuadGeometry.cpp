
#include "CudaTexturedQuadGeometry.hpp"

CudaTexturedQuadGeometry::CudaTexturedQuadGeometry(unsigned int width, unsigned int height)
    : m_width(width),
      m_height(height),
      osg::Geometry(*osg::createTexturedQuadGeometry(
                        osg::Vec3(0.0f, 0.0f, 0.0f),
                        osg::Vec3(width, 0.0f, 0.0f),
                        osg::Vec3(0.0f, 0.0f, height),
                        0.0f,
                        0.0f,
                        1.0f,
                        1.0f),
                    osg::CopyOp::SHALLOW_COPY)
{
  m_texture = new opencover::CudaTexture2D();

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
  stateset->setTextureAttribute(0, m_texture, osg::StateAttribute::ON);
  stateset->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::ON);
  setUseDisplayList(false);

  m_image = new osg::Image();
  m_image->allocateImage(m_width, m_height, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1);
  CudaTextureSubloadCallback *cb = new CudaTextureSubloadCallback(m_texture, m_width, m_height);
  m_texture->setSubloadCallback(cb);
  cb->setImage(m_image);
  m_texture->setImage(m_image);
  m_texture->setDataVariance(osg::Object::DYNAMIC);
}

void CudaTexturedQuadGeometry::drawImplementation(osg::RenderInfo &renderInfo) const
{

  if (m_texture->getTextureWidth() != m_width || m_texture->getTextureHeight() != m_height)
  {
    m_texture->setResizeNonPowerOfTwoHint(false);
    m_texture->setBorderWidth(0);
    m_texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    m_texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    m_texture->setTextureSize(m_width, m_height);
    m_texture->setSourceFormat(GL_RGBA);
    m_texture->setSourceType(GL_UNSIGNED_BYTE);
    m_texture->setInternalFormat(GL_RGBA8);
    m_texture->resize(*renderInfo.getState(), m_width, m_height, 4);
  }

  runCudaKernel();

  m_image->dirty();

  osg::Geometry::drawImplementation(renderInfo);
}