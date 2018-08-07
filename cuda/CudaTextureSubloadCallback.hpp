#pragma once

#include <osg/Texture2D>
class CudaTextureSubloadCallback : public osg::Texture2D::SubloadCallback
{
public:
  CudaTextureSubloadCallback(osg::Texture2D *tex, unsigned int width, unsigned int height);
  virtual void load(const osg::Texture2D &m_texture, osg::State &state) const;
  virtual void subload(const osg::Texture2D &m_texture, osg::State &state) const;
  void setImage(osg::Image *image);

private:
  unsigned int m_width, m_height;
  osg::ref_ptr<osg::Image> m_image;
};