#pragma once

#include <osg/Texture2D>

/**
 * @brief Maps a CUDA generated texture onto an image object, for dynamic
 * texture updates
 *
 */
class CudaTextureSubloadCallback : public osg::Texture2D::SubloadCallback {
 private:
  unsigned int m_width, m_height;
  osg::ref_ptr<osg::Image> m_image;

 public:
  /**
   * @brief Construct a new Cuda Texture Subload Callback object
   *
   * @param width Width of the image
   * @param height Height of the image
   */
  CudaTextureSubloadCallback(unsigned int width, unsigned int height);
  /**
   * @brief Maps a texture to an image object
   *
   * @param texture The texture to draw
   */
  virtual void load(const osg::Texture2D& texture, osg::State&) const;
  /**
   * @brief Subloaded images can have different texture and image sizes
   *
   * @param texture The texture to draw
   */
  virtual void subload(const osg::Texture2D& texture, osg::State&) const;
  /**
   * @brief Set a reference to an image
   *
   * @param image
   */
  void setImage(osg::ref_ptr<osg::Image> image);
};
