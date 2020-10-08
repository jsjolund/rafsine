#pragma once

#include <osg/Geometry>
#include <osg/Image>
#include <osg/StateSet>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>

#include "CudaTexture2D.hpp"
#include "CudaTextureSubloadCallback.hpp"
#include "CudaUtils.hpp"

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
  virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

  inline real_t* gpu_ptr() { return m_plot2dPtr; }

 protected:
  thrust::device_vector<real_t> m_plot2d;
  real_t* m_plot2dPtr;
  osg::ref_ptr<opencover::CudaTexture2D>
      m_texture;  //!< Pointer to the cuda texture
  int m_width;    //!< Texture width
  int m_height;   //!< Texture height

  /**
   * @brief Callback for drawing the texture
   *
   * @param texDevPtr CUDA texture pointer
   * @param texWidth Texture width
   * @param texHeight Texture height
   */
  virtual void runCudaKernel(real_t* plot2dPtr,
                             uchar3* texDevPtr,
                             unsigned int texWidth,
                             unsigned int texHeight) const = 0;
};
