/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#pragma once

#ifdef HAVE_CUDA

#include <osg/State>
#include <osg/Texture2D>

#include "CudaGraphicsResource.hpp"

namespace opencover {

class CudaTexture2D : public osg::Texture2D {
 public:
  CudaTexture2D();

  virtual void apply(osg::State &state) const;

  void resize(osg::State &state, int w, int h, int dataTypeSize);
  void *resourceData();
  void clear();

 protected:
  ~CudaTexture2D();
  osg::ref_ptr<osg::PixelDataBufferObject> m_pbo;
  CudaGraphicsResource m_resource;
  int m_resourceDataSize;
};

}  // namespace opencover

#endif
