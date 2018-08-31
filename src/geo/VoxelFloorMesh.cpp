#include "VoxelFloorMesh.hpp"

VoxelFloorMesh::VoxelFloorMesh(VoxelArray *voxels)
    : m_width(voxels->getSizeX()),
      m_height(voxels->getSizeY()),
      m_voxels(voxels),
      m_transform(new osg::PositionAttitudeTransform()),
      osg::Geometry(*osg::createTexturedQuadGeometry(
                        osg::Vec3(0.0f, 0.0f, 0.0f),
                        osg::Vec3(voxels->getSizeX(), 0.0f, 0.0f),
                        osg::Vec3(0.0f, 0.0f, voxels->getSizeY()),
                        0.0f,
                        0.0f,
                        1.0f,
                        1.0f),
                    osg::CopyOp::SHALLOW_COPY)
{
  m_texture = new osg::Texture2D();

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
  stateset->setTextureAttribute(0, m_texture, osg::StateAttribute::ON);
  stateset->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::ON);

  m_image = new osg::Image();
  m_image->allocateImage(m_width, m_height, 1, GL_RGB, GL_UNSIGNED_BYTE, 1);

  m_texture->setDataVariance(osg::Object::DYNAMIC);
  m_texture->setImage(m_image);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);
  m_transform->addChild(geode);

  GLubyte *ptr;
  int inc = 0;
  ptr = m_image->data();

  float r = 0.1;
  float g = 0.1;
  float b = 0.1;
  int format = m_image->getPixelSizeInBits();
  for (int x = 0; x < m_width; x++)
  {
    for (int y = 0; y < m_height; y++)
    {
      inc = (y * m_width + x) * format / 8; //???????? ?? RGBARGBARGBA... ??? 4
      *(ptr + inc + 0) = (GLubyte)(r * 255.0f);
      *(ptr + inc + 1) = (GLubyte)(g * 255.0f);
      *(ptr + inc + 2) = (GLubyte)(b * 255.0f);
    }
  }

  // //background color
  // col3 bc = col3::black;
  // //line color
  // col3 c0 = col3(253, 254, 210);
  // col3 c1 = col3(36, 49, 52);
  // col3 c2 = col3(48, 91, 116);
  // col3 c3 = col3(104, 197, 214);
  // //conversion factors
  // double Cx = m_voxels->getSizeX() / double(m_texture->getTextureWidth());
  // double Cy = m_voxels->getSizeY() / double(m_texture->getTextureHeight());
  // for (unsigned int i = 0; i < m_texture->getWidth(); i++)
  //   for (unsigned int j = 0; j < m_texture->getHeight(); j++)
  //   {
  //     int x = i * Cx;
  //     int y = j * Cy;
  //     (*ptr)(i, j) = bc;
  //     if (i % 5 == 0)
  //       (*ptr)(i, j) += c1;
  //     if (j % 5 == 0)
  //       (*ptr)(i, j) += c1;
  //     if (((i % 20 < 2) || (i % 20 > 18)) && ((j % 20 < 2) || (j % 20 > 18)))
  //     {
  //       (*ptr)(i, j) += c2;
  //     }
  //     if ((array->isEmptyStrict(x, y, 0)) && (!(array->isEmptyStrict(x + 1, y, 0)) ||
  //                                             !(array->isEmptyStrict(x - 1, y, 0)) ||
  //                                             !(array->isEmptyStrict(x, y + 1, 0)) ||
  //                                             !(array->isEmptyStrict(x, y - 1, 0))))
  //       (*ptr)(i, j) = c0;
  //     //(*tex_floor_)(i,j) = col3((255.0*i)/tex_floor_->getWidth(),0,0);
  //   }
  m_image->dirty();
}

void VoxelFloorMesh::drawImplementation(osg::RenderInfo &renderInfo) const
{
  if (m_texture->getTextureWidth() != m_width || m_texture->getTextureHeight() != m_height)
  {
    m_texture->setResizeNonPowerOfTwoHint(false);
    m_texture->setBorderWidth(0);
    m_texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    m_texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    m_texture->setTextureSize(m_width, m_height);
    m_texture->setSourceFormat(GL_RGB);
    m_texture->setSourceType(GL_UNSIGNED_BYTE);
    m_texture->setInternalFormat(GL_RGB8);
  }

  osg::Geometry::drawImplementation(renderInfo);
}